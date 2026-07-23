import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from prompt_utils import (
    IMAGE_MIN_PIXELS,
    IMAGE_MAX_PIXELS,
    MAX_OCR_CHARS,
    load_patch_file,
    extract_ocr_from_patch_file,
    patch_file_path,
    page_image_path,
    load_and_resize_image,
    compute_adaptive_page_pixels,
    build_prompt,
    is_refusal,
)

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PATCH_NAME_RE = re.compile(r"^([0-9a-f]{32})_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def build_docid_page_index(patch_dir: Path):
    index = defaultdict(list)
    for f in patch_dir.glob("*.json"):
        m = PATCH_NAME_RE.match(f.name)
        if not m:
            continue
        doc_id, page = m.group(1), int(m.group(2))
        index[doc_id].append(page)
    for doc_id in index:
        index[doc_id].sort()
    return index


def get_answer_page(item: dict):
    boxes = item.get("answers_page_bounding_boxes") or []
    if not boxes or not boxes[0]:
        return None
    return boxes[0][0]["page"]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_answer_correct(generated: str, gold_answers: list, gold_variants: list) -> bool:
    """Normalized exact match OR substring containment, checked against
    all gold answers and variants. Not full ANLS -- a reasonable first-pass
    heuristic, not a precision scoring method."""
    gen_norm = normalize_text(generated)
    if not gen_norm:
        return False
    candidates = [a for a in (gold_answers or [])] + [v for v in (gold_variants or [])]
    for cand in candidates:
        cand_norm = normalize_text(str(cand))
        if not cand_norm:
            continue
        if gen_norm == cand_norm or cand_norm in gen_norm or gen_norm in cand_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate_corrupted.py)
# ---------------------------------------------------------------------------
def load_model(model_dir):
    base_model_id = BASE_MODEL_NAME
    is_lora = False
    if model_dir:
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            is_lora = True
            print(f"Detected LoRA adapter at: {model_dir}")
        else:
            base_model_id = model_dir
            print(f"Loading full model from: {model_dir}")
    else:
        print(f"Loading base model: {base_model_id}")

    processor = AutoProcessor.from_pretrained(
        base_model_id, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    if is_lora:
        from peft import PeftModel
        print(f"Applying LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}.\n")
    label = f"{base_model_id}" + (" (LoRA)" if is_lora else "")
    return model, processor, device, label


@torch.no_grad()
def run_inference(model, processor, device, image, question, ocr_text, max_new_tokens=64):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": IMAGE_MIN_PIXELS, "max_pixels": IMAGE_MAX_PIXELS},
            {"type": "text", "text": build_prompt(question, ocr_text)},
        ],
    }]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt").to(device)

    gen_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )
    gen_text = processor.tokenizer.decode(
        gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return gen_text.strip()


@torch.no_grad()
def run_inference_multi(model, processor, device, doc_id, pages, patch_dir, image_dir,
                         question, max_images_per_doc=20, multipage_target_pages=1,
                         max_new_tokens=64):
    """
    Multi-page inference for answerable items with no verified page
    (abstractive/list types). Safe at EVAL time (no backprop, no
    hallucination-training risk) unlike the equivalent training-time
    recovery -- we're just measuring the model's actual output when shown
    the whole document, same as a real user would see it. Mirrors
    fine_tune.py's VRDUQADataset multi-page path: hard page cap first,
    then adaptive per-page resolution within that cap.
    """
    capped_pages = sorted(pages)[:max_images_per_doc]
    per_page_pixels = compute_adaptive_page_pixels(
        n_pages=len(capped_pages), target_pages_equivalent=multipage_target_pages,
        base_max_pixels=IMAGE_MAX_PIXELS,
    )

    images, ocr_pieces = [], []
    for page in capped_pages:
        img_path = page_image_path(image_dir, doc_id, page)
        images.append(load_and_resize_image(img_path, max_pixels=per_page_pixels))
        patch_path = patch_file_path(patch_dir, doc_id, page)
        if patch_path.exists():
            patches = load_patch_file(patch_path)
            page_text = extract_ocr_from_patch_file(patches)
            if page_text:
                ocr_pieces.append(f"[Page {page}]\n{page_text}")
    ocr_text = "\n\n".join(ocr_pieces)
    if len(ocr_text) > MAX_OCR_CHARS:
        ocr_text = ocr_text[:MAX_OCR_CHARS]

    content = [
        {"type": "image", "image": img, "min_pixels": per_page_pixels, "max_pixels": per_page_pixels}
        for img in images
    ]
    content.append({"type": "text", "text": build_prompt(question, ocr_text)})
    messages = [{"role": "user", "content": content}]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt").to(device)

    gen_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )
    gen_text = processor.tokenizer.decode(
        gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return gen_text.strip(), capped_pages


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    model, processor, device, label = load_model(args.model_dir)

    print(f"Loading: {args.data_file}")
    with open(args.data_file) as f:
        raw = json.load(f)
    items = raw["data"]
    print(f"  {len(items)} questions")

    if args.max_samples:
        items = items[: args.max_samples]
        print(f"  -> limited to {len(items)} samples")

    patch_dir = Path(args.patch_dir)
    image_dir = Path(args.image_dir)
    docid_pages = build_docid_page_index(patch_dir)

    answerable_results = []
    unanswerable_results = []
    skipped = defaultdict(int)

    for i, item in enumerate(items):
        doc_id = item["docId"]
        is_answerable = item["answer_type"] != "not-answerable"
        available_pages = docid_pages.get(doc_id)

        if not available_pages:
            skipped["no_patch_coverage"] += 1
            continue

        if is_answerable:
            page = get_answer_page(item)

            if page is None:
                # No verified page (abstractive/list). Skip only if
                # explicitly requested; otherwise evaluate via multi-page.
                if args.skip_unverified_answerable:
                    skipped["answerable_missing_page"] += 1
                    continue
                try:
                    generated, used_pages = run_inference_multi(
                        model, processor, device, doc_id, available_pages,
                        patch_dir, image_dir, item["question"],
                        max_images_per_doc=args.max_images_per_doc,
                        multipage_target_pages=args.multipage_target_pages,
                    )
                except Exception as e:
                    print(f"  WARNING: skipping multi-page item {i} ({doc_id}) -- {e}")
                    skipped["multipage_inference_failed"] += 1
                    continue
                refused = is_refusal(generated)
                correct = (not refused) and is_answer_correct(
                    generated, item.get("answers"), item.get("answers_variants")
                )
                answerable_results.append({
                    "questionId": item["questionId"], "docId": doc_id, "page": used_pages,
                    "question": item["question"], "gold_answers": item.get("answers"),
                    "generated": generated, "refused": refused, "correct": correct,
                    "answer_type": item["answer_type"], "eval_method": "multipage",
                })
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [item {i+1}/{len(items)}] doc={doc_id} pages={used_pages} "
                          f"(multipage) refused={refused} generated={generated[:80]!r}")
                continue

            if page not in available_pages:
                skipped["answer_page_not_in_patch_index"] += 1
                continue
            pages_to_test = [page]
        else:
            pages_to_test = available_pages

        page_preds = []
        for page in pages_to_test:
            img_path = page_image_path(image_dir, doc_id, page)
            patch_path = patch_file_path(patch_dir, doc_id, page)
            try:
                image = load_and_resize_image(img_path)
                ocr_text = ""
                if patch_path.exists():
                    patches = load_patch_file(patch_path)
                    ocr_text = extract_ocr_from_patch_file(patches)
                generated = run_inference(model, processor, device, image, item["question"], ocr_text)
            except Exception as e:
                print(f"  WARNING: skipping page {page} for item {i} ({doc_id}) -- {e}")
                continue

            refused = is_refusal(generated)
            page_preds.append({"page": page, "generated": generated, "refused": refused})

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [item {i+1}/{len(items)}] doc={doc_id} page={page} "
                      f"refused={refused} generated={generated[:80]!r}")

        if not page_preds:
            skipped["all_pages_failed"] += 1
            continue

        if is_answerable:
            # single page -- correct if NOT refused AND text matches gold
            pred = page_preds[0]
            correct = (not pred["refused"]) and is_answer_correct(
                pred["generated"], item.get("answers"), item.get("answers_variants")
            )
            answerable_results.append({
                "questionId": item["questionId"], "docId": doc_id, "page": pred["page"],
                "question": item["question"], "gold_answers": item.get("answers"),
                "generated": pred["generated"], "refused": pred["refused"], "correct": correct,
                "answer_type": item["answer_type"], "eval_method": "single_page",
            })
        else:
            all_refused = all(p["refused"] for p in page_preds)
            frac_refused = sum(p["refused"] for p in page_preds) / len(page_preds)
            hallucinated_pages = sum(1 for p in page_preds if not p["refused"])
            unanswerable_results.append({
                "questionId": item["questionId"], "docId": doc_id,
                "question": item["question"], "total_pages": len(page_preds),
                "all_pages_correct": all_refused, "page_accuracy": frac_refused,
                "hallucinated_pages": hallucinated_pages, "page_details": page_preds,
            })

    # ── Metrics ──
    extractive_results = [r for r in answerable_results if r["eval_method"] == "single_page"]
    ans_metrics_extractive = compute_answerable_metrics(extractive_results)
    ans_metrics_full = compute_answerable_metrics(answerable_results)
    unans_metrics = compute_unanswerable_metrics(unanswerable_results)

    print("\n" + "=" * 60)
    print("BALANCED TEST SET RESULTS")
    print("=" * 60)
    print(f"  Model             : {label}")
    print(f"  Data              : {args.data_file}")
    print(f"  Skipped           : {dict(skipped)}")
    print()
    print(f"  ANSWERABLE -- extractive only, page-verified (n={ans_metrics_extractive['total']})")
    print(f"    Accuracy (correct answer, not refused) : {ans_metrics_extractive['accuracy']:.4f}")
    print(f"    Refusal rate (wrongly refused)          : {ans_metrics_extractive['refusal_rate']:.4f}")
    print()
    print(f"  ANSWERABLE -- FULL dataset, incl. multi-page abstractive (n={ans_metrics_full['total']})")
    print(f"    Accuracy (correct answer, not refused) : {ans_metrics_full['accuracy']:.4f}")
    print(f"    Refusal rate (wrongly refused)          : {ans_metrics_full['refusal_rate']:.4f}")
    print()
    print(f"  NOT-ANSWERABLE  (n={unans_metrics['total']})")
    print(f"    AccP (page-level)     : {unans_metrics['AccP']:.4f}")
    print(f"    AccD (doc-level)      : {unans_metrics['AccD']:.4f}")
    print(f"    Hallucination rate    : {unans_metrics['hallucination_rate']:.4f}")
    print()
    print(f"  MACRO F1 (extractive-only accuracy vs unanswerable-AccD): "
          f"{macro_f1(ans_metrics_extractive['accuracy'], unans_metrics['AccD']):.4f}")
    print(f"  MACRO F1 (FULL answerable accuracy vs unanswerable-AccD): "
          f"{macro_f1(ans_metrics_full['accuracy'], unans_metrics['AccD']):.4f}")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": label, "data_file": args.data_file,
            "skipped": dict(skipped),
            "answerable_metrics_extractive_only": ans_metrics_extractive,
            "answerable_metrics_full": ans_metrics_full,
            "unanswerable_metrics": unans_metrics,
            "macro_f1_extractive_only": macro_f1(ans_metrics_extractive["accuracy"], unans_metrics["AccD"]),
            "macro_f1_full": macro_f1(ans_metrics_full["accuracy"], unans_metrics["AccD"]),
            "answerable_predictions": answerable_results,
            "unanswerable_predictions": unanswerable_results,
        }, f, indent=2)
    print(f"\nFull results saved to: {args.output}")


def compute_answerable_metrics(results):
    if not results:
        return {"total": 0, "accuracy": 0.0, "refusal_rate": 0.0}
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    refused = sum(1 for r in results if r["refused"])
    return {"total": n, "accuracy": correct / n, "refusal_rate": refused / n}


def compute_unanswerable_metrics(results):
    if not results:
        return {"total": 0, "AccP": 0.0, "AccD": 0.0, "hallucination_rate": 0.0}
    n = len(results)
    acc_p = sum(r["page_accuracy"] for r in results) / n
    acc_d = sum(1 for r in results if r["all_pages_correct"]) / n
    total_pages = sum(r["total_pages"] for r in results)
    hall = sum(r["hallucinated_pages"] for r in results) / total_pages if total_pages else 0.0
    return {"total": n, "AccP": acc_p, "AccD": acc_d, "hallucination_rate": hall}


def macro_f1(ans_acc, unans_acc):
    if (ans_acc + unans_acc) == 0:
        return 0.0
    return 2 * ans_acc * unans_acc / (ans_acc + unans_acc)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on annotations_balanced_test.json")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--patch_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=None,
                         help="LoRA adapter dir or full model dir. Omit for zero-shot base model.")
    parser.add_argument("--output", default="balanced_test_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_unverified_answerable", action="store_true",
                         help="Restore old behavior: skip abstractive/list answerable items "
                              "with no page bbox instead of evaluating via multi-page.")
    parser.add_argument("--max_images_per_doc", type=int, default=20,
                         help="Hard cap on pages shown per multi-page abstractive item.")
    parser.add_argument("--multipage_target_pages", type=int, default=1,
                         help="Adaptive resolution target for multi-page eval -- matches "
                              "the value found safe via probe_multipage_memory.py for "
                              "training; eval has no backward pass so is less memory-"
                              "constrained, but kept consistent for comparability.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()