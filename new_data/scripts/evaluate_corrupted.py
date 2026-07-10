import os
import json
import argparse
from collections import defaultdict

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from prompt_utils import (
    MAX_OCR_CHARS,
    IMAGE_MIN_PIXELS,
    IMAGE_MAX_PIXELS,
    REFUSAL_PHRASES,
    get_all_page_ids,
    extract_ocr_from_layout_analysis as extract_ocr,
    build_prompt,
    is_refusal,
    load_and_resize_image,
)

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# MAX_OCR_CHARS, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS, REFUSAL_PHRASES,
# get_all_page_ids, extract_ocr (-> extract_ocr_from_layout_analysis),
# build_prompt, and is_refusal now all live in prompt_utils.py -- the same
# module fine_tune.py imports from. Do not redefine any of these here;
# that's exactly how the 2000-vs-8000 MAX_OCR_CHARS mismatch happened.


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(model, processor, device, image: Image.Image, question: str, ocr_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": build_prompt(question, ocr_text)},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text], images=image_inputs, return_tensors="pt", padding=False
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """
    AccP = mean of per-question (correct_pages / total_pages)
    AccD = fraction of questions where all pages are correctly answered
    """
    if not results:
        return {}

    acc_p = sum(r["page_accuracy"] for r in results) / len(results)
    acc_d = sum(1 for r in results if r["all_pages_correct"]) / len(results)
    hall  = sum(r["hallucinated_pages"] for r in results) / sum(r["total_pages"] for r in results)

    # Per-item binary for F1 — a question is "correct" if model refuses on all pages
    y_true = [1] * len(results)  # all items are unanswerable
    y_pred = [1 if r["all_pages_correct"] else 0 for r in results]

    tp = sum(y_pred)
    fn = len(results) - tp
    fp = 0  # no answerable items in this benchmark
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "AccP": acc_p,
        "AccD": acc_d,
        "hallucination_rate": hall,
        "total": len(results),
        "unanswerable": {
            "f1": f1, "precision": precision, "recall": recall,
            "support": len(results), "tp": tp, "fp": fp, "fn": fn,
        },
    }


def compute_complexity_breakdown(results: list) -> dict:
    buckets = defaultdict(list)
    for r in results:
        buckets[str(r.get("complexity", "?"))].append(r)
    return {
        level: compute_metrics(items)
        for level, items in sorted(buckets.items())
    }


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    # 1. Load model — supports base model, full fine-tuned model, or LoRA adapter
    base_model_id = BASE_MODEL_NAME
    is_lora = False
    if args.model_dir:
        if os.path.exists(os.path.join(args.model_dir, "adapter_config.json")):
            is_lora = True
            print(f"Detected LoRA adapter at: {args.model_dir}")
        else:
            base_model_id = args.model_dir
            print(f"Loading full model from: {args.model_dir}")
    else:
        print(f"Loading base model: {base_model_id}")

    processor = AutoProcessor.from_pretrained(
        base_model_id, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if is_lora:
        from peft import PeftModel
        print(f"Applying LoRA adapter from: {args.model_dir}")
        model = PeftModel.from_pretrained(model, args.model_dir)
        model = model.merge_and_unload()

    model.eval()
    device = next(model.parameters()).device
    model_id = args.model_dir or base_model_id
    print(f"Model loaded on {device}.\n")

    # 2. Load data
    print(f"Loading: {args.data_file}")
    with open(args.data_file) as f:
        raw = json.load(f)
    items = raw["corrupted_questions"]
    print(f"  {len(items)} questions")

    if args.complexity:
        items = [x for x in items if x.get("complexity") == args.complexity]
        print(f"  Filtered to complexity={args.complexity}: {len(items)} items")

    if args.max_samples:
        items = items[: args.max_samples]
        print(f"  → limited to {len(items)} samples")

    # Count total page inferences
    total_pages = sum(len(get_all_page_ids(item)) for item in items)
    print(f"  {total_pages} total page inferences to run\n")

    # 3. Inference — ALL pages per question
    results   = []
    skipped_q = 0
    page_num  = 0

    for i, item in enumerate(items):
        question = item["corrupted_question"]
        page_ids = get_all_page_ids(item)

        if not page_ids:
            print(f"  WARNING: no pages for item {i}, skipping")
            skipped_q += 1
            continue

        page_results = []
        skip_item    = False

        for page_id in page_ids:
            page_num += 1
            if page_num % 100 == 0:
                print(f"  [page {page_num}/{total_pages}  question {i+1}/{len(items)}] ...")

            img_path = os.path.join(args.images_dir, page_id)
            try:
                image = load_and_resize_image(img_path)
            except Exception as e:
                print(f"  WARNING: skipping page {page_id} for item {i} — {e}")
                skip_item = True
                break

            # truncation to MAX_OCR_CHARS is already applied inside
            # extract_ocr (prompt_utils._sort_and_join) -- do not re-truncate
            # here, it's redundant and risks drifting out of sync if
            # MAX_OCR_CHARS ever changes again.
            ocr_text = extract_ocr(item, page_id)

            generated = run_inference(model, processor, device, image, question, ocr_text)
            refused   = is_refusal(generated)

            page_results.append({
                "page_id":   page_id,
                "predicted": generated,
                "refused":   refused,
                "correct":   refused,  # all pages should be refused (corrupted question)
            })

        if skip_item:
            skipped_q += 1
            continue

        correct_pages    = sum(p["correct"] for p in page_results)
        total_pages_q    = len(page_results)
        hallucinated     = total_pages_q - correct_pages
        page_acc         = correct_pages / total_pages_q
        all_correct      = correct_pages == total_pages_q

        results.append({
            "index":              i,
            "original_question":  item["original_question"],
            "corrupted_question": question,
            "complexity":         item.get("complexity"),
            "entity_type":        item.get("entity_type"),
            "is_corrupted":       item.get("is_corrupted", True),
            "total_pages":        total_pages_q,
            "correct_pages":      correct_pages,
            "hallucinated_pages": hallucinated,
            "page_accuracy":      page_acc,
            "all_pages_correct":  all_correct,
            "page_details":       page_results,
        })

    # 4. Metrics
    metrics   = compute_metrics(results)
    breakdown = compute_complexity_breakdown(results)

    print("\n" + "=" * 60)
    print("CORRUPTED BENCHMARK RESULTS  (paper methodology)")
    print("=" * 60)
    label = f"{model_id}" + (" (LoRA)" if is_lora else "")
    print(f"  Model            : {label}")
    print(f"  Data             : {args.data_file}")
    print(f"  Questions        : {metrics['total']}  ({skipped_q} skipped)")
    print(f"  Pages tested     : {sum(r['total_pages'] for r in results)}")
    print()
    print(f"  AccP  (page-level)    : {metrics['AccP']:.4f}")
    print(f"  AccD  (doc-level)     : {metrics['AccD']:.4f}")
    print(f"  Hallucination Rate    : {metrics['hallucination_rate']:.4f}   (pages where model gave answer)")
    print()
    ua = metrics["unanswerable"]
    print(f"  F1  (all-pages correct)  : {ua['f1']:.4f}")
    print(f"  TP / FN                  : {ua['tp']} / {ua['fn']}")
    print()
    print("  Breakdown by complexity:")
    for level, m in breakdown.items():
        if not m:
            continue
        ua_b = m.get("unanswerable", {})
        print(f"    C{level}  n={m['total']:3d}  "
              f"AccP={m['AccP']:.3f}  AccD={m['AccD']:.3f}  "
              f"Hallucination={m['hallucination_rate']:.3f}  F1={ua_b.get('f1', 0):.3f}")
    print()
    print("  Paper reference (Qwen2.5-VL-7B, DUDE verified 187):")
    print("    AccP=0.835  AccD=0.460")
    print("    C1: AccP=0.843  C2: AccP=0.847  C3: AccP=0.731")
    print("=" * 60)

    # 5. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model":              label,
            "data_file":          args.data_file,
            "prompt_version":     "paper_explicit",
            "evaluation":         "all_pages_per_question",
            "AccP":               metrics["AccP"],
            "AccD":               metrics["AccD"],
            "hallucination_rate": metrics["hallucination_rate"],
            "metrics":            metrics,
            "complexity_breakdown": breakdown,
            "predictions":        results,
        }, f, indent=2)
    print(f"\nFull results saved to: {args.output}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Corrupted benchmark evaluation for VRD-UQA")
    parser.add_argument("--data_file",   required=True,
                        help="Path to DUDE_verified.json")
    parser.add_argument("--images_dir",  required=True,
                        help="Path to DUDE train images dir")
    parser.add_argument("--output",      default="corrupted_results.json")
    parser.add_argument("--model_dir",   default=None,
                        help="Path to fine-tuned model checkpoint. Omit to use base model.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Evaluate only the first N questions (quick sanity check)")
    parser.add_argument("--complexity",  type=int, default=None, choices=[1, 2, 3],
                        help="Filter to a single complexity level")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()