import os
import re
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel

from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

from prompt_utils import (
    MAX_OCR_CHARS,
    IMAGE_MIN_PIXELS,
    IMAGE_MAX_PIXELS,
    load_patch_file,
    extract_ocr_from_patch_file,
    patch_file_path,
    page_image_path,
    load_and_resize_image,
    build_messages,
    build_messages_multi,
    build_target,
    is_refusal,
)

MAX_SEQ_LENGTH = 8192  # separate from MAX_OCR_CHARS; tokenized-sequence cap
PATCH_NAME_RE = re.compile(r"^([0-9a-f]{32})_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Data construction (folded in from the former build_dataset.py)
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


def pick_random_page(question_id: str, available_pages: list, seed: int) -> int:
    """Deterministic pseudo-random page pick, seeded per questionId --
    reproducible across runs, but not fixed to page 0 across the dataset."""
    rng = random.Random(f"{seed}-{question_id}")
    return rng.choice(available_pages)


def get_answer_page(item: dict):
    boxes = item.get("answers_page_bounding_boxes") or []
    if not boxes or not boxes[0]:
        return None
    return boxes[0][0]["page"]


def build_and_balance_records(annotations_path: str, patch_dir: Path, seed: int,
                               balance_mode: str = "downsample_na"):
    """
    Builds records from ONE annotations file (join + page assignment +
    multi-page recovery + balance), with NO train/val split applied.
    Returns (records, summary_dict).

    balance_mode:
      "downsample_na" -> randomly drop not-answerable records down to
                          match the answerable count (default, safe).
      "none"          -> keep the natural (imbalanced) counts as-is.
    """
    with open(annotations_path) as f:
        raw = json.load(f)
    data = raw["data"]

    docid_pages = build_docid_page_index(patch_dir)

    records = []
    dropped = defaultdict(int)

    for item in data:
        doc_id = item["docId"]
        question_id = item["questionId"]
        is_answerable = item["answer_type"] != "not-answerable"

        available_pages = docid_pages.get(doc_id)
        if not available_pages:
            dropped["no_patch_coverage"] += 1
            continue

        if is_answerable:
            page = get_answer_page(item)
            if page is None:
                # No page bbox (e.g. "abstractive" answer_type). Recovered
                # as a multi-page example over ALL of the doc's available
                # pages -- no cap, per decision on 2026-07-07. Long
                # documents may fail to encode at train time (see
                # VRDUQADataset._build_example's try/except); such
                # examples are skipped with a warning rather than crashing
                # the run, not silently truncated to fewer pages.
                if not available_pages:
                    dropped["answerable_missing_page"] += 1
                    continue
                answer = item["answers"][0] if item.get("answers") else None
                if not answer:
                    dropped["answerable_missing_page"] += 1
                    continue
                records.append({
                    "example_id": question_id,
                    "doc_id": doc_id,
                    "page": sorted(available_pages),
                    "is_multipage": True,
                    "question": item["question"],
                    "answer": answer,
                    "is_answerable": True,
                    "answer_type": item["answer_type"],
                })
                continue
            if page not in available_pages:
                dropped["answer_page_not_in_patch_index"] += 1
                continue
            answer = item["answers"][0] if item.get("answers") else None
        else:
            page = pick_random_page(question_id, available_pages, seed)
            answer = None

        records.append({
            "example_id": question_id,
            "doc_id": doc_id,
            "page": page,
            "question": item["question"],
            "answer": answer,
            "is_answerable": is_answerable,
            "answer_type": item["answer_type"],
        })

    ans_records = [r for r in records if r["is_answerable"]]
    unans_records = [r for r in records if not r["is_answerable"]]

    rng = random.Random(seed)
    rng.shuffle(ans_records)
    rng.shuffle(unans_records)

    pre_balance_counts = {"answerable": len(ans_records), "not_answerable": len(unans_records)}

    if balance_mode == "downsample_na" and len(unans_records) > len(ans_records):
        unans_records = unans_records[:len(ans_records)]

    summary = {
        "dropped": dict(dropped),
        "pre_balance_counts": pre_balance_counts,
        "post_balance_counts": {"answerable": len(ans_records), "not_answerable": len(unans_records)},
        "balance_mode": balance_mode,
    }

    balanced_records = ans_records + unans_records
    rng.shuffle(balanced_records)
    return balanced_records, summary


def build_records(annotations_path: str, patch_dir: Path, seed: int,
                   balance_mode: str = "downsample_na", val_fraction: float = 0.05):
    """
    Backward-compatible single-file path: builds from ONE annotations
    file and carves off val_fraction internally. Only used when
    --val_annotations is NOT provided. Prefer build_train_val_from_two_files
    when you have a real held-out val file.
    """
    records, summary = build_and_balance_records(annotations_path, patch_dir, seed, balance_mode)
    ans_records = [r for r in records if r["is_answerable"]]
    unans_records = [r for r in records if not r["is_answerable"]]

    def split(lst, frac):
        n_val = int(len(lst) * frac)
        return lst[n_val:], lst[:n_val]

    rng = random.Random(seed)
    ans_train, ans_val = split(ans_records, val_fraction)
    unans_train, unans_val = split(unans_records, val_fraction)

    train_records = ans_train + unans_train
    val_records = ans_val + unans_val
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    return train_records, val_records, summary


def build_train_val_from_two_files(train_annotations_path: str, val_annotations_path: str,
                                    train_patch_dir: Path, val_patch_dir: Path,
                                    seed: int, balance_mode: str = "downsample_na"):
    """
    Preferred path when a real held-out val file (e.g.
    annotations_balanced_val.json) is available: builds train and val
    INDEPENDENTLY from their own files, each balanced on its own terms,
    with no train/val split logic at all (all of train file -> train,
    all of val file -> val). train and val may live under different
    patch-json folders (e.g. patches/train/ vs patches/val/) -- the
    coverage/join step uses each file's own patch_dir.
    """
    train_records, train_summary = build_and_balance_records(
        train_annotations_path, train_patch_dir, seed, balance_mode)
    val_records, val_summary = build_and_balance_records(
        val_annotations_path, val_patch_dir, seed, balance_mode)

    combined_summary = {
        "train": train_summary,
        "val": val_summary,
        "balance_mode": balance_mode,
        # kept for print_build_summary()'s existing single-summary shape
        "dropped": train_summary["dropped"],
        "pre_balance_counts": train_summary["pre_balance_counts"],
        "post_balance_counts": train_summary["post_balance_counts"],
    }
    return train_records, val_records, combined_summary


def print_build_summary(summary: dict, n_train: int, n_val: int, records: list = None):
    print("=== DATASET BUILD SUMMARY ===")
    print(f"Pre-balance:  answerable={summary['pre_balance_counts']['answerable']}  "
          f"not_answerable={summary['pre_balance_counts']['not_answerable']}")
    print(f"Post-balance: answerable={summary['post_balance_counts']['answerable']}  "
          f"not_answerable={summary['post_balance_counts']['not_answerable']}  "
          f"(mode={summary['balance_mode']})")
    if records is not None:
        n_multi = sum(1 for r in records if r.get("is_multipage"))
        print(f"Multi-page (recovered abstractive) records: {n_multi}")
    if summary["dropped"]:
        print("Dropped:")
        for reason, count in summary["dropped"].items():
            print(f"  {reason}: {count}")
    print(f"Train records: {n_train}  Val records: {n_val}")
    print()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VRDUQADataset(Dataset):
    """
    Wraps an in-memory list of records (from build_records) and lazily
    builds (messages, target) at __getitem__ time via the shared
    prompt_utils functions.
    """

    def __init__(self, records: list, patch_dir: str, image_dir: str,
                 processor, include_ocr: bool = True):
        self.records = records
        self.patch_dir = Path(patch_dir)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.include_ocr = include_ocr
        self.multipage_skipped = 0
        self.multipage_skipped_ids = []

    def __len__(self):
        return len(self.records)

    def _build_example(self, record):
        doc_id = record["doc_id"]
        is_multipage = record.get("is_multipage", False)

        if is_multipage:
            pages = record["page"]  # list of ints
            images, ocr_pieces = [], []
            for page in pages:
                img_path = page_image_path(self.image_dir, doc_id, page)
                images.append(load_and_resize_image(img_path))
                if self.include_ocr:
                    patch_path = patch_file_path(self.patch_dir, doc_id, page)
                    if patch_path.exists():
                        patches = load_patch_file(patch_path)
                        page_text = extract_ocr_from_patch_file(patches, max_chars=MAX_OCR_CHARS)
                        if page_text:
                            ocr_pieces.append(f"[Page {page}]\n{page_text}")
            # Cap applied once over the WHOLE joined multi-page text (not
            # per-page) -- this is a training-only convention for the
            # recovered abstractive items; the eval side never hits this
            # path since DUDE_verified is always single-page.
            ocr_text = "\n\n".join(ocr_pieces)
            if len(ocr_text) > MAX_OCR_CHARS:
                ocr_text = ocr_text[:MAX_OCR_CHARS]
            messages = build_messages_multi(images=images, question=record["question"], ocr_text=ocr_text)
            target = build_target(record.get("answer"), record["is_answerable"])
            return messages, target, images

        page = record["page"]
        img_path = page_image_path(self.image_dir, doc_id, page)

        ocr_text = ""
        if self.include_ocr:
            patch_path = patch_file_path(self.patch_dir, doc_id, page)
            if patch_path.exists():
                patches = load_patch_file(patch_path)
                ocr_text = extract_ocr_from_patch_file(patches, max_chars=MAX_OCR_CHARS)

        image = load_and_resize_image(img_path)
        messages = build_messages(image=image, question=record["question"], ocr_text=ocr_text)
        target = build_target(record.get("answer"), record["is_answerable"])
        return messages, target, image

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            return self._encode(record)
        except Exception as e:
            if record.get("is_multipage"):
                # Safety net (not a cap): a long multi-page example failed
                # to encode (sequence too long / OOM / image-token
                # misalignment after truncation). Skip it rather than
                # crash the run, and fall back to the next example.
                self.multipage_skipped += 1
                self.multipage_skipped_ids.append(record["example_id"])
                print(f"WARNING: skipping multi-page example {record['example_id']} "
                      f"({len(record['page'])} pages) -- encode failed: {e}")
                fallback_idx = (idx + 1) % len(self.records)
                if fallback_idx == idx:
                    raise
                return self.__getitem__(fallback_idx)
            raise

    def _encode(self, record):
        messages, target, image_or_images = self._build_example(record)

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target + self.processor.tokenizer.eos_token

        image_inputs, _ = process_vision_info(messages)

        prompt_enc = self.processor(
            text=[prompt_text], images=image_inputs, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        )
        full_enc = self.processor(
            text=[full_text], images=image_inputs, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        )

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # loss only on the completion

        item = {k: v[0] for k, v in full_enc.items()}
        item["labels"] = labels
        return item


@dataclass
class SingleExampleCollator:
    """BATCH_SIZE=1 -- nothing to pad across examples, just add batch dim."""

    def __call__(self, features):
        assert len(features) == 1, "This collator assumes BATCH_SIZE=1"
        f = features[0]
        return {k: v.unsqueeze(0) for k, v in f.items()}


# ---------------------------------------------------------------------------
# LoRA target module discovery
# ---------------------------------------------------------------------------
def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)
    return sorted(names)


# ---------------------------------------------------------------------------
# Generation-based evaluation metrics
# ---------------------------------------------------------------------------
def compute_binary_classification_metrics(preds_refused, trues_refused):
    """Macro F1 over {refuse, answer}, plus per-class recall
    (ans_acc = recall on 'answerable', unans_acc = recall on 'unanswerable')."""
    n = len(preds_refused)
    if n == 0:
        return {"macro_f1": 0.0, "ans_acc": 0.0, "unans_acc": 0.0}

    tp_ref = sum(1 for p, t in zip(preds_refused, trues_refused) if p and t)
    fp_ref = sum(1 for p, t in zip(preds_refused, trues_refused) if p and not t)
    fn_ref = sum(1 for p, t in zip(preds_refused, trues_refused) if not p and t)

    tp_ans = sum(1 for p, t in zip(preds_refused, trues_refused) if not p and not t)
    fp_ans = fn_ref
    fn_ans = fp_ref

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    f1_ref = f1(tp_ref, fp_ref, fn_ref)
    f1_ans = f1(tp_ans, fp_ans, fn_ans)
    macro_f1 = (f1_ref + f1_ans) / 2

    n_true_refuse = sum(trues_refused)
    n_true_answer = n - n_true_refuse
    unans_acc = tp_ref / n_true_refuse if n_true_refuse > 0 else 0.0
    ans_acc = tp_ans / n_true_answer if n_true_answer > 0 else 0.0

    return {"macro_f1": macro_f1, "ans_acc": ans_acc, "unans_acc": unans_acc}


class VRDUQATrainer(Trainer):
    """Overrides evaluate() to run real generation and merge macro_f1 /
    ans_acc / unans_acc / selection_score into the metrics dict BEFORE
    Trainer's best-model / early-stopping logic consumes it."""

    def __init__(self, *args, gen_max_new_tokens=64, gen_subset_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_subset_size = gen_subset_size

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        gen_metrics = self._run_generation_eval(ds, metric_key_prefix)
        output.update(gen_metrics)
        self.log(gen_metrics)
        return output

    @torch.no_grad()
    def _run_generation_eval(self, dataset: VRDUQADataset, prefix: str):
        self.model.eval()
        records = dataset.records
        if self.gen_subset_size is not None and self.gen_subset_size < len(records):
            rng = random.Random(0)
            records = rng.sample(records, self.gen_subset_size)

        preds_refused, trues_refused = [], []
        for record in records:
            messages, _, _ = dataset._build_example(record)
            prompt_text = dataset.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = dataset.processor(
                text=[prompt_text], images=image_inputs, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LENGTH,
            ).to(self.model.device)

            gen_ids = self.model.generate(
                **inputs, max_new_tokens=self.gen_max_new_tokens, do_sample=False,
            )
            gen_text = dataset.processor.tokenizer.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            preds_refused.append(is_refusal(gen_text))
            trues_refused.append(not record["is_answerable"])

        metrics = compute_binary_classification_metrics(preds_refused, trues_refused)
        selection_score = metrics["macro_f1"] - 0.5 * max(
            0.0, metrics["unans_acc"] - metrics["ans_acc"]
        )
        metrics["selection_score"] = selection_score

        self.model.train()
        return {f"{prefix}_{k}": v for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--annotations", required=True,
                     help="Path to annotations_balanced_train.json")
    ap.add_argument("--val_annotations", default=None,
                     help="Path to a real held-out val file (e.g. "
                          "annotations_balanced_val.json). If given, train "
                          "uses 100%% of --annotations and val is built "
                          "independently from this file (no internal split). "
                          "If omitted, falls back to carving val_fraction off "
                          "--annotations.")
    ap.add_argument("--patch_dir", required=True, help="Train patch-json folder")
    ap.add_argument("--image_dir", required=True, help="Train page-image folder")
    ap.add_argument("--val_patch_dir", default=None,
                     help="Val patch-json folder, if different from --patch_dir "
                          "(e.g. a separate patches/val/ directory). Defaults to "
                          "--patch_dir if not given.")
    ap.add_argument("--val_image_dir", default=None,
                     help="Val page-image folder, if different from --image_dir. "
                          "Defaults to --image_dir if not given.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--balance_mode", choices=["downsample_na", "none"], default="downsample_na")
    ap.add_argument("--no_ocr", action="store_true",
                     help="Train the no-OCR arm instead of the OCR arm.")
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--early_stopping_patience", type=int, default=4)
    ap.add_argument("--gen_subset_size", type=int, default=200,
                     help="Cap generation-eval to N examples per eval round; "
                          "None = full val set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    include_ocr = not args.no_ocr
    ocr_suffix = "ocr" if include_ocr else "noocr"
    output_dir = Path(args.output_dir) / ocr_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    done_marker = output_dir / "TRAINING_COMPLETE"
    if done_marker.exists():
        print(f"{done_marker} already exists -- training already completed. Exiting (no-op).")
        print("(This is expected if a chained successor job started after the run "
              "already finished cleanly; the .sh script's cancel step should "
              "normally prevent this from happening, but this is a safety net.)")
        return

    val_patch_dir = args.val_patch_dir or args.patch_dir
    val_image_dir = args.val_image_dir or args.image_dir

    print(f"Building dataset from {args.annotations} + {args.patch_dir} ...")
    if args.val_annotations:
        print(f"Using held-out val file: {args.val_annotations} (no internal split)")
        print(f"Val patch_dir: {val_patch_dir}  Val image_dir: {val_image_dir}")
        train_records, val_records, summary = build_train_val_from_two_files(
            args.annotations, args.val_annotations,
            Path(args.patch_dir), Path(val_patch_dir),
            args.seed, balance_mode=args.balance_mode,
        )
    else:
        print("No --val_annotations given -- carving 5% off --annotations for val "
              "(pass --val_annotations to use a real held-out file instead).")
        train_records, val_records, summary = build_records(
            args.annotations, Path(args.patch_dir), args.seed, balance_mode=args.balance_mode,
        )
    print_build_summary(summary, len(train_records), len(val_records), records=train_records + val_records)

    print(f"Loading processor/model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS,
    )
    model = QwenVLModel.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    target_modules = find_all_linear_names(model)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = VRDUQADataset(train_records, args.patch_dir, args.image_dir, processor, include_ocr=include_ocr)
    val_ds = VRDUQADataset(val_records, val_patch_dir, val_image_dir, processor, include_ocr=include_ocr)
    print(f"Train examples: {len(train_ds)}  Val examples: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_selection_score",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tb_logs"),
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = VRDUQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=SingleExampleCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        gen_subset_size=args.gen_subset_size,
    )

    resume_ckpt = get_last_checkpoint(str(output_dir))
    if resume_ckpt:
        print(f"Found existing checkpoint, resuming from: {resume_ckpt}")
    else:
        print("No existing checkpoint found -- starting fresh.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"Multi-page examples skipped (train): {train_ds.multipage_skipped}")
    print(f"Multi-page examples skipped (val):   {val_ds.multipage_skipped}")
    if train_ds.multipage_skipped_ids:
        print(f"  skipped train example_ids: {train_ds.multipage_skipped_ids}")
    if val_ds.multipage_skipped_ids:
        print(f"  skipped val example_ids:   {val_ds.multipage_skipped_ids}")

    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))
    done_marker.touch()
    print(f"Done. Best model saved to {output_dir / 'best'}")
    print(f"Wrote completion marker: {done_marker}")


if __name__ == "__main__":
    main()