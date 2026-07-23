import os
import re
import json
import math
import time
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
from transformers.trainer_utils import get_last_checkpoint, speed_metrics

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
    compute_adaptive_page_pixels,
    build_messages,
    build_messages_multi,
    build_target,
    is_refusal,
)

MAX_SEQ_LENGTH = 8192
MERGE_SIZE = 2  # Qwen2.5-VL vision merger group size (2x2 patches -> 1 token)
WINDOW_SIZE_PATCHES = 8  # windowed-attention block size in raw 14px patches (112px window)
# The merge-only check (h,w divisible by MERGE_SIZE=2) isn't sufficient --
# Qwen2.5-VL's windowed attention needs whole windows of 8x8 raw patches.
# load_and_resize_image() already rounds to multiples of this, so this
# check should never actually fire in normal operation; it's a backup
# in case a resize bug is ever introduced upstream.
PATCH_NAME_RE = re.compile(r"^([0-9a-f]{32})_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------
def build_docid_page_index(patch_dir: Path):
    """Scan the patch folder once and build {docId: sorted page numbers}."""
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
    """Deterministic page pick for not-answerable questions -- seeded per
    question so it's reproducible, but not always page 0."""
    rng = random.Random(f"{seed}-{question_id}")
    return rng.choice(available_pages)


def get_answer_page(item: dict):
    boxes = item.get("answers_page_bounding_boxes") or []
    if not boxes or not boxes[0]:
        return None
    return boxes[0][0]["page"]


def build_and_balance_records(annotations_path: str, patch_dir: Path, seed: int,
                               balance_mode: str = "downsample_na",
                               max_images_per_doc: int = 20):
    """
    Builds training records from one annotations file: joins questions
    against the patch folder, assigns pages, recovers abstractive items
    as multi-page examples, and balances answerable vs not-answerable.

    balance_mode:
      "downsample_na" -> drop not-answerable records down to match the
                          answerable count (default).
      "none"          -> keep the natural counts as-is.

    max_images_per_doc caps how many pages a multi-page example can show,
    applied before the adaptive resolution scaling. Without a hard cap,
    a pathologically long document can still exceed the GPU memory budget
    even at the lowest resolution the adaptive scaling allows.
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
                # No page bbox -- recover as a multi-page example instead
                # of dropping it.
                answer = item["answers"][0] if item.get("answers") else None
                if not answer:
                    dropped["answerable_missing_page"] += 1
                    continue
                capped_pages = sorted(available_pages)[:max_images_per_doc]
                records.append({
                    "example_id": question_id,
                    "doc_id": doc_id,
                    "page": capped_pages,
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
                   balance_mode: str = "downsample_na", val_fraction: float = 0.05,
                   max_images_per_doc: int = 20):
    """
    Single-file path: builds from one annotations file and carves off
    val_fraction internally. Used when --val_annotations isn't given --
    prefer build_train_val_from_two_files when you have a real held-out
    val set.
    """
    records, summary = build_and_balance_records(annotations_path, patch_dir, seed, balance_mode,
                                                   max_images_per_doc=max_images_per_doc)
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
                                    seed: int, balance_mode: str = "downsample_na",
                                    max_images_per_doc: int = 20):
    """
    Preferred path when a real held-out val file is available: builds
    train and val independently from their own files (and, optionally,
    their own patch folders), with no split logic needed.
    """
    train_records, train_summary = build_and_balance_records(
        train_annotations_path, train_patch_dir, seed, balance_mode,
        max_images_per_doc=max_images_per_doc)
    val_records, val_summary = build_and_balance_records(
        val_annotations_path, val_patch_dir, seed, balance_mode,
        max_images_per_doc=max_images_per_doc)

    combined_summary = {
        "train": train_summary,
        "val": val_summary,
        "balance_mode": balance_mode,
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
    """Wraps an in-memory list of records and lazily builds (messages,
    target) at __getitem__ time via the shared prompt_utils functions."""

    def __init__(self, records: list, patch_dir: str, image_dir: str,
                 processor, include_ocr: bool = True, multipage_target_pages: int = 5):
        self.records = records
        self.patch_dir = Path(patch_dir)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.include_ocr = include_ocr
        self.multipage_target_pages = multipage_target_pages
        self.skipped_total = 0
        self.skipped_ids = []
        self.multipage_skipped = 0
        self.multipage_skipped_ids = []

    def __len__(self):
        return len(self.records)

    def _build_example(self, record):
        doc_id = record["doc_id"]
        is_multipage = record.get("is_multipage", False)

        if is_multipage:
            pages = record["page"]
            per_page_pixels = compute_adaptive_page_pixels(
                n_pages=len(pages), target_pages_equivalent=self.multipage_target_pages,
                base_max_pixels=IMAGE_MAX_PIXELS,
            )
            images, ocr_pieces = [], []
            for page in pages:
                img_path = page_image_path(self.image_dir, doc_id, page)
                images.append(load_and_resize_image(img_path, max_pixels=per_page_pixels))
                if self.include_ocr:
                    patch_path = patch_file_path(self.patch_dir, doc_id, page)
                    if patch_path.exists():
                        patches = load_patch_file(patch_path)
                        page_text = extract_ocr_from_patch_file(patches, max_chars=MAX_OCR_CHARS)
                        if page_text:
                            ocr_pieces.append(f"[Page {page}]\n{page_text}")
            # The char cap applies once over the whole joined multi-page
            # text, not per page -- OCR doesn't get resolution-scaled the
            # way images do.
            ocr_text = "\n\n".join(ocr_pieces)
            if len(ocr_text) > MAX_OCR_CHARS:
                ocr_text = ocr_text[:MAX_OCR_CHARS]
            messages = build_messages_multi(images=images, question=record["question"], ocr_text=ocr_text,
                                             max_pixels=per_page_pixels)
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
            # Skip and fall through to the next example instead of
            # crashing the whole run -- a long multi-page example can
            # still be too big for MAX_SEQ_LENGTH, or hit a degenerate
            # image grid (see the check in _encode).
            self.skipped_total += 1
            self.skipped_ids.append(record["example_id"])
            if record.get("is_multipage"):
                self.multipage_skipped += 1
                self.multipage_skipped_ids.append(record["example_id"])
                print(f"WARNING: skipping multi-page example {record['example_id']} "
                      f"({len(record['page'])} pages) -- encode failed: {e}")
            else:
                print(f"WARNING: skipping example {record['example_id']} "
                      f"(doc_id={record['doc_id']} page={record.get('page')}) "
                      f"-- encode failed: {e}")
            fallback_idx = (idx + 1) % len(self.records)
            if fallback_idx == idx:
                raise
            return self.__getitem__(fallback_idx)

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

        # The vision merger groups patches in 2x2 blocks along height and
        # width independently -- a grid like (t=1,h=1,w=4) has a total
        # patch count divisible by 4 but still can't form a valid 2-row
        # group, and crashes deep in model.forward(). Check h and w
        # separately rather than just the total.
        if "image_grid_thw" in full_enc:
            for grid in full_enc["image_grid_thw"]:
                t, h, w = (int(x) for x in grid.tolist())
                if h % WINDOW_SIZE_PATCHES != 0 or w % WINDOW_SIZE_PATCHES != 0 \
                        or h < WINDOW_SIZE_PATCHES or w < WINDOW_SIZE_PATCHES:
                    raise ValueError(
                        f"Degenerate image grid (t={t},h={h},w={w}) -- h and w must "
                        f"each be >= and divisible by WINDOW_SIZE_PATCHES={WINDOW_SIZE_PATCHES} "
                        f"for doc_id={record['doc_id']} page={record.get('page')}"
                    )

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # loss only on the completion

        # pixel_values and image_grid_thw aren't batched per-example the
        # way input_ids/attention_mask are -- the processor returns
        # pixel_values as a flat (total_patches, feature_dim) tensor and
        # image_grid_thw as (num_images, 3), with no real leading batch
        # dimension to strip. Only squeeze the keys that actually have one.
        no_batch_dim_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        item = {
            k: (v if k in no_batch_dim_keys else v[0])
            for k, v in full_enc.items()
        }
        item["labels"] = labels
        return item


@dataclass
class SingleExampleCollator:
    """Batch size is always 1, so there's nothing to pad across examples --
    just add the batch dimension back for the keys that need it.
    pixel_values/image_grid_thw are already in the shape the model
    expects and must not be unsqueezed."""

    NO_BATCH_DIM_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}

    def __call__(self, features):
        assert len(features) == 1, "This collator assumes batch_size=1"
        f = features[0]
        return {
            k: (v if k in self.NO_BATCH_DIM_KEYS else v.unsqueeze(0))
            for k, v in f.items()
        }


# ---------------------------------------------------------------------------
# LoRA target module discovery
# ---------------------------------------------------------------------------
def find_all_linear_names(model):
    """
    Collects the LLM decoder's linear projection names for LoRA's
    target_modules -- q/k/v/o_proj and gate/up/down_proj. Deliberately
    excludes the vision tower entirely: LoRA-adapting vision layers isn't
    necessary here and caused problems with Qwen2.5-VL's windowed
    attention during testing, so this sticks to the standard practice of
    only targeting the language model side.
    """
    llm_target_leaves = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    names = set()
    for name, module in model.named_modules():
        if name.startswith("visual") or ".visual." in name:
            continue
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in llm_target_leaves:
                names.add(leaf)
    return sorted(names)


def freeze_vision_tower_no_grad(model):
    """
    Wraps the vision tower's forward pass in torch.no_grad(). LoRA
    already only targets the LLM side, so the vision tower's weights
    don't need gradients -- but gradient checkpointing doesn't know that
    in advance, and wraps every block uniformly, recomputing the vision
    tower's forward pass on every backward step for no benefit. This
    removes that cost entirely instead of relying on requires_grad=False
    alone.
    """
    if not hasattr(model, "visual"):
        print("WARNING: model has no top-level 'visual' attribute -- "
              "check model.named_children() for the actual name. "
              "Skipping; vision tower will still be gradient-checkpointed normally.")
        return model

    original_forward = model.visual.forward

    def no_grad_forward(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    model.visual.forward = no_grad_forward
    for p in model.visual.parameters():
        p.requires_grad = False
    print("Vision tower forward wrapped in torch.no_grad().")
    return model


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
    """
    Overrides evaluate() to run real generation and merge macro_f1 /
    ans_acc / unans_acc / selection_score into the metrics dict before
    Trainer's best-model / early-stopping logic reads it -- calling
    super().evaluate() directly would fire the early-stopping callback
    with only the base eval_loss, one step too early to see these.

    Also overrides training_step()/prediction_step() as a safety net:
    the per-image grid check in _encode() catches the obvious cases, but
    Qwen2.5-VL's windowed attention has constraints across concatenated
    multi-page images that a simple per-image check can't fully predict,
    and large multi-page examples can occasionally OOM. Either failure
    gets caught here, logged, and the batch is skipped rather than
    losing the whole run.
    """

    def __init__(self, *args, gen_max_new_tokens=64, gen_subset_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_subset_size = gen_subset_size
        self.degenerate_batches_skipped = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            is_shape_bug = "is invalid for input of size" in str(e) or "spatial_merge_unit" in str(e)
            is_oom = "out of memory" in str(e).lower()
            if is_shape_bug or is_oom:
                self.degenerate_batches_skipped += 1
                kind = "OOM" if is_oom else "degenerate image batch"
                print(f"WARNING: skipping training step {self.state.global_step} -- "
                      f"{kind} (total skipped so far: {self.degenerate_batches_skipped}): {e}")
                model.zero_grad(set_to_none=True)
                if is_oom:
                    # `inputs` is a function parameter, so it stays
                    # referenced (and its GPU tensors stay resident) for
                    # the rest of this function even inside this except
                    # block -- empty_cache() alone won't release memory
                    # that's still referenced. Drop the reference and
                    # force garbage collection first, since the failed
                    # step's tensors would otherwise linger and eat into
                    # the next step's budget.
                    del inputs
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                return torch.tensor(0.0, device=self.args.device)
            raise

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        except RuntimeError as e:
            is_shape_bug = "is invalid for input of size" in str(e) or "spatial_merge_unit" in str(e)
            is_oom = "out of memory" in str(e).lower()
            if is_shape_bug or is_oom:
                self.degenerate_batches_skipped += 1
                kind = "OOM" if is_oom else "degenerate image batch"
                print(f"WARNING: skipping eval prediction step -- {kind} "
                      f"(total skipped so far: {self.degenerate_batches_skipped}): {e}")
                if is_oom:
                    del inputs
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                return (None, None, None)
            raise

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset_to_use)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix, start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        metrics = dict(output.metrics)

        gen_metrics = self._run_generation_eval(eval_dataset_to_use, metric_key_prefix)
        metrics.update(gen_metrics)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    @torch.no_grad()
    def _run_generation_eval(self, dataset: VRDUQADataset, prefix: str):
        self.model.eval()
        records = dataset.records
        if self.gen_subset_size is not None and self.gen_subset_size < len(records):
            rng = random.Random(0)
            records = rng.sample(records, self.gen_subset_size)

        preds_refused, trues_refused = [], []
        gen_eval_skipped = 0
        for record in records:
            try:
                messages, _, _ = dataset._build_example(record)
                prompt_text = dataset.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)
                inputs = dataset.processor(
                    text=[prompt_text], images=image_inputs, return_tensors="pt",
                    truncation=True, max_length=MAX_SEQ_LENGTH,
                ).to(self.model.device)

                if "image_grid_thw" in inputs:
                    for grid in inputs["image_grid_thw"]:
                        t, h, w = (int(x) for x in grid.tolist())
                        if h % WINDOW_SIZE_PATCHES != 0 or w % WINDOW_SIZE_PATCHES != 0 \
                                or h < WINDOW_SIZE_PATCHES or w < WINDOW_SIZE_PATCHES:
                            raise ValueError(
                                f"Degenerate image grid (t={t},h={h},w={w}) "
                                f"for doc_id={record['doc_id']}"
                            )

                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=self.gen_max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                )
                gen_text = dataset.processor.tokenizer.decode(
                    gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            except Exception as e:
                gen_eval_skipped += 1
                print(f"WARNING: skipping generation-eval example {record['example_id']} "
                      f"-- failed: {e}")
                continue

            preds_refused.append(is_refusal(gen_text))
            trues_refused.append(not record["is_answerable"])

        if gen_eval_skipped:
            print(f"Generation eval: skipped {gen_eval_skipped}/{len(records)} examples")

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
                     help="Path to a real held-out val file. If given, train uses "
                          "100%% of --annotations and val is built independently "
                          "from this file. If omitted, falls back to carving "
                          "val_fraction off --annotations.")
    ap.add_argument("--patch_dir", required=True, help="Train patch-json folder")
    ap.add_argument("--image_dir", required=True, help="Train page-image folder")
    ap.add_argument("--val_patch_dir", default=None,
                     help="Val patch-json folder, if different from --patch_dir.")
    ap.add_argument("--val_image_dir", default=None,
                     help="Val page-image folder, if different from --image_dir.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--balance_mode", choices=["downsample_na", "none"], default="downsample_na")
    ap.add_argument("--multipage_target_pages", type=int, default=5,
                     help="Target image-token budget for multi-page examples, "
                          "expressed as N full-resolution-equivalent pages. Docs "
                          "with <= N pages get full resolution; longer docs get "
                          "every page shown (up to --max_images_per_doc) but "
                          "scaled down so the total stays roughly constant.")
    ap.add_argument("--max_images_per_doc", type=int, default=20,
                     help="Hard cap on pages per multi-page example, applied "
                          "before the adaptive resolution scaling above.")
    ap.add_argument("--no_ocr", action="store_true",
                     help="Train the no-OCR arm instead of the OCR arm.")
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--freeze_vision_tower", action="store_true", default=True,
                     help="Wrap the vision tower forward in torch.no_grad() to "
                          "avoid gradient-checkpointing recomputation cost for it. "
                          "LoRA already only targets the LLM side, so this doesn't "
                          "change what's trainable.")
    ap.add_argument("--no_freeze_vision_tower", dest="freeze_vision_tower", action="store_false")
    ap.add_argument("--early_stopping_patience", type=int, default=4)
    ap.add_argument("--gen_subset_size", type=int, default=200,
                     help="Cap generation-eval to N examples per round; "
                          "None = full val set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    include_ocr = not args.no_ocr
    ocr_suffix = "ocr" if include_ocr else "noocr"
    output_dir = Path(args.output_dir) / ocr_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    done_marker = output_dir / "TRAINING_COMPLETE"
    if done_marker.exists():
        print(f"{done_marker} already exists -- training already completed. Exiting.")
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
            max_images_per_doc=args.max_images_per_doc,
        )
    else:
        print("No --val_annotations given -- carving 5% off --annotations for val.")
        train_records, val_records, summary = build_records(
            args.annotations, Path(args.patch_dir), args.seed, balance_mode=args.balance_mode,
            max_images_per_doc=args.max_images_per_doc,
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
    if args.freeze_vision_tower:
        model = freeze_vision_tower_no_grad(model)
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
    # Needed when combining LoRA with gradient checkpointing: the
    # checkpoint mechanism needs at least one tensor in the checkpointed
    # region to require grad to track backprop through it correctly.
    model.enable_input_require_grads()

    train_ds = VRDUQADataset(train_records, args.patch_dir, args.image_dir, processor,
                              include_ocr=include_ocr, multipage_target_pages=args.multipage_target_pages)
    val_ds = VRDUQADataset(val_records, val_patch_dir, val_image_dir, processor,
                            include_ocr=include_ocr, multipage_target_pages=args.multipage_target_pages)
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

    print(f"Total examples skipped (train): {train_ds.skipped_total}  "
          f"(of which multi-page: {train_ds.multipage_skipped})")
    print(f"Total examples skipped (val):   {val_ds.skipped_total}  "
          f"(of which multi-page: {val_ds.multipage_skipped})")
    print(f"Degenerate training batches skipped (caught at model.forward): "
          f"{trainer.degenerate_batches_skipped}")
    if train_ds.skipped_ids:
        print(f"  skipped train example_ids: {train_ds.skipped_ids}")
    if val_ds.skipped_ids:
        print(f"  skipped val example_ids:   {val_ds.skipped_ids}")

    trainer.save_model(str(output_dir / "best"))
    processor.save_pretrained(str(output_dir / "best"))
    done_marker.touch()
    print(f"Done. Best model saved to {output_dir / 'best'}")
    print(f"Wrote completion marker: {done_marker}")


if __name__ == "__main__":
    main()