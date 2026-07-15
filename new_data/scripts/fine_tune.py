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

MAX_SEQ_LENGTH = 8192  # separate from MAX_OCR_CHARS; tokenized-sequence cap
MERGE_SIZE = 2  # Qwen2.5-VL vision merger spatial group size (2x2 patches -> 1 merged token)
_DIAGNOSTIC_COUNT = [0]  # temporary, see _encode() -- remove once the mismatch is identified
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
                # pages, using ADAPTIVE per-page resolution (see
                # compute_adaptive_page_pixels in prompt_utils.py) so long
                # documents don't blow the token budget -- unlike the
                # earlier unbounded-full-resolution attempt (reverted after
                # a Qwen2.5-VL windowed-attention crash), which turned out
                # to be caused by an unrelated bug in this file's own
                # tensor handling (now fixed), not by multi-page
                # concatenation itself. No pages are excluded, so the
                # answer page is never at risk of being cut out --
                # avoiding the hallucination-training risk that a hard
                # page cap would reintroduce.
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
        self.skipped_total = 0
        self.skipped_ids = []

    def __len__(self):
        return len(self.records)

    def _build_example(self, record):
        doc_id = record["doc_id"]
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
            # Safety net (not a cap): the example failed to encode --
            # either a long multi-page example too big for MAX_SEQ_LENGTH,
            # OOM, or a degenerate image (see the image_grid_thw check in
            # _encode). Skip it and fall back to the next example rather
            # than crashing the whole run.
            self.skipped_total += 1
            self.skipped_ids.append(record["example_id"])
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

        # TEMPORARY DIAGNOSTIC (2026-07-12): the image_grid_thw pre-check
        # below has never actually fired despite a 100% model.forward()
        # crash rate -- meaning grid_thw itself looks fine by our check,
        # but something about the actual pixel_values tensor doesn't match
        # what that grid claims. Dump real numbers for the first few calls
        # so we can see the actual mismatch instead of guessing further.
        if _DIAGNOSTIC_COUNT[0] < 5:
            _DIAGNOSTIC_COUNT[0] += 1
            img = image_or_images if not isinstance(image_or_images, list) else image_or_images[0]
            pv_shape = tuple(full_enc["pixel_values"].shape) if "pixel_values" in full_enc else None
            grid_vals = full_enc["image_grid_thw"].tolist() if "image_grid_thw" in full_enc else None
            n_images = len(image_inputs) if isinstance(image_inputs, list) else 1
            print(f"DIAGNOSTIC[{_DIAGNOSTIC_COUNT[0]}] doc_id={record['doc_id']} "
                  f"page={record.get('page')} PIL_size(w,h)={img.size} "
                  f"n_images_in_message={n_images} "
                  f"pixel_values.shape={pv_shape} image_grid_thw={grid_vals} "
                  f"input_ids.shape={tuple(full_enc['input_ids'].shape)}")

        # Proactive check: Qwen2.5-VL's vision merger groups patches in
        # blocks of SPATIAL_MERGE_UNIT (4) via a 2D (h,w) spatial grouping --
        # NOT just total patch count divisible by 4. A lopsided grid like
        # (t=1,h=1,w=4) has total=4 (passes a naive total%4==0 check) but
        # h=1 can't be split into 2-row groups, and still crashes deep in
        # model.forward(). Check h and w individually divisible by
        # MERGE_SIZE instead.
        if "image_grid_thw" in full_enc:
            for grid in full_enc["image_grid_thw"]:
                t, h, w = (int(x) for x in grid.tolist())
                if h % MERGE_SIZE != 0 or w % MERGE_SIZE != 0 or h < MERGE_SIZE or w < MERGE_SIZE:
                    raise ValueError(
                        f"Degenerate image grid (t={t},h={h},w={w}) -- h and w must "
                        f"each be >= and divisible by MERGE_SIZE={MERGE_SIZE} "
                        f"for doc_id={record['doc_id']} page={record.get('page')}"
                    )

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # loss only on the completion

        # BUG FIX (2026-07-12): pixel_values and image_grid_thw are NOT
        # batched per-example the way input_ids/attention_mask are.
        # Qwen2.5-VL's processor returns pixel_values as a FLAT tensor of
        # shape (total_patches, feature_dim) across however many images
        # were passed, and image_grid_thw as (num_images, 3) -- neither
        # has a real leading "batch" dimension to strip. Blindly doing
        # v[0] on pixel_values (as this used to) sliced out just the
        # FIRST PATCH ROW, collapsing e.g. a (4480, 1176) tensor down to
        # (1176,). After the collator's unsqueeze(0), the model received
        # exactly one patch -- which is a (1, hidden_size) tensor after
        # patch embedding, i.e. exactly 1280 total elements for this
        # model's hidden_size -- an EXACT match for the
        # "shape '[0,4,-1]' is invalid for input of size 1280" crash that
        # was 100% reproducible on every single training image,
        # regardless of image content, LoRA config, attention backend, or
        # transformers version. This was never caught by minimal_repro.py
        # because that script uses processor() output directly, without
        # ever going through this slicing at all.
        NO_BATCH_DIM_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        item = {
            k: (v if k in NO_BATCH_DIM_KEYS else v[0])
            for k, v in full_enc.items()
        }
        item["labels"] = labels
        return item


@dataclass
class SingleExampleCollator:
    """BATCH_SIZE=1 -- nothing to pad across examples. Only input_ids/
    attention_mask/labels get a batch dimension added; pixel_values and
    image_grid_thw are already in the flat shape the model expects
    (see the matching fix in VRDUQADataset._encode()) and must NOT be
    unsqueezed -- doing so previously added a spurious leading dimension
    on top of an already-wrong single-patch slice."""

    NO_BATCH_DIM_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}

    def __call__(self, features):
        assert len(features) == 1, "This collator assumes BATCH_SIZE=1"
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
    Collects nn.Linear leaf names for LoRA's target_modules -- LLM decoder
    only, matching the last-known-working version's LORA_TARGET_MODULES
    (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj). This
    explicitly EXCLUDES anything under the vision tower ("visual.*":
    attention qkv/proj, and the Sequential-indexed merger MLP).

    Earlier versions of this file collected every nn.Linear in the WHOLE
    model, including the vision tower, and wrapped those in LoRA too --
    that produced a 100% reproducible crash on every single training
    image ("shape '[0,4,-1]' is invalid...", deep in Qwen2.5-VL's window-
    index/reshape logic), which several rounds of image-resize and
    version-pin fixes failed to resolve. The old working version never
    touched the vision tower with LoRA at all -- only the LLM decoder's
    projections -- which is also the standard/common practice for
    Qwen2.5-VL LoRA fine-tuning. Restricting to LLM-only here to match.
    """
    llm_target_leaves = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    names = set()
    for name, module in model.named_modules():
        if name.startswith("visual") or ".visual." in name:
            continue  # exclude the entire vision tower
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in llm_target_leaves:
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
    Trainer's best-model / early-stopping logic consumes it. Also
    overrides training_step() as a last-resort safety net: the
    image_grid_thw check in VRDUQADataset._encode() catches the obvious
    single-image "too few patches" case, but Qwen2.5-VL's windowed
    attention has more complex constraints across CONCATENATED multi-page
    images that a simple per-image patch-count check can't fully predict.
    If a batch still fails deep inside model.forward() despite passing
    that check, this catches it here, logs which step, and skips the
    batch (zero contribution, no crash) instead of losing the whole run."""

    def __init__(self, *args, gen_max_new_tokens=64, gen_subset_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_subset_size = gen_subset_size
        self.degenerate_batches_skipped = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "is invalid for input of size" in str(e) or "spatial_merge_unit" in str(e):
                self.degenerate_batches_skipped += 1
                print(f"WARNING: skipping training step {self.state.global_step} -- "
                      f"degenerate image batch made it past the pre-check "
                      f"(total skipped so far: {self.degenerate_batches_skipped}): {e}")
                model.zero_grad(set_to_none=True)
                return torch.tensor(0.0, device=self.args.device)
            raise

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Same crash class as training_step, but this is what HF's own
        # evaluation_loop calls internally during super().evaluate() below
        # -- the __getitem__/_encode-level pre-check can pass (the item
        # "encoded fine") and the crash still only surfaces once the
        # actual forward pass runs, same as training. Mirror the same
        # catch-and-skip here so eval doesn't lose the whole run either.
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        except RuntimeError as e:
            if "is invalid for input of size" in str(e) or "spatial_merge_unit" in str(e):
                self.degenerate_batches_skipped += 1
                print(f"WARNING: skipping eval prediction step -- degenerate image batch "
                      f"(total skipped so far: {self.degenerate_batches_skipped}): {e}")
                return (None, None, None)
            raise

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Deliberately does NOT call super().evaluate() -- that high-level
        wrapper internally fires callback_handler.on_evaluate() (which is
        what EarlyStoppingCallback listens to) BEFORE returning control
        back here, using only the base eval_loss-style metrics. Our
        generation-based eval_selection_score would then be merged in too
        late for EarlyStoppingCallback to ever see it on its one look per
        round -- exactly the "early stopping required metric_for_best_model,
        but did not find eval_selection_score" warning, every epoch.

        Checkpoint-saving (load_best_model_at_end) was unaffected by that
        bug since it re-reads whatever this method eventually returns, but
        early stopping specifically needs the callback fired with the
        FULL metrics dict, which means computing everything first and
        only then triggering logging/callbacks ourselves.
        """
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
                        if h % MERGE_SIZE != 0 or w % MERGE_SIZE != 0 or h < MERGE_SIZE or w < MERGE_SIZE:
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
    # UNMISTAKABLE VERSION FINGERPRINT (2026-07-12): five consecutive
    # supposedly-different fixes have produced byte-identical failure
    # counts (1978->1987, steps 247-248) in the training log. That's not
    # plausible as a coincidence of real model behavior -- it strongly
    # suggests the running job is NOT using the code being edited. This
    # print is the definitive test: if this exact string does not appear
    # at the top of your .out log, the file that ran is NOT this file,
    # and the actual next step is investigating deployment (stale copy,
    # __pycache__, network-filesystem mtime granularity, wrong path) --
    # NOT another guess about Qwen2.5-VL internals.
    print(">>> FINGERPRINT: fine_tune.py VERSION=LLM_ONLY_LORA_FIX_2026_07_12 <<<", flush=True)
    import sys as _sys
    print(f">>> FINGERPRINT: running from {__file__} <<<", flush=True)
    print(f">>> FINGERPRINT: python executable = {_sys.executable} <<<", flush=True)

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
    # Required when combining LoRA (frozen base model) with gradient
    # checkpointing: PyTorch's checkpoint mechanism needs at least one
    # tensor in the checkpointed region to require grad to properly track
    # backprop through it. Without this, the "None of the inputs have
    # requires_grad=True" warning isn't just noise -- it can mean frozen
    # layers wrapping LoRA adapters silently don't get gradients flowing
    # through them correctly.
    model.enable_input_require_grads()

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

    print(f"Total examples skipped (train): {train_ds.skipped_total}")
    print(f"Total examples skipped (val):   {val_ds.skipped_total}")
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