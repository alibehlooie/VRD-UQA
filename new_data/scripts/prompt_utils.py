"""
prompt_utils.py

Single source of truth for prompt construction, OCR extraction/sorting,
refusal detection, and image handling -- shared by fine_tune.py and
evaluate_corrupted.py.

AUTHORITATIVE SOURCE: build_prompt(), is_refusal(), REFUSAL_PHRASES,
IMAGE_MIN_PIXELS/IMAGE_MAX_PIXELS, and the (y0, x0) sort convention are
copied verbatim from evaluate_corrupted.py (the paper-methodology-locked
eval script), NOT invented here. If you need to change wording, sort
order, or the refusal phrase, change it here and evaluate_corrupted.py
picks it up automatically -- do not fork a second copy.

MAX_OCR_CHARS = 8000 (updated from the eval script's original 2000 to match
training config, per decision on 2026-07-07 -- evaluate_corrupted.py has
been updated to import this value rather than hardcode 2000).

Two data sources need OCR extraction, with different schemas:
  - Training side: per-page patch-json files (bbox/label/content_type/
    text/caption/patch_image_path), one flat list per page.
  - Eval side (DUDE_verified.json): embedded layout_analysis.pages, each
    page a dict of objects with BBOX/ObjectType/ObjectTypeID/OCR (figure
    captions already merged into the OCR field, no separate field).

Both funnel into the same core sort+join so train and eval never diverge.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

from PIL import Image

# ---------------------------------------------------------------------------
# Constants (authoritative)
# ---------------------------------------------------------------------------
IMAGE_MIN_PIXELS = 256 * 28 * 28
IMAGE_MAX_PIXELS = 1280 * 28 * 28
MAX_OCR_CHARS = 8000  # was 2000 in the original eval script; unified 2026-07-07

# Mirrors the paper's Gemini 2.0 Flash output standardizer
REFUSAL_PHRASES = [
    "unable to determine",
    "cannot determine",
    "not determinable",
    "cannot be determined",
    "not possible to determine",
    "no information",
    "not provided",
    "not available",
    "not mentioned",
    "not found",
    "not present",
    "not specified",
    "not stated",
    "not given",
    "does not provide",
    "does not contain",
    "does not mention",
    "cannot find",
    "no answer",
]

# The canonical phrase used as the SFT training target for unanswerable
# questions. Must be one of REFUSAL_PHRASES (it is -- first entry) so
# is_refusal() recognizes it, and matches the eval prompt's own guideline
# wording ("If uncertain, return 'Unable to determine'").
REFUSAL_TARGET = "Unable to determine"


def is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Core reading-order sort + join (shared by both schema adapters below)
# ---------------------------------------------------------------------------
def _sort_and_join(items: List[Tuple[list, str]], max_chars: int = MAX_OCR_CHARS) -> str:
    """
    items: list of (bbox, text) where bbox = [x0, y0, x1, y1].
    Sort key is (y0, x0) -- plain top-to-bottom-then-left-to-right, NO
    column-splitting heuristic. This is the eval script's verified
    convention (~0.55 raw monotonicity without sorting, confirmed fixed
    by this simple sort) -- do not "improve" this into a 2-column-aware
    sort without re-validating against the eval script, or you reintroduce
    train/eval divergence.
    """
    valid = [(bbox, text.strip()) for bbox, text in items if text and text.strip()]
    valid.sort(key=lambda x: (x[0][1], x[0][0]))
    joined = " ".join(text for _, text in valid)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined


# ---------------------------------------------------------------------------
# Schema adapter: training-side patch-json files
# ---------------------------------------------------------------------------
def load_patch_file(patch_path: Path) -> List[dict]:
    with open(patch_path) as f:
        return json.load(f)


def extract_ocr_from_patch_file(patches: List[dict], max_chars: int = MAX_OCR_CHARS) -> str:
    """
    Training-side OCR extraction from a patch-json file (one page).
    content_type == "ocr"      -> use `text`
    content_type == "caption"  -> use `caption` (figure captions are
                                   merged in as plain text, matching how
                                   DUDE_verified's OCR field already
                                   contains captions with no special
                                   wrapping/prefix)
    content_type == "skip"     -> dropped (abandon elements)
    """
    items = []
    for p in patches:
        ct = p.get("content_type")
        if ct == "ocr" and p.get("text"):
            items.append((p["bbox"], p["text"]))
        elif ct == "caption" and p.get("caption"):
            items.append((p["bbox"], p["caption"]))
        # ct == "skip" -> dropped
    return _sort_and_join(items, max_chars=max_chars)


def patch_file_path(patch_dir: Path, doc_id: str, page: int) -> Path:
    return patch_dir / f"{doc_id}_{page}.json"


def page_image_path(image_dir: Path, doc_id: str, page: int, ext: str = "jpg") -> Path:
    return image_dir / f"{doc_id}_{page}.{ext}"


# ---------------------------------------------------------------------------
# Schema adapter: eval-side DUDE_verified.json embedded layout_analysis
# ---------------------------------------------------------------------------
def get_all_page_ids(item: dict) -> list:
    """Return ALL page ids in the document (from layout_analysis)."""
    pages = item.get("layout_analysis", {}).get("pages", {})
    return list(pages.keys())


def extract_ocr_from_layout_analysis(item: dict, page_id: str, max_chars: int = MAX_OCR_CHARS) -> str:
    """Eval-side OCR extraction from DUDE_verified.json's embedded
    layout_analysis for a given page_id."""
    try:
        page_data = item["layout_analysis"]["pages"][page_id]
        objs = page_data.get("layout_analysis", {})
        items = [
            (obj.get("BBOX", [0, 0, 0, 0]), obj.get("OCR", ""))
            for obj in objs.values()
            if isinstance(obj, dict) and obj.get("OCR", "").strip()
        ]
        return _sort_and_join(items, max_chars=max_chars)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Image loading (matches evaluate_corrupted.py's resize-if-over-max logic)
# ---------------------------------------------------------------------------
def load_and_resize_image(path: Union[str, Path]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width * image.height > IMAGE_MAX_PIXELS:
        scale = (IMAGE_MAX_PIXELS / (image.width * image.height)) ** 0.5
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS,
        )
    return image


# ---------------------------------------------------------------------------
# Prompt / chat-message construction (verbatim from evaluate_corrupted.py)
# ---------------------------------------------------------------------------
def build_prompt(question: str, ocr_text: str) -> str:
    header = (
        "You are an AI assistant specialized in analyzing document images and text.\n"
        "Your task is to answer questions about the document image content precisely.\n"
    )
    ocr_line = f"\nFor this question, you have the following OCR text:\n{ocr_text}\n" if ocr_text else ""
    footer = (
        "\nGuidelines:\n"
        "- Provide concise, focused answers (single word or short phrase preferred)\n"
        "- Base your answer on both the image and the provided OCR text\n"
        "- If uncertain, return 'Unable to determine'\n"
        "- If you can't find the answer, return 'Unable to determine'\n"
        f"\nQuestion: {question}"
    )
    return header + ocr_line + footer


def build_messages(image: Union[str, Path, Image.Image], question: str, ocr_text: str) -> List[dict]:
    """
    Single user-role message, matching evaluate_corrupted.py's
    run_inference exactly -- NO system role. `image` can be a path or an
    already-loaded PIL.Image (qwen_vl_utils.process_vision_info accepts
    either).
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(question, ocr_text)},
            ],
        }
    ]


def build_messages_multi(images: List[Union[str, Path, Image.Image]], question: str, ocr_text: str) -> List[dict]:
    """
    Multi-page variant: one user-role message with multiple image blocks
    followed by the same build_prompt() text used everywhere else. Only
    used on the training side for answerable items with no page bbox
    (e.g. "abstractive" answer_type) -- the eval side (DUDE_verified) is
    always single-page per the paper's methodology, so this is additive,
    not a change to the shared single-page path.
    """
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": build_prompt(question, ocr_text)})
    return [{"role": "user", "content": content}]


def build_target(answer: Optional[str], is_answerable: bool) -> str:
    if not is_answerable or not answer:
        return REFUSAL_TARGET
    return answer