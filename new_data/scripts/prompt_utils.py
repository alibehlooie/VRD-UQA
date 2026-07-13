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
print(">>> FINGERPRINT: prompt_utils.py VERSION=LLM_ONLY_LORA_FIX_2026_07_12 "
      f"(_PATCH_FACTOR will be defined below) imported from {__file__} <<<", flush=True)

_PATCH_FACTOR = 112  # Qwen2.5-VL-7B window_size in pixels (8 patches of 14px each).
# NOTE: 28 (patch_size*merge_size) only guarantees clean 2x2 MERGE alignment.
# It does NOT guarantee clean WINDOW alignment (window_size=112px=8 patches),
# and a dimension that isn't a whole number of windows produces a partial
# edge window whose patch count can fail the merge reshape deep in
# model.forward() -- this was the actual root cause of a 100%-reproducible
# "shape '[0,4,-1]' is invalid" crash across every single training image
# (confirmed via diagnostic: every failing image had grid h or w not
# divisible by 8, even though all were correctly divisible by 2). Since
# 112 is itself a multiple of 28, rounding to 112 automatically satisfies
# both constraints at once.


def load_and_resize_image(path: Union[str, Path]) -> Image.Image:
    """
    Resize to satisfy BOTH IMAGE_MAX_PIXELS and IMAGE_MIN_PIXELS, with
    dimensions rounded to multiples of the patch size (28) -- mirrors
    Qwen2.5-VL's own smart_resize logic.

    The original version of this function (copied from
    evaluate_corrupted.py) only capped the MAXIMUM size and never enforced
    a minimum or patch-size rounding. For naturally small page images,
    that meant no upscaling ever happened, and Qwen2.5-VL's vision merger
    (which groups patches in blocks of 4) could end up with fewer than 4
    total patches for the whole image, crashing with a reshape error
    ("shape '[0, 4, -1]' is invalid ...") deep inside the model forward
    pass -- not something a dataset-level try/except can catch, since the
    tensors are valid, just too small. This version fixes that for both
    training and eval.
    """
    image = Image.open(path).convert("RGB")
    w, h = image.width, image.height

    h_bar = max(_PATCH_FACTOR, round(h / _PATCH_FACTOR) * _PATCH_FACTOR)
    w_bar = max(_PATCH_FACTOR, round(w / _PATCH_FACTOR) * _PATCH_FACTOR)
    area = h_bar * w_bar

    if area > IMAGE_MAX_PIXELS:
        beta = (h * w / IMAGE_MAX_PIXELS) ** 0.5
        h_bar = max(_PATCH_FACTOR, int(h / beta // _PATCH_FACTOR) * _PATCH_FACTOR)
        w_bar = max(_PATCH_FACTOR, int(w / beta // _PATCH_FACTOR) * _PATCH_FACTOR)
    elif area < IMAGE_MIN_PIXELS:
        beta = (IMAGE_MIN_PIXELS / (h * w)) ** 0.5
        h_bar = -(-int(h * beta) // _PATCH_FACTOR) * _PATCH_FACTOR  # ceil to multiple
        w_bar = -(-int(w * beta) // _PATCH_FACTOR) * _PATCH_FACTOR

    if (h_bar, w_bar) != (h, w):
        image = image.resize((w_bar, h_bar), Image.LANCZOS)
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

    min_pixels/max_pixels are passed explicitly per-image so
    process_vision_info uses OUR bounds rather than its own internal
    defaults -- it re-derives image size independently of whatever we
    already did to the PIL object, so relying on pre-resizing alone isn't
    sufficient.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", "image": image,
                    "min_pixels": IMAGE_MIN_PIXELS, "max_pixels": IMAGE_MAX_PIXELS,
                },
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
    content = [
        {
            "type": "image", "image": img,
            "min_pixels": IMAGE_MIN_PIXELS, "max_pixels": IMAGE_MAX_PIXELS,
        }
        for img in images
    ]
    content.append({"type": "text", "text": build_prompt(question, ocr_text)})
    return [{"role": "user", "content": content}]


def build_target(answer: Optional[str], is_answerable: bool) -> str:
    if not is_answerable or not answer:
        return REFUSAL_TARGET
    return answer