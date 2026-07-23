from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_MIN_PIXELS = 256 * 28 * 28
IMAGE_MAX_PIXELS = 1280 * 28 * 28
MAX_OCR_CHARS = 8000

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

# The training target for unanswerable questions. Needs to be one of
# REFUSAL_PHRASES (it's the first entry) so is_refusal() picks it up.
REFUSAL_TARGET = "Unable to determine"


def is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Reading-order sort + join
# ---------------------------------------------------------------------------
def _sort_and_join(items: List[Tuple[list, str]], max_chars: int = MAX_OCR_CHARS) -> str:
    """
    items: list of (bbox, text), bbox = [x0, y0, x1, y1].
    Sorted top-to-bottom then left-to-right by (y0, x0) -- plain sort, no
    column-splitting. This matches what the eval side already does, so
    don't get clever with a 2-column-aware sort here without updating
    both places at once.
    """
    valid = [(bbox, text.strip()) for bbox, text in items if text and text.strip()]
    valid.sort(key=lambda x: (x[0][1], x[0][0]))
    joined = " ".join(text for _, text in valid)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined


# ---------------------------------------------------------------------------
# Training-side OCR (patch-json files)
# ---------------------------------------------------------------------------
def load_patch_file(patch_path: Path) -> List[dict]:
    with open(patch_path) as f:
        return json.load(f)


def extract_ocr_from_patch_file(patches: List[dict], max_chars: int = MAX_OCR_CHARS) -> str:
    """
    content_type == "ocr"      -> use `text`
    content_type == "caption"  -> use `caption` (figure captions, folded
                                   in as plain text)
    content_type == "skip"     -> dropped (headers/footers/etc.)
    """
    items = []
    for p in patches:
        ct = p.get("content_type")
        if ct == "ocr" and p.get("text"):
            items.append((p["bbox"], p["text"]))
        elif ct == "caption" and p.get("caption"):
            items.append((p["bbox"], p["caption"]))
    return _sort_and_join(items, max_chars=max_chars)


def patch_file_path(patch_dir: Path, doc_id: str, page: int) -> Path:
    return patch_dir / f"{doc_id}_{page}.json"


def page_image_path(image_dir: Path, doc_id: str, page: int, ext: str = "jpg") -> Path:
    return image_dir / f"{doc_id}_{page}.{ext}"


# ---------------------------------------------------------------------------
# Eval-side OCR (DUDE_verified.json's embedded layout_analysis)
# ---------------------------------------------------------------------------
def get_all_page_ids(item: dict) -> list:
    """All page ids for this document, from its embedded layout_analysis."""
    pages = item.get("layout_analysis", {}).get("pages", {})
    return list(pages.keys())


def extract_ocr_from_layout_analysis(item: dict, page_id: str, max_chars: int = MAX_OCR_CHARS) -> str:
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
# Image loading / resizing
# ---------------------------------------------------------------------------
_PATCH_FACTOR = 112  # window size in pixels for Qwen2.5-VL-7B (8 patches of 14px)
# Rounding to a multiple of 28 (patch*merge) is enough for the vision
# merger's 2x2 grouping, but not enough for the windowed attention, which
# needs whole windows of 8 patches per side. An image sized to a 28-
# multiple but not a 112-multiple can produce a partial edge window that
# crashes deep in the model's forward pass. 112 is itself a multiple of
# 28, so rounding to it satisfies both at once.

_MIN_DOWNSCALE_PIXELS = 64 * 28 * 28  # floor so long documents stay legible
_MULTIPAGE_MAX_PIXELS_PER_PAGE = 300_000
# Ceiling on top of the ratio formula below. Without it, a document with
# only 2-3 pages can get a much larger per-page budget than a 6+ page one
# (since there's less need to shrink to hit the target), and a handful of
# large images turned out to cost more GPU memory than many small ones at
# a similar total token count. 300K keeps some margin under the largest
# value that measured safely in testing.


def compute_adaptive_page_pixels(n_pages: int, target_pages_equivalent: int = 5,
                                  base_max_pixels: int = IMAGE_MAX_PIXELS) -> int:
    """
    For multi-page documents, show every page but shrink each one's
    resolution so the total image-token cost stays roughly equivalent to
    `target_pages_equivalent` pages at full resolution -- regardless of
    how long the document actually is. A short doc gets full resolution;
    a long one gets each page scaled down proportionally. This avoids
    having to pick which pages to drop (which risks cutting the one page
    that actually has the answer).
    """
    if n_pages <= 1:
        return base_max_pixels
    if n_pages <= target_pages_equivalent:
        px = base_max_pixels
    else:
        px = max(_MIN_DOWNSCALE_PIXELS, int(base_max_pixels * target_pages_equivalent / n_pages))
    return min(px, _MULTIPAGE_MAX_PIXELS_PER_PAGE)


def load_and_resize_image(path: Union[str, Path], max_pixels: int = None) -> Image.Image:
    """
    Resize to fit within max_pixels (defaults to IMAGE_MAX_PIXELS) while
    respecting IMAGE_MIN_PIXELS as a floor, rounding both dimensions to
    multiples of _PATCH_FACTOR. Pass max_pixels explicitly (e.g. via
    compute_adaptive_page_pixels) for multi-page examples that need a
    smaller per-page budget than the single-page default.
    """
    effective_max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_PIXELS

    image = Image.open(path).convert("RGB")
    w, h = image.width, image.height

    h_bar = max(_PATCH_FACTOR, round(h / _PATCH_FACTOR) * _PATCH_FACTOR)
    w_bar = max(_PATCH_FACTOR, round(w / _PATCH_FACTOR) * _PATCH_FACTOR)
    area = h_bar * w_bar

    if area > effective_max_pixels:
        beta = (h * w / effective_max_pixels) ** 0.5
        h_bar = max(_PATCH_FACTOR, int(h / beta // _PATCH_FACTOR) * _PATCH_FACTOR)
        w_bar = max(_PATCH_FACTOR, int(w / beta // _PATCH_FACTOR) * _PATCH_FACTOR)
    elif area < IMAGE_MIN_PIXELS:
        beta = (IMAGE_MIN_PIXELS / (h * w)) ** 0.5
        h_bar = -(-int(h * beta) // _PATCH_FACTOR) * _PATCH_FACTOR
        w_bar = -(-int(w * beta) // _PATCH_FACTOR) * _PATCH_FACTOR

    if (h_bar, w_bar) != (h, w):
        image = image.resize((w_bar, h_bar), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Prompt / chat-message construction
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
    Single user-role message, no system role -- matches the eval script's
    run_inference exactly. `image` can be a path or an already-loaded
    PIL.Image. min_pixels/max_pixels are passed explicitly per-image so
    process_vision_info uses our bounds rather than recomputing its own.
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


def build_messages_multi(images: List[Union[str, Path, Image.Image]], question: str, ocr_text: str,
                          max_pixels: int = None) -> List[dict]:
    """
    Multi-page variant of build_messages: several image blocks in one
    user message, same prompt text. Used on the training side for
    answerable questions that don't have a verified answer page.

    max_pixels should be the SAME value passed to load_and_resize_image()
    for these images (e.g. from compute_adaptive_page_pixels), not the
    global IMAGE_MAX_PIXELS. process_vision_info re-derives image size
    from whatever min_pixels/max_pixels are in the message dict, using
    its own rounding (factor=28), regardless of how we already resized
    the PIL object -- if the message declares a larger budget than what
    we actually resized to, it recomputes a different, larger size that
    isn't guaranteed to be window-aligned (multiple of 112), which is
    what caused a "Degenerate image grid" failure on nearly every
    multi-page example. Pinning min_pixels == max_pixels == our own
    target leaves the library nothing to recompute: since that target is
    already a multiple of 112 (hence also of 28), its own rounding is a
    no-op and the size comes back unchanged.
    """
    effective_max = max_pixels if max_pixels is not None else IMAGE_MAX_PIXELS
    content = [
        {
            "type": "image", "image": img,
            "min_pixels": effective_max, "max_pixels": effective_max,
        }
        for img in images
    ]
    content.append({"type": "text", "text": build_prompt(question, ocr_text)})
    return [{"role": "user", "content": content}]


def build_target(answer: Optional[str], is_answerable: bool) -> str:
    if not is_answerable or not answer:
        return REFUSAL_TARGET
    return answer