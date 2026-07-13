import json
import argparse

# Davide's reference numbers, as printed inline by evaluate_corrupted.py
# (Qwen2.5-VL-7B, DUDE_verified, 187 questions, paper's Explicit+OCR condition)
PAPER_REFERENCE = {
    "overall": {"AccP": 0.835, "AccD": 0.460},
    "by_complexity": {
        "1": {"AccP": 0.843},
        "2": {"AccP": 0.847},
        "3": {"AccP": 0.731},
    },
}


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, width=8):
    return f"{v:.4f}".rjust(width) if v is not None else "N/A".rjust(width)


def delta_str(a, b, width=8):
    if a is None or b is None:
        return "N/A".rjust(width)
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}".rjust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", required=True)
    ap.add_argument("--finetuned", required=True)
    args = ap.parse_args()

    zs = load(args.zeroshot)
    ft = load(args.finetuned)

    print("=" * 78)
    print("VRD-UQA CORRUPTED BENCHMARK -- 3-WAY COMPARISON")
    print("=" * 78)
    print(f"{'':20} {'Paper ref':>10} {'Zero-shot':>10} {'Fine-tuned':>10} {'FT vs ZS':>10} {'FT vs paper':>12}")
    print("-" * 78)

    ref = PAPER_REFERENCE["overall"]
    print(f"{'AccP (page-level)':20} {fmt(ref['AccP'], 10)} {fmt(zs['AccP'], 10)} "
          f"{fmt(ft['AccP'], 10)} {delta_str(zs['AccP'], ft['AccP'], 10)} "
          f"{delta_str(ref['AccP'], ft['AccP'], 12)}")
    print(f"{'AccD (doc-level)':20} {fmt(ref['AccD'], 10)} {fmt(zs['AccD'], 10)} "
          f"{fmt(ft['AccD'], 10)} {delta_str(zs['AccD'], ft['AccD'], 10)} "
          f"{delta_str(ref['AccD'], ft['AccD'], 12)}")
    print(f"{'Hallucination rate':20} {'N/A':>10} {fmt(zs['hallucination_rate'], 10)} "
          f"{fmt(ft['hallucination_rate'], 10)} "
          f"{delta_str(zs['hallucination_rate'], ft['hallucination_rate'], 10)} {'N/A':>12}")
    print()

    print("Breakdown by complexity (AccP):")
    print(f"{'':10} {'Paper ref':>10} {'Zero-shot':>10} {'Fine-tuned':>10}")
    for level in ["1", "2", "3"]:
        ref_c = PAPER_REFERENCE["by_complexity"].get(level, {}).get("AccP")
        zs_c = zs.get("complexity_breakdown", {}).get(level, {}).get("AccP")
        ft_c = ft.get("complexity_breakdown", {}).get(level, {}).get("AccP")
        print(f"C{level:<9} {fmt(ref_c, 10)} {fmt(zs_c, 10)} {fmt(ft_c, 10)}")

    print()
    print("=" * 78)
    print("COMPLIANCE CHECK (zero-shot vs. paper reference)")
    print("=" * 78)
    accp_diff = abs(zs["AccP"] - ref["AccP"])
    accd_diff = abs(zs["AccD"] - ref["AccD"])
    print(f"  |zero-shot AccP - paper AccP| = {accp_diff:.4f}")
    print(f"  |zero-shot AccD - paper AccD| = {accd_diff:.4f}")
    if accp_diff < 0.03 and accd_diff < 0.05:
        print("  -> Within a reasonable margin of the paper's reported numbers.")
        print("     The fine-tuned comparison above can be trusted.")
    else:
        print("  -> LARGER than expected gap from the paper's numbers.")
        print("     Before trusting the fine-tuned delta, check: prompt wording,")
        print("     OCR extraction/sort order, refusal-phrase detection, and")
        print("     whether DUDE_verified.json / images_dir match what the")
        print("     paper's own run used.")
    print("=" * 78)


if __name__ == "__main__":
    main()