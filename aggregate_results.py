"""
aggregate_results.py — Combine per-variant WT/TC/ET JSON outputs into:

    1. tab:perbandingan_5varian   -- mean DSC/HD95 per variant per region
    2. tab:wilcoxon               -- Wilcoxon signed-rank p-values for the
                                      4 paired comparisons defined in Bab III
                                      (sec:perancangan)
    3. tab:gate_interpretasi      -- mean [a1,a2] per region (Proposed only,
                                      or any variant evaluated with --log-gate)

Input
-----
Run evaluate_wt_tc_et.py once per variant first, e.g.:

    output/Baseline1_wt_tc_et.json
    output/Baseline2_wt_tc_et.json
    output/Ablasi1_wt_tc_et.json
    output/Ablasi2_wt_tc_et.json
    output/Proposed_wt_tc_et.json

Usage
-----
    python aggregate_results.py \
        --input_dir ./wt_tc_et_results \
        --output_dir ./bab4_tables

Output
------
    <output_dir>/tab_perbandingan_5varian.tex   (ready-to-paste LaTeX table body)
    <output_dir>/tab_wilcoxon.tex
    <output_dir>/tab_gate_interpretasi.tex      (only if gate_weights present)
    <output_dir>/aggregated_summary.json        (machine-readable, all numbers)

Notes
-----
- Wilcoxon test requires per-case arrays of EQUAL LENGTH and from the SAME
  case ordering across variants (i.e. both variants evaluated on the
  identical validation file list, in the identical order). This holds
  automatically if both used the same --dataset_json with shuffle=False,
  which evaluate_wt_tc_et.py enforces (DataLoader(..., shuffle=False)).
- NaN handling: HD95 can be NaN when a region is absent in both prediction
  and ground truth for a given case. Wilcoxon is run only on cases where
  BOTH variants have a non-NaN value for that metric/region (paired
  case-wise dropna). If too few valid pairs remain (<10), the result is
  reported as "insufficient data" rather than a potentially misleading
  p-value from a tiny sample.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


REGIONS = ["WT", "TC", "ET"]

# Paired comparisons exactly as designed in sec:perancangan (Bab III)
PAIRED_COMPARISONS = [
    ("Baseline1", "Baseline2", "Isolasi Loss (Dice vs Tversky)"),
    ("Baseline2", "Ablasi1",   "Isolasi Arsitektur SK"),
    ("Ablasi1",   "Proposed",  "Isolasi Cross-Modal Gating"),
    ("Ablasi2",   "Proposed",  "Isolasi Tversky Loss pada MMSK"),
]

VARIANT_DISPLAY_NAMES = {
    "Baseline1": "Baseline~1",
    "Baseline2": "Baseline~2",
    "Ablasi1":   "Ablasi~1",
    "Ablasi2":   "Ablasi~2",
    "Proposed":  "Model Usulan",
}

VARIANT_ORDER = ["Baseline1", "Baseline2", "Ablasi1", "Ablasi2", "Proposed"]


def load_all_variants(input_dir: Path) -> dict:
    """Load every *_wt_tc_et.json found in input_dir into a dict keyed by variant name."""
    results = {}
    for fp in sorted(input_dir.glob("*_wt_tc_et.json")):
        with open(fp) as f:
            data = json.load(f)
        results[data["variant"]] = data
    return results


def fmt(x: float, decimals: int = 3) -> str:
    """Format a float for LaTeX, handling NaN gracefully."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{decimals}f}"


# ── 1. Comparison table (tab:perbandingan_5varian) ──────────────────────────────
def build_comparison_table(results: dict) -> str:
    rows = []
    for variant in VARIANT_ORDER:
        if variant not in results:
            rows.append(
                f"    {VARIANT_DISPLAY_NAMES[variant]} ([TBD belum dievaluasi])"
                f" & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
            )
            continue
        data = results[variant]
        s = data["summary"]
        epoch = data.get("checkpoint_epoch", "?")
        dsc_vals = [s[r]["dsc_mean"] for r in REGIONS]
        dsc_avg = float(np.mean(dsc_vals))
        hd_vals = [s[r]["hd95_mean"] for r in REGIONS]
        hd_avg_valid = [v for v in hd_vals if not np.isnan(v)]
        hd_avg = float(np.mean(hd_avg_valid)) if hd_avg_valid else float("nan")

        row = (
            f"    {VARIANT_DISPLAY_NAMES[variant]} ({epoch}) & "
            f"{fmt(s['WT']['dsc_mean'])} & {fmt(s['TC']['dsc_mean'])} & "
            f"{fmt(s['ET']['dsc_mean'])} & {fmt(dsc_avg)} & "
            f"{fmt(s['WT']['hd95_mean'], 1)} & {fmt(s['TC']['hd95_mean'], 1)} & "
            f"{fmt(s['ET']['hd95_mean'], 1)} & {fmt(hd_avg, 1)} \\\\"
        )
        rows.append(row)
    return "\n".join(rows)


# ── 2. Wilcoxon table (tab:wilcoxon) ─────────────────────────────────────────────
def paired_wilcoxon(vals_a: list, vals_b: list, min_pairs: int = 10):
    """
    Run Wilcoxon signed-rank on paired arrays, dropping pairs where either
    value is NaN. Returns (p_value, n_pairs) or (None, n_pairs) if too few.
    """
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Mismatched case counts ({len(a)} vs {len(b)}) — variants must "
            f"be evaluated on the identical validation file list/order."
        )
    valid = ~np.isnan(a) & ~np.isnan(b)
    n_pairs = int(valid.sum())
    if n_pairs < min_pairs:
        return None, n_pairs
    a_v, b_v = a[valid], b[valid]
    if np.allclose(a_v, b_v):
        # wilcoxon() raises if all differences are zero
        return 1.0, n_pairs
    try:
        _, p = wilcoxon(a_v, b_v)
    except ValueError:
        return None, n_pairs
    return float(p), n_pairs


def build_wilcoxon_table(results: dict) -> str:
    rows = []
    for var_a, var_b, label in PAIRED_COMPARISONS:
        if var_a not in results or var_b not in results:
            rows.append(
                f"    {VARIANT_DISPLAY_NAMES.get(var_a, var_a)} vs "
                f"{VARIANT_DISPLAY_NAMES.get(var_b, var_b)} "
                f"& [TBD] & [TBD] & [TBD] \\\\"
            )
            continue

        p_cells = []
        for region in REGIONS:
            dsc_a = results[var_a]["per_case"][region]["dsc"]
            dsc_b = results[var_b]["per_case"][region]["dsc"]
            p, n_pairs = paired_wilcoxon(dsc_a, dsc_b)
            if p is None:
                p_cells.append(f"data tidak cukup (n={n_pairs})")
            else:
                marker = "*" if p < 0.05 else ""
                p_cells.append(f"{fmt(p, 4)}{marker}")

        rows.append(
            f"    {VARIANT_DISPLAY_NAMES[var_a]} vs {VARIANT_DISPLAY_NAMES[var_b]} "
            f"& {p_cells[0]} & {p_cells[1]} & {p_cells[2]} \\\\"
        )
    return "\n".join(rows)


# ── 3. Gate interpretability table (tab:gate_interpretasi) ─────────────────────
def build_gate_table(results: dict, preferred_variant: str = "Proposed") -> str | None:
    """Builds the gate table for one variant (default: Proposed model)."""
    candidates = [preferred_variant] + [v for v in VARIANT_ORDER if v != preferred_variant]
    data = None
    for v in candidates:
        if v in results and results[v].get("gate_weights") is not None:
            data = results[v]
            break
    if data is None:
        return None

    gw = data["gate_weights"]
    rows = []
    # Hypothesis from tab:mmsk_behavior (Bab III):
    #   ET -> a2 (kernel halus) tinggi
    #   WT -> a1 (kernel luas) tinggi
    #   TC -> menengah
    hypothesis_check = {
        "ET": lambda a1, a2: a2 > a1,
        "WT": lambda a1, a2: a1 > a2,
        "TC": lambda a1, a2: True,  # "menengah" has no strict pass/fail rule
    }
    for region in REGIONS:
        a1 = gw[region]["a1_mean"]
        a2 = gw[region]["a2_mean"]
        if region == "TC":
            sesuai = "(tidak ada kriteria ketat)"
        else:
            sesuai = "Ya" if hypothesis_check[region](a1, a2) else "Tidak"
        rows.append(
            f"    {region} & {fmt(a1)} & {fmt(a2)} & {sesuai} \\\\"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-variant WT/TC/ET JSONs into Bab IV tables."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_variants(input_dir)
    print(f"Loaded {len(results)} variant(s): {list(results.keys())}")
    missing = [v for v in VARIANT_ORDER if v not in results]
    if missing:
        print(f"WARNING: missing variants (will show [TBD] in tables): {missing}")

    # ── Table 1: comparison ────────────────────────────────────────────────
    comp_table = build_comparison_table(results)
    (output_dir / "tab_perbandingan_5varian.tex").write_text(comp_table)
    print("\n--- tab_perbandingan_5varian.tex ---")
    print(comp_table)

    # ── Table 2: Wilcoxon ──────────────────────────────────────────────────
    try:
        wilcoxon_table = build_wilcoxon_table(results)
        (output_dir / "tab_wilcoxon.tex").write_text(wilcoxon_table)
        print("\n--- tab_wilcoxon.tex ---")
        print(wilcoxon_table)
    except ValueError as e:
        print(f"\nWilcoxon table skipped: {e}")

    # ── Table 3: gate interpretability ─────────────────────────────────────
    gate_table = build_gate_table(results)
    if gate_table:
        (output_dir / "tab_gate_interpretasi.tex").write_text(gate_table)
        print("\n--- tab_gate_interpretasi.tex ---")
        print(gate_table)
    else:
        print(
            "\nGate interpretability table skipped — no variant has "
            "gate_weights (re-run evaluate_wt_tc_et.py with --log-gate "
            "--architecture mmsk for Ablasi2 and/or Proposed)."
        )

    # ── Machine-readable summary ────────────────────────────────────────────
    summary_out = {
        v: {"summary": results[v]["summary"],
            "checkpoint_epoch": results[v]["checkpoint_epoch"],
            "n_cases": results[v]["n_cases"]}
        for v in results
    }
    with open(output_dir / "aggregated_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved aggregated_summary.json -> {output_dir}")


if __name__ == "__main__":
    main()
