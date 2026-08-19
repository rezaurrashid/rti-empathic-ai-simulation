"""
Create Table 1 and Table 5 from problem-specific files.
Inputs:
  evaluation_results/problem_instance_labels.csv
  evaluation_results/problem_instance_recognition.csv
Outputs:
  evaluation_results/table1_problem_specific_severity_distribution.csv
  evaluation_results/table5_problem_specific_triage_performance.csv
  evaluation_results/table5_problem_specific_bootstrap_ci.csv
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "./evaluation_results"
OUTPUT_DIR = "./evaluation_results/tables"
SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]

def make_table1(problem_labels_df: pd.DataFrame) -> pd.DataFrame:
    df = problem_labels_df.copy()
    df["problem_severity"] = df["problem_severity"].astype(str).str.strip().str.upper()
    total = len(df)
    rows = []
    for sev in SEVERITY_ORDER:
        n = int((df["problem_severity"] == sev).sum())
        pct = 100 * n / total if total else 0.0
        rows.append({
            "severity": sev.title(),
            "problem_mentions_n": n,
            "problem_mentions_percent": round(pct, 1),
            "table_value": f"{n:,} ({pct:.1f}%)"
        })
    return pd.DataFrame(rows)

def make_table5(problem_recognition_df: pd.DataFrame) -> pd.DataFrame:
    df = problem_recognition_df.copy()
    df["problem_severity"] = df["problem_severity"].astype(str).str.strip().str.upper()
    for col in ["recognized", "handled", "missed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    rows = []
    for sev in SEVERITY_ORDER:
        sub = df[df["problem_severity"] == sev]
        n = len(sub)
        rec_n = int(sub["recognized"].sum())
        hand_n = int(sub["handled"].sum())
        miss_n = int(sub["missed"].sum())
        rec_pct = 100 * rec_n / n if n else 0.0
        hand_pct = 100 * hand_n / n if n else 0.0
        miss_pct = 100 * miss_n / n if n else 0.0
        gap = rec_pct - hand_pct
        rows.append({
            "severity": sev.title(),
            "problem_mentions_n": n,
            "recognized_n": rec_n,
            "recognition_rate_percent": round(rec_pct, 1),
            "recognized_table_value": f"{rec_n:,} ({rec_pct:.1f}%)",
            "handled_n": hand_n,
            "handling_rate_percent": round(hand_pct, 1),
            "handled_table_value": f"{hand_n:,} ({hand_pct:.1f}%)",
            "missed_n": miss_n,
            "missed_rate_percent": round(miss_pct, 1),
            "missed_table_value": f"{miss_n:,} ({miss_pct:.1f}%)",
            "recognition_to_action_gap_points": round(gap, 1)
        })
    return pd.DataFrame(rows)

def bootstrap_ci(problem_recognition_df: pd.DataFrame, n_boot=1000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = problem_recognition_df.copy()
    df["problem_severity"] = df["problem_severity"].astype(str).str.strip().str.upper()
    for col in ["recognized", "handled", "missed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    conv_ids = df["conversation_id"].dropna().unique()
    records = []
    for b in range(n_boot):
        sampled = rng.choice(conv_ids, size=len(conv_ids), replace=True)
        boot = pd.concat([df[df["conversation_id"] == cid] for cid in sampled], ignore_index=True)
        for sev in SEVERITY_ORDER:
            sub = boot[boot["problem_severity"] == sev]
            n = len(sub)
            if n == 0:
                continue
            rec = 100 * sub["recognized"].mean()
            hand = 100 * sub["handled"].mean()
            miss = 100 * sub["missed"].mean()
            gap = rec - hand
            records.extend([
                {"severity": sev.title(), "metric": "recognition_rate_percent", "value": rec},
                {"severity": sev.title(), "metric": "handling_rate_percent", "value": hand},
                {"severity": sev.title(), "metric": "missed_rate_percent", "value": miss},
                {"severity": sev.title(), "metric": "recognition_to_action_gap_points", "value": gap},
            ])

    boot_df = pd.DataFrame(records)
    ci = boot_df.groupby(["severity", "metric"])["value"].quantile([0.025, 0.975]).unstack().reset_index()
    ci.columns = ["severity", "metric", "ci_lower", "ci_upper"]
    ci["ci_lower"] = ci["ci_lower"].round(1)
    ci["ci_upper"] = ci["ci_upper"].round(1)
    return ci


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build severity-stratified performance tables and bootstrap confidence intervals."
    )
    parser.add_argument(
        "--problem-labels",
        default="outputs/evaluation/problem_instance_labels.csv",
    )
    parser.add_argument(
        "--recognition",
        default="outputs/evaluation/problem_instance_recognition.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/summary",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_csv(args.problem_labels)
    rec = pd.read_csv(args.recognition)

    table1 = make_table1(labels)
    table5 = make_table5(rec)
    ci = bootstrap_ci(rec, n_boot=args.bootstrap, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table1.to_csv(output_dir / "severity_distribution.csv", index=False)
    table5.to_csv(output_dir / "severity_performance.csv", index=False)
    ci.to_csv(output_dir / "severity_bootstrap_ci.csv", index=False)

    print(table1.to_string(index=False))
    print(table5.to_string(index=False))
    print(ci.to_string(index=False))
