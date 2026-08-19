#!/usr/bin/env python
"""Generate core manuscript tables and figures from final evaluation outputs.

Inputs
------
automated_metrics.csv
assistant_recognition_rates_problem_specific.csv
problem_instance_labels.csv
problem_instance_recognition.csv
human_evaluation_summary.csv (optional, aggregate only)

Outputs
-------
Main Figure 2: automated dialogue characterization
Main Figure 3: severity-stratified triage performance
Supplementary Figure S1: Spearman correlation matrix
Supplementary Figure S5: problem distribution and severity profile
CSV source tables used for these figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import spearmanr


SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]


def normalize_rate(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x / 100.0 if x.dropna().max() > 1.5 else x


def pretty_label(x: str) -> str:
    return x.replace("_", " ").title().replace("Mgt", "Management")


def main_figure_2(auto: pd.DataFrame, rec: pd.DataFrame, labels: pd.DataFrame, out: Path):
    df = auto.merge(rec, on="conversation_id", how="inner", suffixes=("", "_rec"))
    for c in ["recognition_rate", "handling_rate", "missed_issue_rate"]:
        df[c] = normalize_rate(df[c])

    high_load = (
        labels.assign(is_high=(labels["problem_severity"].str.upper() == "HIGH").astype(int))
        .groupby("conversation_id")
        .agg(total=("problem_severity", "size"), high=("is_high", "sum"))
    )
    high_load["high_severity_percent"] = 100 * high_load["high"] / high_load["total"]
    df = df.merge(high_load[["high_severity_percent"]], left_on="conversation_id", right_index=True, how="left")

    empathy_col = "empathy_overall"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(df[empathy_col].dropna(), bins=20)
    ax.set_title("A. Empathy score distribution")
    ax.set_xlabel("Empathy score")
    ax.set_ylabel("Conversations")

    ax = axes[0, 1]
    ax.hist(df["recognition_rate"].dropna(), bins=20, alpha=0.6, label="Recognized")
    ax.hist(df["handling_rate"].dropna(), bins=20, alpha=0.6, label="Handled")
    ax.hist(df["missed_issue_rate"].dropna(), bins=20, alpha=0.6, label="Missed")
    ax.set_title("B. Recognition, handling, and missed-problem rates")
    ax.set_xlabel("Rate")
    ax.legend()

    ax = axes[1, 0]
    vals = [
        auto["action_ratio"].mean(),
        auto["concrete_suggestions"].mean(),
        auto["information_provision"].mean(),
    ]
    ax.bar(["Action ratio", "Concrete suggestions", "Information provision"], vals)
    ax.set_ylim(0, 1)
    ax.set_title("C. Action-orientation components")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    sc = ax.scatter(
        df[empathy_col],
        df["recognition_rate"],
        c=df["high_severity_percent"],
        alpha=0.8,
    )
    ax.set_title("D. Empathy vs recognition")
    ax.set_xlabel("Empathy score")
    ax.set_ylabel("Recognition rate")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("High-severity problem mentions (%)")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure2_Automated_Characterization.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bootstrap_severity(rec: pd.DataFrame, n_boot: int = 1000, seed: int = 42):
    df = rec.copy()
    df["problem_severity"] = df["problem_severity"].astype(str).str.upper()
    conv_ids = sorted(df["conversation_id"].unique())
    index = {c: i for i, c in enumerate(conv_ids)}
    arr = np.zeros((len(conv_ids), len(SEVERITY_ORDER), 4), dtype=float)

    grouped = df.groupby(["conversation_id", "problem_severity"]).agg(
        n=("recognized", "size"),
        recognized=("recognized", "sum"),
        handled=("handled", "sum"),
        missed=("missed", "sum"),
    ).reset_index()

    for _, row in grouped.iterrows():
        if row["problem_severity"] not in SEVERITY_ORDER:
            continue
        i = index[row["conversation_id"]]
        j = SEVERITY_ORDER.index(row["problem_severity"])
        arr[i, j] = [row["n"], row["recognized"], row["handled"], row["missed"]]

    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(conv_ids), size=len(conv_ids))
        sums = arr[idx].sum(axis=0)
        for j, sev in enumerate(SEVERITY_ORDER):
            n, r, h, m = sums[j]
            if n == 0:
                continue
            rec_rate = 100 * r / n
            hand_rate = 100 * h / n
            miss_rate = 100 * m / n
            records.extend(
                [
                    (sev, "recognized", rec_rate),
                    (sev, "handled", hand_rate),
                    (sev, "missed", miss_rate),
                    (sev, "gap", rec_rate - hand_rate),
                ]
            )
    boot = pd.DataFrame(records, columns=["severity", "metric", "value"])
    return (
        boot.groupby(["severity", "metric"])["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "ci_lower", 0.975: "ci_upper"})
        .reset_index()
    )


def main_figure_3(rec: pd.DataFrame, out: Path):
    d = rec.copy()
    d["problem_severity"] = d["problem_severity"].astype(str).str.upper()
    summary = (
        d.groupby("problem_severity")
        .agg(
            n=("recognized", "size"),
            recognized=("recognized", "mean"),
            handled=("handled", "mean"),
            missed=("missed", "mean"),
        )
        .reindex(SEVERITY_ORDER)
    )
    for c in ["recognized", "handled", "missed"]:
        summary[c] *= 100
    summary["gap"] = summary["recognized"] - summary["handled"]

    ci = bootstrap_severity(d, 1000, 42)
    x = np.arange(len(SEVERITY_ORDER))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 6))
    metrics = ["recognized", "handled", "missed"]
    offsets = [-width, 0, width]

    for metric, offset in zip(metrics, offsets):
        vals = summary[metric].to_numpy()
        lo, hi = [], []
        for sev, val in zip(SEVERITY_ORDER, vals):
            row = ci[(ci["severity"] == sev) & (ci["metric"] == metric)].iloc[0]
            lo.append(val - row["ci_lower"])
            hi.append(row["ci_upper"] - val)
        bars = ax.bar(x + offset, vals, width, label=metric.title(), yerr=[lo, hi], capsize=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=9)

    for i, sev in enumerate(SEVERITY_ORDER):
        ax.text(i, 96, f"Gap = {summary.loc[sev, 'gap']:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{sev.title()}\n(n={int(summary.loc[sev, 'n']):,})" for sev in SEVERITY_ORDER]
    )
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Severity-Stratified Triage Performance")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure3_Severity_Stratified_Performance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary.reset_index().to_csv(out / "Figure3_source_data.csv", index=False)
    ci.to_csv(out / "Figure3_bootstrap_ci.csv", index=False)


def supplementary_s1(auto: pd.DataFrame, rec: pd.DataFrame, out: Path):
    df = auto.merge(rec, on="conversation_id", how="inner")
    selected = {
        "mean_relevance": "Semantic relevance",
        "contextual_coherence": "Contextual coherence",
        "empathy_overall": "Empathy score",
        "sdoh_overall_relevance": "SDoH relevance",
        "action_ratio": "Action ratio",
        "concrete_suggestions": "Concrete suggestions",
        "information_provision": "Information provision",
        "recognition_rate": "Recognition rate",
        "handling_rate": "Handling rate",
        "resolution_rate": "Resolution rate",
        "missed_issue_rate": "Missed-problem rate",
        "avg_problems_per_turn": "Problems/turn",
    }
    cols = [c for c in selected if c in df.columns]
    x = df[cols].copy()
    for c in ["recognition_rate", "handling_rate", "resolution_rate", "missed_issue_rate"]:
        if c in x:
            x[c] = normalize_rate(x[c])
    corr = x.corr(method="spearman")
    corr.index = [selected[c] for c in corr.index]
    corr.columns = [selected[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=50, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"FigureS1_Correlation_Matrix.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(out / "FigureS1_source_data.csv")


def supplementary_s5(labels: pd.DataFrame, out: Path):
    d = labels.copy()
    d["problem_category"] = d["problem_category"].astype(str).str.upper()
    d["problem_severity"] = d["problem_severity"].astype(str).str.upper()

    freq = (
        d.groupby("problem_category")
        .size()
        .sort_values(ascending=False)
        .rename("total_mentions")
        .reset_index()
    )
    order = freq["problem_category"].tolist()
    sev = (
        d.groupby(["problem_category", "problem_severity"])
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    for s in ["LOW", "MEDIUM", "HIGH"]:
        if s not in sev.columns:
            sev[s] = 0
    sev_pct = sev[["LOW", "MEDIUM", "HIGH"]].div(sev.sum(axis=1), axis=0) * 100

    labels_pretty = [pretty_label(c) for c in order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, len(order) * 0.38)))
    ax = axes[0]
    ax.barh(y, freq["total_mentions"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels_pretty)
    ax.invert_yaxis()
    ax.set_xlabel("Problem mentions, n")
    ax.set_title("A. Problem mentions by category")

    ax = axes[1]
    left = np.zeros(len(order))
    for sev_name in ["LOW", "MEDIUM", "HIGH"]:
        vals = sev_pct[sev_name].to_numpy()
        ax.barh(y, vals, left=left, label=sev_name.title())
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Severity composition within category (%)")
    ax.set_title("B. Severity composition by category")
    ax.legend()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"FigureS5_Problem_Distribution_Severity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    source = freq.set_index("problem_category").join(sev_pct).reset_index()
    source.to_csv(out / "FigureS5_source_data.csv", index=False)


def human_evaluation_figure(summary_file: Path, out: Path):
    if not summary_file.exists():
        return
    d = pd.read_csv(summary_file)
    d["se"] = d["sd"] / np.sqrt(d["n"])
    d["ci_half"] = [t.ppf(0.975, max(int(n) - 1, 1)) * se for n, se in zip(d["n"], d["se"])]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, max(7, 0.42 * len(d))))
    ax.errorbar(d["mean"], y, xerr=d["ci_half"], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["item"])
    ax.invert_yaxis()
    ax.set_xlim(1, 7)
    ax.set_xlabel("Mean rating (1=Strongly disagree, 7=Strongly agree)")
    ax.set_title("Human evaluation summary (mean ± 95% CI)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"FigureS4_Human_Evaluation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--automated-metrics", required=True)
    parser.add_argument("--recognition-summary", required=True)
    parser.add_argument("--problem-labels", required=True)
    parser.add_argument("--problem-recognition", required=True)
    parser.add_argument(
        "--human-summary",
        default="results/aggregate/human_evaluation_summary.csv",
    )
    parser.add_argument("--output-dir", default="outputs/manuscript")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    auto = pd.read_csv(args.automated_metrics)
    rec = pd.read_csv(args.recognition_summary)
    labels = pd.read_csv(args.problem_labels)
    problem_rec = pd.read_csv(args.problem_recognition)

    main_figure_2(auto, rec, labels, out)
    main_figure_3(problem_rec, out)
    supplementary_s1(auto, rec, out)
    supplementary_s5(labels, out)
    human_evaluation_figure(Path(args.human_summary), out)

    print(f"Saved manuscript outputs to {out}")


if __name__ == "__main__":
    main()
