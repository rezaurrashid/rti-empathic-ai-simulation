#!/usr/bin/env python
"""Summarize the structured human evaluation survey.

The raw Google Forms/Sheets export is treated as restricted study data and is not
included in the public repository. This script converts that response-level CSV into
aggregate tables used for manuscript reporting without exporting free-text responses.

Agreement is defined as a Likert response of 6 or 7 on the 1-7 scale.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


CONVERSATION_COL = (
    "Which conversation number are you evaluating? ( Enter only the three-digit number. "
    "For example, if the file name is conv_010, type 010  )"
)
AGE_COL = "How old are you? "
GENDER_COL = "What is your gender?"
ROLE_COL = "What is your professional role?"
EXPERIENCE_COL = "How many years of experience do you have in your field?\n"

ITEMS = [
    ("AI Assistant Communication", "Maintained consistent tone", "The AI assistant maintained appropriate and consistent tone throughout the conversation."),
    ("AI Assistant Communication", "Recognized main problems", "The AI assistant recognized the patient’s main problems or concerns."),
    ("AI Assistant Communication", "Empathic and respectful tone", "The AI assistant used a tone that felt empathic and respectful."),
    ("AI Assistant Communication", "Clear and easy to understand", "The AI assistant’s responses were clear and easy to understand."),
    ("AI Assistant Communication", "Practical and useful", "The AI assistant’s responses were practical and useful for the situation."),
    ("Persona Realism", "Dialogue realistic", "The patient persona’s dialogue felt realistic and believable."),
    ("Persona Realism", "Natural emotional variation", "The patient showed natural emotional variation during the interaction."),
    ("Persona Realism", "Communication coherent", "The patient’s communication was coherent and easy to follow."),
    ("Persona Realism", "Concerns reflect oncology care", "The patient’s concerns reflected common issues in oncology care."),
    ("Persona Realism", "Useful for training", "The persona would be useful for training or communication research."),
    ("Interaction Quality", "Acknowledged emotions first", "The AI assistant acknowledged the patient’s emotions before giving information."),
    ("Interaction Quality", "Responded naturally", "The AI assistant and patient responded naturally to each other."),
    ("Interaction Quality", "Logical flow", "The conversation showed logical flow and resolution."),
    ("Clinical Appropriateness", "Clinically appropriate guidance", "The AI’s guidance was clinically appropriate for this type of case."),
    ("Clinical Appropriateness", "Mirrored navigator-patient interaction", "The conversation mirrored realistic navigator–patient interaction."),
    ("Clinical Appropriateness", "Enhances empathy training", "LLM-based roleplay could enhance communication or empathy training."),
    ("General Recommendation", "Recommend for research/education", "I would recommend using this type of simulation in research or  patient education. "),
]


def parse_conversation_id(value) -> float:
    match = re.search(r"(\d{1,3})", str(value))
    return float(match.group(1)) if match else np.nan


def summarize_item(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "mean": np.nan,
            "sd": np.nan,
            "agree_percent": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        critical = float(t.ppf(0.975, df=n - 1))
        half = critical * sd / np.sqrt(n)
    else:
        half = 0.0

    return {
        "n": n,
        "median": float(x.median()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
        "mean": mean,
        "sd": sd,
        "agree_percent": float((x >= 6).mean() * 100),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def normalize_role(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"chss", "community health support specialist"}:
        return "Community Health Support Specialist"
    if text == "researcher":
        return "Researcher"
    if text == "oncologist":
        return "Oncologist"
    if text == "clinician":
        return "Clinician"
    if text == "medical student":
        return "Medical Student"
    if text == "nursing student":
        return "Nursing Student"
    if "pre med" in text or "pre-med" in text:
        return "Pre-Med Student"
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw Google Forms/Sheets response CSV")
    parser.add_argument("--output-dir", default="results/aggregate")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {CONVERSATION_COL, AGE_COL, GENDER_COL, ROLE_COL, EXPERIENCE_COL}
    required.update(item[2] for item in ITEMS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input survey export is missing expected columns: {missing}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Item-level Likert summaries.
    rows = []
    for domain, short_name, column in ITEMS:
        row = {"domain": domain, "item": short_name}
        row.update(summarize_item(df[column]))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "human_evaluation_summary.csv", index=False, float_format="%.4f")

    # Response-level demographic summaries. These do not identify unique evaluators because
    # the survey export contains no evaluator identifier.
    demo_rows = []
    for variable, column in [
        ("Age group", AGE_COL),
        ("Gender", GENDER_COL),
        ("Years of experience", EXPERIENCE_COL),
    ]:
        counts = df[column].fillna("Missing").value_counts(dropna=False)
        for category, count in counts.items():
            demo_rows.append(
                {
                    "variable": variable,
                    "category": category,
                    "n": int(count),
                    "percent": float(count / len(df) * 100),
                }
            )

    roles = df[ROLE_COL].map(normalize_role)
    for category, count in roles.value_counts(dropna=False).items():
        demo_rows.append(
            {
                "variable": "Professional role (response-level)",
                "category": category,
                "n": int(count),
                "percent": float(count / len(df) * 100),
            }
        )
    pd.DataFrame(demo_rows).to_csv(
        out / "human_evaluation_demographics.csv", index=False, float_format="%.4f"
    )

    conversation_ids = df[CONVERSATION_COL].map(parse_conversation_id)
    counts = conversation_ids.dropna().astype(int).value_counts()
    duplicate_ids = sorted(counts[counts > 1].index.tolist())

    clinical_column = "The AI’s guidance was clinically appropriate for this type of case."
    recommendation_column = "I would recommend using this type of simulation in research or  patient education. "

    coverage = pd.DataFrame(
        [
            ("evaluation_forms", len(df)),
            ("unique_conversation_ids", int(conversation_ids.nunique())),
            ("conversation_ids_evaluated_more_than_once", len(duplicate_ids)),
            ("clinical_item_responses", int(pd.to_numeric(df[clinical_column], errors="coerce").notna().sum())),
            ("recommendation_item_responses", int(pd.to_numeric(df[recommendation_column], errors="coerce").notna().sum())),
        ],
        columns=["measure", "value"],
    )
    coverage.to_csv(out / "human_evaluation_coverage.csv", index=False)

    if duplicate_ids:
        pd.DataFrame({"conversation_id": duplicate_ids}).to_csv(
            out / "human_evaluation_duplicate_conversation_ids.csv", index=False
        )

    print(f"Read {len(df)} evaluation forms")
    print(f"Unique conversation IDs: {int(conversation_ids.nunique())}")
    print(f"Clinical-item responses: {coverage.loc[coverage.measure == 'clinical_item_responses', 'value'].iloc[0]}")
    print(f"Saved aggregate human-evaluation tables to {out}")


if __name__ == "__main__":
    main()
