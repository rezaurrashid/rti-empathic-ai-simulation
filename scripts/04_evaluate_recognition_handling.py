#!/usr/bin/env python
"""Evaluate whether the navigator recognized and handled each patient-raised problem.

This cleaned public script reproduces the study's final problem-specific recognition
and handling logic. Each patient turn is paired with the next assistant response.
GPT-5 evaluates recognition and actionability under a hierarchical rule:
handled ⊆ recognized and missed = not recognized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd
from docx import Document
from dotenv import load_dotenv


def load_conversation_from_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)


def parse_full_conversation(conversation_text: str) -> Dict[int, Dict[str, str]]:
    """Parse sequential speaker turns using the same indexing as problem labeling."""
    utterances: Dict[int, Dict[str, str]] = {}
    conversation_text = conversation_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = conversation_text.split("\n")

    current_speaker: Optional[str] = None
    current_text: List[str] = []
    turn_number = 0

    def flush_current() -> None:
        nonlocal turn_number, current_speaker, current_text
        if current_speaker and current_text:
            text = "\n".join(current_text).strip()
            if text:
                utterances[turn_number] = {current_speaker: text}
                turn_number += 1
        current_speaker = None
        current_text = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if current_speaker is None and line.startswith("#"):
            continue
        if re.match(r"^PATIENT\s*:", line, flags=re.IGNORECASE):
            flush_current()
            current_speaker = "PATIENT"
            current_text = [re.sub(r"^PATIENT\s*:\s*", "", line, flags=re.IGNORECASE).strip()]
        elif re.match(r"^(A|ASSISTANT)\s*:", line, flags=re.IGNORECASE):
            flush_current()
            current_speaker = "ASSISTANT"
            current_text = [re.sub(r"^(A|ASSISTANT)\s*:\s*", "", line, flags=re.IGNORECASE).strip()]
        elif current_speaker and not line.startswith("#"):
            current_text.append(line)

    flush_current()
    return utterances


def get_cache_key(patient_text: str, assistant_text: str, problems: List[str]) -> str:
    cache_version = "problem_specific_recognition_v2"
    problems_key = "|".join(str(p).strip().upper() for p in problems)
    combined = f"{cache_version}|||{patient_text}|||{assistant_text}|||{problems_key}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def load_cached_recognition(cache_key: str, cache_dir: Path) -> Optional[Dict]:
    path = cache_dir / f"{cache_key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_cached_recognition(cache_key: str, result: Dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{cache_key}.json").write_text(json.dumps(result, indent=2))


def extract_json_object(text: str) -> Dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def build_recognition_prompt(
    patient_text: str, patient_problems: List[str], assistant_text: str
) -> str:
    problems_str = "\n".join(f"- {p}" for p in patient_problems)
    return f"""You are evaluating a cancer patient-AI navigator conversation.

PATIENT STATEMENT:
"{patient_text}"

PROBLEMS IDENTIFIED IN PATIENT'S STATEMENT:
{problems_str}

ASSISTANT'S RESPONSE:
"{assistant_text}"

Evaluate each problem strictly according to these rules:

1. RECOGNIZED:
The assistant clearly acknowledges, names, paraphrases, or directly responds to the problem.

2. HANDLED:
The assistant recognizes the problem AND offers a specific action, suggestion, resource,
escalation step, or practical next step that addresses it.

3. MISSED:
The assistant does NOT recognize the problem at all.
A problem that is recognized but not handled must NOT be labeled as missed.

Constraints:
- "handled_problems" must be a subset of "recognized_problems".
- "missed_problems" must contain only problems that are NOT in "recognized_problems".
- Return problem names EXACTLY as provided above.
- Do not invent problem names.

Respond ONLY with valid JSON:
{{
  "recognized_problems": ["PROBLEM_1", "PROBLEM_2"],
  "handled_problems": ["PROBLEM_1"],
  "missed_problems": ["PROBLEM_3"],
  "notes": "Brief explanation"
}}"""


def evaluate_assistant_response(
    patient_text: str,
    patient_problems: List[str],
    assistant_text: str,
    client: openai.OpenAI,
    cache_dir: Path,
    model: str,
) -> Dict:
    patient_problems = [
        str(p).strip().upper() for p in patient_problems if str(p).strip()
    ]
    cache_key = get_cache_key(patient_text, assistant_text, patient_problems)
    cached = load_cached_recognition(cache_key, cache_dir)
    if cached:
        return cached

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": build_recognition_prompt(
                    patient_text, patient_problems, assistant_text
                ),
            }
        ],
    )
    raw = extract_json_object(response.output_text)

    all_problems = list(patient_problems)
    all_set = set(all_problems)
    raw_recognized = {
        str(x).strip().upper() for x in raw.get("recognized_problems", [])
    }
    raw_handled = {
        str(x).strip().upper() for x in raw.get("handled_problems", [])
    }

    recognized_set = all_set & raw_recognized
    handled_set = recognized_set & raw_handled
    missed_set = all_set - recognized_set

    recognized = [p for p in all_problems if p in recognized_set]
    handled = [p for p in all_problems if p in handled_set]
    missed = [p for p in all_problems if p in missed_set]
    total = len(all_problems)

    result = {
        "recognized_problems": recognized,
        "handled_problems": handled,
        "missed_problems": missed,
        "recognition_rate": round(len(recognized) / total, 3) if total else 0.0,
        "handling_rate": round(len(handled) / total, 3) if total else 0.0,
        "missed_rate": round(len(missed) / total, 3) if total else 0.0,
        "notes": raw.get("notes", ""),
    }
    save_cached_recognition(cache_key, result, cache_dir)
    return result


def calculate_problem_specific_recognition(
    problem_labels_df: pd.DataFrame,
    conversations_dir: str,
    client: openai.OpenAI,
    cache_dir: Path,
    model: str,
    max_conversations: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {
        "problem_instance_id",
        "utterance_id",
        "conversation_id",
        "turn_number",
        "problem_category",
        "problem_severity",
    }
    missing = required_cols - set(problem_labels_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = problem_labels_df.copy()
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["problem_category"] = df["problem_category"].astype(str).str.strip().str.upper()
    df["problem_severity"] = df["problem_severity"].astype(str).str.strip().str.upper()
    df["turn_number"] = df["turn_number"].astype(int)

    problem_rows: List[Dict] = []
    conversation_stats = defaultdict(
        lambda: {
            "total_problems": 0,
            "recognized": 0,
            "handled": 0,
            "missed": 0,
            "patient_turns": set(),
            "assistant_turns_with_response": 0,
        }
    )

    docx_files = sorted(Path(conversations_dir).glob("conv_*.docx"))[:max_conversations]

    for docx_file in docx_files:
        conv_id = docx_file.stem
        full_utterances = parse_full_conversation(
            load_conversation_from_docx(str(docx_file))
        )
        conv_problem_labels = df[df["conversation_id"] == conv_id].copy()
        if conv_problem_labels.empty:
            continue

        grouped = conv_problem_labels.groupby(["utterance_id", "turn_number"], sort=True)
        for (utterance_id, patient_turn), group in grouped:
            patient_turn = int(patient_turn)
            if patient_turn not in full_utterances or "PATIENT" not in full_utterances[patient_turn]:
                continue

            patient_text = full_utterances[patient_turn]["PATIENT"]
            next_assistant_turn = None
            next_assistant_text = None
            for turn_num in sorted(full_utterances):
                if turn_num > patient_turn and "ASSISTANT" in full_utterances[turn_num]:
                    next_assistant_turn = turn_num
                    next_assistant_text = full_utterances[turn_num]["ASSISTANT"]
                    break

            categories_unique: List[str] = []
            for cat in group["problem_category"].tolist():
                cat = str(cat).strip().upper()
                if cat and cat not in categories_unique:
                    categories_unique.append(cat)

            if next_assistant_text is None:
                eval_result = {
                    "recognized_problems": [],
                    "handled_problems": [],
                    "missed_problems": categories_unique,
                    "recognition_rate": 0.0,
                    "handling_rate": 0.0,
                    "missed_rate": 1.0,
                    "notes": "No assistant response found after patient turn.",
                }
            else:
                eval_result = evaluate_assistant_response(
                    patient_text,
                    categories_unique,
                    next_assistant_text,
                    client,
                    cache_dir,
                    model,
                )

            recognized_set = {
                str(x).strip().upper()
                for x in eval_result.get("recognized_problems", [])
            }
            handled_set = {
                str(x).strip().upper()
                for x in eval_result.get("handled_problems", [])
            }

            for _, prow in group.iterrows():
                cat = str(prow["problem_category"]).strip().upper()
                rec = int(cat in recognized_set)
                hand = int(cat in handled_set and rec == 1)
                miss = int(rec == 0)

                problem_rows.append(
                    {
                        "problem_instance_id": prow["problem_instance_id"],
                        "utterance_id": utterance_id,
                        "conversation_id": conv_id,
                        "patient_turn": patient_turn,
                        "assistant_turn": next_assistant_turn,
                        "problem_category": cat,
                        "problem_severity": str(prow["problem_severity"]).strip().upper(),
                        "problem_confidence": prow.get("problem_confidence", ""),
                        "recognized": rec,
                        "handled": hand,
                        "missed": miss,
                        "patient_text": patient_text,
                        "assistant_text": next_assistant_text or "",
                        "evaluator_notes": eval_result.get("notes", ""),
                    }
                )

                conversation_stats[conv_id]["total_problems"] += 1
                conversation_stats[conv_id]["recognized"] += rec
                conversation_stats[conv_id]["handled"] += hand
                conversation_stats[conv_id]["missed"] += miss

            conversation_stats[conv_id]["patient_turns"].add(patient_turn)
            if next_assistant_text is not None:
                conversation_stats[conv_id]["assistant_turns_with_response"] += 1

    conv_rows = []
    for conv_id, stats in sorted(conversation_stats.items()):
        total = stats["total_problems"]
        recognized = stats["recognized"]
        handled = stats["handled"]
        missed = stats["missed"]
        conv_rows.append(
            {
                "conversation_id": conv_id,
                "total_problems_raised": total,
                "total_patient_turns": len(stats["patient_turns"]),
                "total_assistant_turns": stats["assistant_turns_with_response"],
                "problems_recognized": recognized,
                "problems_handled": handled,
                "problems_missed": missed,
                "recognition_rate": round(recognized / total, 3) if total else 0.0,
                "handling_rate": round(handled / total, 3) if total else 0.0,
                "missed_issue_rate": round(missed / total, 3) if total else 0.0,
                "resolution_rate": round(handled / recognized, 3) if recognized else 0.0,
                "avg_problems_per_turn": round(
                    total / len(stats["patient_turns"]), 2
                )
                if stats["patient_turns"]
                else 0.0,
            }
        )

    return pd.DataFrame(conv_rows), pd.DataFrame(problem_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", default="outputs/conversations")
    parser.add_argument(
        "--problem-labels",
        default="outputs/evaluation/problem_instance_labels.csv",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--cache-dir", default=".cache/recognition")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-conversations", type=int, default=300)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env file.")
    client = openai.OpenAI(api_key=api_key)

    labels = pd.read_csv(args.problem_labels)
    conv_df, problem_df = calculate_problem_specific_recognition(
        labels,
        conversations_dir=args.conversations,
        client=client,
        cache_dir=Path(args.cache_dir),
        model=args.model,
        max_conversations=args.max_conversations,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conv_path = output_dir / "assistant_recognition_rates_problem_specific.csv"
    problem_path = output_dir / "problem_instance_recognition.csv"
    conv_df.to_csv(conv_path, index=False)
    problem_df.to_csv(problem_path, index=False)
    print(f"Saved {conv_path}")
    print(f"Saved {problem_path}")


if __name__ == "__main__":
    main()
