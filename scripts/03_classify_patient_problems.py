#!/usr/bin/env python
"""Classify patient utterances into problem categories and problem-specific severity levels.

This is the cleaned public version of the study's final problem-labeling pipeline.
It implements the 19-category taxonomy, GPT-5 classification prompt, batching,
cache behavior, JSON cleaning, and problem-instance outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
from docx import Document
from dotenv import load_dotenv


PROBLEM_CATEGORIES = {
    "PAIN_SIDE_EFFECTS": "Physical pain, burning, soreness, nausea, dizziness, numbness, tingling, swelling, rash",
    "FATIGUE_WEAKNESS": "Tiredness, exhaustion, worn out, weak, unable to function, kicked by mule",
    "TREATMENT_SCHEDULING": "Appointments, scheduling changes, session coordination, time slots, rescheduling",
    "MEDICATION_SYMPTOM_MGT": "Pills, medications, drugs, side effects, constipation, interactions, dosage",
    "FINANCIAL_STRESS": "Bills, copays, costs, price, insurance, Medicaid, payment, afford, money tight",
    "EMPLOYMENT_STRAIN": "Work, job, employment, cashier, warehouse, income",
    "FOOD_BASIC_NEEDS": "Food, nutrition, eating, taste changes, utilities, heat, electricity",
    "TRANSPORTATION": "Car, gas, distance, parking, rides, commute, American Cancer Society Road to Recovery",
    "ADMIN_PAPERWORK": "Forms, documents, portal, applications, signatures, fill out",
    "SYSTEM_FRAGMENTATION": "Multiple providers, system fragmentation, coordinating care",
    "DIGITAL_COMMUNICATION": "Portal messages, email, text, phone, technology, small print, app, patient portal",
    "HOUSING_INSTABILITY": "Housing, living situation, living alone, homeless, rent, utilities",
    "ENVIRONMENTAL_SAFETY": "Safety, neighborhood, crime",
    "LANGUAGE_LITERACY": "English, Arabic, jargon, acronyms, confusing, health literacy, understand",
    "INFO_OVERLOAD": "Overwhelmed, confused, too much, mixed up, can't keep track",
    "EMOTIONAL_DISTRESS": "Angry, furious, fed up, exhausted, anxious, depressed, hopeless, scared, rattled, frustrated",
    "SOCIAL_ISOLATION": "Alone, lonely, isolated, estranged, lack of support, family",
    "TRUST_COMMUNICATION": "Skeptical, distrust, brushed off, not heard, dismissed, rush, sarcasm",
    "DISCRIMINATION_STIGMA": "Discrimination, stigma, prejudice",
}


@dataclass
class UtteranceLabel:
    utterance_id: str
    conversation_id: str
    turn_number: int
    speaker: str
    text: str
    problems: List[str] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)
    severity: str = "UNSET"
    problem_instances: List[Dict] = field(default_factory=list)


def load_conversations_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    conversations = []
    for docx_file in sorted(Path(folder_path).glob("conv_*.docx")):
        doc = Document(docx_file)
        conversation_text = "\n".join(para.text for para in doc.paragraphs)
        conversations.append((docx_file.stem, conversation_text))
    return conversations


def parse_conversation(conversation_text: str, conversation_id: str) -> List[UtteranceLabel]:
    """Parse ASSISTANT:/A: and PATIENT: lines as sequential speaker turns."""
    utterances: List[UtteranceLabel] = []
    conversation_text = conversation_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = conversation_text.split("\n")
    current_speaker = None
    current_text: List[str] = []
    turn_number = 0

    def flush_current():
        nonlocal current_speaker, current_text, turn_number
        if current_speaker and current_text:
            text = "\n".join(current_text).strip()
            if text:
                utterances.append(
                    UtteranceLabel(
                        utterance_id=f"{conversation_id}_{current_speaker}_{turn_number}",
                        conversation_id=conversation_id,
                        turn_number=turn_number,
                        speaker=current_speaker,
                        text=text,
                    )
                )
                turn_number += 1
        current_text = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^PATIENT\s*:", line, flags=re.IGNORECASE):
            flush_current()
            current_speaker = "PATIENT"
            current_text = [re.sub(r"^PATIENT\s*:\s*", "", line, flags=re.IGNORECASE).strip()]
        elif re.match(r"^(A|ASSISTANT)\s*:", line, flags=re.IGNORECASE):
            flush_current()
            current_speaker = "ASSISTANT"
            current_text = [re.sub(r"^(A|ASSISTANT)\s*:\s*", "", line, flags=re.IGNORECASE).strip()]
        elif current_speaker:
            current_text.append(line)

    flush_current()
    return utterances


def get_cache_key(utterance_text: str) -> str:
    cache_version = "problem_specific_severity_v2"
    return hashlib.md5(f"{cache_version}::{utterance_text}".encode()).hexdigest()


def load_cached_label(utterance_text: str, cache_dir: Path) -> Optional[Dict]:
    cache_file = cache_dir / f"{get_cache_key(utterance_text)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def save_cached_label(utterance_text: str, label_result: Dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{get_cache_key(utterance_text)}.json").write_text(
        json.dumps(label_result, indent=2)
    )


def build_classification_prompt(utterances: List[str]) -> str:
    batch_text = "\n---\n".join(
        f"Utterance {i + 1}:\n{utt}" for i, utt in enumerate(utterances)
    )
    categories_str = "\n".join(
        f"- {cat}: {desc}" for cat, desc in PROBLEM_CATEGORIES.items()
    )
    return f"""You are a clinical NLP expert analyzing cancer patient-AI conversations.

For each patient utterance below, identify ALL applicable problem categories. For EACH problem category, assign a separate clinical severity label.

Available categories:
{categories_str}

Severity definitions:
- HIGH: Acute or urgent problem, unmanaged or severe symptom burden, active distress, or an immediate barrier likely to disrupt radiation treatment continuity without prompt support.
- MEDIUM: Meaningful but not immediately urgent problem, partially managed barrier, near-term risk to treatment continuity, or moderate distress requiring follow-up.
- LOW: Background concern, mild problem, informational need, or issue unlikely to disrupt treatment continuity immediately.

Important rules:
1. Severity must be assigned separately for each problem category, not once for the whole utterance.
2. If one utterance mentions both severe pain and parking, PAIN_SIDE_EFFECTS may be HIGH while TRANSPORTATION may be LOW or MEDIUM, depending on context.
3. Use only the category names listed above.
4. Do not invent categories.
5. If no patient problem is present, return an empty problem_instances list.

Utterances to classify:
{batch_text}

Respond ONLY with a valid JSON array, one object per utterance in the same order. Each object must have this structure:
{{
  "problem_instances": [
    {{"category": "CATEGORY_1", "severity": "LOW|MEDIUM|HIGH", "confidence": 0.95, "reasoning": "brief reason"}},
    {{"category": "CATEGORY_2", "severity": "LOW|MEDIUM|HIGH", "confidence": 0.88, "reasoning": "brief reason"}}
  ],
  "utterance_summary": "brief summary"
}}

Example:
[
  {{
    "problem_instances": [
      {{"category": "PAIN_SIDE_EFFECTS", "severity": "HIGH", "confidence": 0.98, "reasoning": "severe pain may require urgent clinical follow-up"}},
      {{"category": "TRANSPORTATION", "severity": "MEDIUM", "confidence": 0.90, "reasoning": "ride barrier may affect treatment attendance"}}
    ],
    "utterance_summary": "Patient reports severe pain and trouble getting to treatment."
  }}
]

Respond with ONLY the JSON array, no markdown or extra text."""


def classify_utterance_batch(
    utterances: List[str],
    client: openai.OpenAI,
    cache_dir: Path,
    model: str = "gpt-5",
) -> List[Dict]:
    results: List[Optional[Dict]] = [None] * len(utterances)
    uncached, uncached_idx = [], []

    for i, utt in enumerate(utterances):
        cached = load_cached_label(utt, cache_dir)
        if cached and "problem_instances" in cached:
            results[i] = cached
        else:
            uncached.append(utt)
            uncached_idx.append(i)

    if uncached:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": build_classification_prompt(uncached)}],
        )
        response_text = response.output_text
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        batch_results = json.loads(json_match.group()) if json_match else json.loads(response_text)

        valid_categories = set(PROBLEM_CATEGORIES)
        valid_severity = {"LOW", "MEDIUM", "HIGH"}
        cleaned_results = []

        for obj in batch_results:
            cleaned_instances, seen = [], set()
            for inst in obj.get("problem_instances", []) or []:
                cat = str(inst.get("category", "")).strip().upper()
                sev = str(inst.get("severity", "UNSET")).strip().upper()
                if cat not in valid_categories or cat in seen:
                    continue
                seen.add(cat)
                if sev not in valid_severity:
                    sev = "MEDIUM"
                try:
                    conf = float(inst.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                cleaned_instances.append(
                    {
                        "category": cat,
                        "severity": sev,
                        "confidence": conf,
                        "reasoning": str(inst.get("reasoning", "")),
                    }
                )
            cleaned_results.append(
                {
                    "problem_instances": cleaned_instances,
                    "utterance_summary": obj.get("utterance_summary", ""),
                }
            )

        for i, label_result in enumerate(cleaned_results):
            save_cached_label(uncached[i], label_result, cache_dir)
            results[uncached_idx[i]] = label_result

    return [x if x is not None else {"problem_instances": [], "utterance_summary": ""} for x in results]


def label_all_utterances(
    utterances: List[UtteranceLabel],
    client: openai.OpenAI,
    cache_dir: Path,
    model: str,
    batch_size: int,
) -> List[UtteranceLabel]:
    patient_utterances = [u for u in utterances if u.speaker == "PATIENT"]

    for batch_start in range(0, len(patient_utterances), batch_size):
        batch = patient_utterances[batch_start : batch_start + batch_size]
        labels = classify_utterance_batch(
            [u.text for u in batch], client, cache_dir, model=model
        )

        for u, label_result in zip(batch, labels):
            instances = label_result.get("problem_instances", []) or []
            u.problem_instances = instances
            u.problems = [x["category"] for x in instances]
            u.confidence = {x["category"]: float(x.get("confidence", 0.0)) for x in instances}
            severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNSET": 0}
            u.severity = (
                max(
                    [x.get("severity", "UNSET") for x in instances],
                    key=lambda s: severity_rank.get(s, 0),
                )
                if instances
                else "UNSET"
            )

    return utterances


def save_outputs(all_utterances: List[UtteranceLabel], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    utterance_file = output_dir / "utterance_labels.csv"
    problem_file = output_dir / "problem_instance_labels.csv"

    with utterance_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "conversation_id",
                "turn_number",
                "speaker",
                "text",
                "problems",
                "severity",
                "avg_confidence",
            ],
        )
        writer.writeheader()
        for utt in all_utterances:
            if utt.speaker == "PATIENT":
                avg_conf = np.mean(list(utt.confidence.values())) if utt.confidence else 0
                writer.writerow(
                    {
                        "utterance_id": utt.utterance_id,
                        "conversation_id": utt.conversation_id,
                        "turn_number": utt.turn_number,
                        "speaker": utt.speaker,
                        "text": utt.text[:200],
                        "problems": "|".join(utt.problems),
                        "severity": utt.severity,
                        "avg_confidence": f"{avg_conf:.2f}",
                    }
                )

    with problem_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "problem_instance_id",
            "utterance_id",
            "conversation_id",
            "turn_number",
            "speaker",
            "problem_category",
            "problem_severity",
            "problem_confidence",
            "problem_reasoning",
            "patient_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for utt in all_utterances:
            if utt.speaker != "PATIENT":
                continue
            for j, inst in enumerate(utt.problem_instances, start=1):
                writer.writerow(
                    {
                        "problem_instance_id": f"{utt.utterance_id}_P{j}",
                        "utterance_id": utt.utterance_id,
                        "conversation_id": utt.conversation_id,
                        "turn_number": utt.turn_number,
                        "speaker": utt.speaker,
                        "problem_category": inst.get("category", ""),
                        "problem_severity": inst.get("severity", ""),
                        "problem_confidence": inst.get("confidence", ""),
                        "problem_reasoning": inst.get("reasoning", ""),
                        "patient_text": utt.text,
                    }
                )

    print(f"Saved {utterance_file}")
    print(f"Saved {problem_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", default="outputs/conversations")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--cache-dir", default=".cache/problem_labels")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-conversations", type=int, default=300)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env file.")
    client = openai.OpenAI(api_key=api_key)

    conversations = load_conversations_from_folder(args.conversations)[: args.max_conversations]
    if not conversations:
        raise FileNotFoundError(f"No conv_*.docx files found in {args.conversations}")

    all_utterances: List[UtteranceLabel] = []
    for i, (conversation_id, conversation_text) in enumerate(conversations, start=1):
        print(f"[{i}/{len(conversations)}] {conversation_id}")
        utterances = parse_conversation(conversation_text, conversation_id)
        utterances = label_all_utterances(
            utterances,
            client,
            cache_dir=Path(args.cache_dir),
            model=args.model,
            batch_size=args.batch_size,
        )
        all_utterances.extend(utterances)

    save_outputs(all_utterances, Path(args.output_dir))


if __name__ == "__main__":
    main()
