"""
Agentic report generation.

RoadRiskReportAgent is the "AI Report Generation" module referenced in the
project report. It follows the classic agent loop: observe the risk record,
reason about which factors actually drove the score, decide a priority and
timeline, then draft the report. Two drafting backends are supported behind
one interface (write_report):

  LocalLLMDrafter    calls a locally running Ollama model, if one is
                      reachable, to phrase the report in natural language.
  TemplateDrafter     a rule based fallback that runs with zero external
                      dependencies, using phrase pools so repeated reports
                      do not all read identically.

The agent tries the LLM backend first and falls back automatically, so the
dashboard keeps working on a machine with no local LLM installed at all.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT_SECONDS = 6

OPENING_PHRASES = [
    "Sensor readings for {road} in {ward}, {city} indicate a {tier_lower} risk condition.",
    "The latest inspection cycle for {road} ({ward}, {city}) places this segment in the {tier_lower} risk band.",
    "Analysis of {road} in {ward}, {city} shows a composite risk score of {score} out of 100, classified as {tier_lower}.",
]

URGENCY_PHRASES = {
    "Critical": [
        "This segment requires immediate attention before the next rainfall event.",
        "Deterioration at this rate poses a near term safety concern for vehicles and pedestrians.",
    ],
    "High": [
        "This segment should be scheduled for repair within the current maintenance cycle.",
        "Condition is deteriorating faster than the ward average and warrants prioritized inspection.",
    ],
    "Medium": [
        "Routine monitoring is recommended alongside the next scheduled maintenance round.",
        "No immediate action is required, but the segment should be re assessed after the next heavy rainfall.",
    ],
    "Low": [
        "The segment is currently within acceptable structural and drainage limits.",
        "No corrective action is needed at this time; continue standard monitoring.",
    ],
}

CLOSING_PHRASES = [
    "This assessment was generated automatically from live sensor data and should be verified by a field engineer before work orders are issued.",
    "Please treat this as a preliminary automated assessment; a physical inspection is advised to confirm the recommended action.",
]

PRIORITY_MAP = {"Critical": "P1 Immediate", "High": "P2 This Cycle", "Medium": "P3 Scheduled", "Low": "P4 Routine"}
TIMELINE_MAP = {
    "Critical": "within 72 hours",
    "High": "within 2 weeks",
    "Medium": "within the next maintenance cycle",
    "Low": "no fixed timeline, next routine review",
}


@dataclass
class GeneratedReport:
    subject: str
    body: str
    priority: str
    timeline: str
    backend_used: str


class TemplateDrafter:
    """Zero dependency fallback drafter using randomized phrase pools."""

    def draft(self, record: dict) -> str:
        opening = random.choice(OPENING_PHRASES).format(
            road=record["road_name"],
            ward=record["ward"],
            city=record["city"],
            tier_lower=record["risk_tier"].lower(),
            score=record["composite_risk_score"],
        )

        factors = record.get("top_factors", "")
        factor_line = ""
        if factors:
            factor_line = f"The primary contributing factors are: {factors.replace(';', ',')}."

        urgency = random.choice(URGENCY_PHRASES.get(record["risk_tier"], URGENCY_PHRASES["Medium"]))
        closing = random.choice(CLOSING_PHRASES)

        recommended_actions = self._recommend_actions(record)

        paragraphs = [opening, factor_line, urgency, recommended_actions, closing]
        return "\n\n".join(p for p in paragraphs if p)

    def _recommend_actions(self, record: dict) -> str:
        actions = []
        if record.get("drainage_quality_score", 100) < 40:
            actions.append("clear and re line stormwater drains along the segment")
        if record.get("road_age_years", 0) > 15:
            actions.append("schedule a structural resurfacing assessment")
        if record.get("rainfall_mm", 0) > 40:
            actions.append("deploy temporary water diversion barriers during peak rainfall")
        if record.get("vehicle_load_tons", 0) > 20:
            actions.append("review heavy vehicle load restrictions for this stretch")
        if not actions:
            actions.append("continue standard preventive maintenance")

        return "Recommended actions: " + "; ".join(actions) + "."


class LocalLLMDrafter:
    """Optional local LLM backend via Ollama, used only if reachable."""

    def is_available(self) -> bool:
        if requests is None:
            return False
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def draft(self, record: dict) -> Optional[str]:
        if requests is None:
            return None

        prompt = (
            "You are a municipal civil engineering assistant. Write a short, "
            "factual maintenance report (three to five sentences, no headings, "
            "no markdown, no bullet points) for a road segment based on this "
            f"data: road name {record['road_name']}, ward {record['ward']}, "
            f"city {record['city']}, composite risk score {record['composite_risk_score']} "
            f"out of 100, risk tier {record['risk_tier']}, contributing factors "
            f"{record.get('top_factors', 'not available')}, road age "
            f"{record.get('road_age_years', 'unknown')} years, recent rainfall "
            f"{record.get('rainfall_mm', 'unknown')} millimetres, drainage quality "
            f"score {record.get('drainage_quality_score', 'unknown')} out of 100. "
            "State the risk level, the likely cause, and one concrete recommended "
            "action. Do not use the dash or hyphen character anywhere in the reply."
        )

        try:
            response = requests.post(
                OLLAMA_ENDPOINT,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                return None
            text = response.json().get("response", "").strip()
            return text or None
        except Exception:
            return None


class RoadRiskReportAgent:
    """Coordinates factor reasoning, drafting backend choice, and priority."""

    def __init__(self, prefer_local_llm: bool = True):
        self.template_drafter = TemplateDrafter()
        self.llm_drafter = LocalLLMDrafter()
        self.prefer_local_llm = prefer_local_llm

    def generate_report(self, record: dict) -> GeneratedReport:
        priority = PRIORITY_MAP.get(record["risk_tier"], "P3 Scheduled")
        timeline = TIMELINE_MAP.get(record["risk_tier"], "within the next maintenance cycle")

        body = None
        backend_used = "template"

        if self.prefer_local_llm and self.llm_drafter.is_available():
            body = self.llm_drafter.draft(record)
            if body:
                backend_used = "local_llm"

        if not body:
            body = self.template_drafter.draft(record)
            backend_used = "template"

        subject = (
            f"Road Maintenance Alert: {record['road_name']}, {record['ward']}, "
            f"{record['city']} ({record['risk_tier']} Risk, Score {record['composite_risk_score']})"
        )

        footer = (
            f"\n\nPriority: {priority}\n"
            f"Recommended timeline: {timeline}\n"
            f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}"
        )

        return GeneratedReport(
            subject=subject,
            body=body + footer,
            priority=priority,
            timeline=timeline,
            backend_used=backend_used,
        )

    def generate_batch(self, records: pd.DataFrame, top_n: int = 10) -> List[GeneratedReport]:
        ranked = records.sort_values("composite_risk_score", ascending=False).head(top_n)
        return [self.generate_report(row.to_dict()) for _, row in ranked.iterrows()]
