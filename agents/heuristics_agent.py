"""Agent 2: extract grounded RPC timeout-setting heuristics from eligible JIRA issues."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ELIGIBLE = {"TIMEOUT_CHANGE", "TIMEOUT_STRATEGY"}
EXTRACTION_STATUSES = {"EXTRACTED", "INSUFFICIENT_EVIDENCE"}
IMPLEMENTATION_STATUSES = {
    "IMPLEMENTED", "PROPOSED", "DISCUSSED", "REJECTED", "MIXED", "UNKNOWN", "NOT_APPLICABLE"
}
CHANGE_TYPES = {
    "INCREASE", "DECREASE", "INTRODUCE", "REMOVE", "INFINITE_TO_FINITE", "SPLIT",
    "MAKE_CONFIGURABLE", "CHANGE_DEFAULT_OR_FALLBACK", "OTHER"
}
FACTOR_CATEGORIES = {
    "EXPECTED_DURATION", "OBSERVED_LATENCY", "PAYLOAD_OR_WORKLOAD_SIZE", "RECOVERY_DURATION",
    "REPLICATION_COST", "FAILOVER_REQUIREMENT", "FAILURE_DETECTION_SPEED", "RETRY_BEHAVIOR",
    "WAITING_TOO_LONG_COST", "TIMING_OUT_TOO_EARLY_COST", "OPERATION_TYPE_DIFFERENCE",
    "PROGRESS_OR_TASK_AWARENESS", "OTHER"
}
FACTOR_EFFECTS = {"INCREASE", "DECREASE", "DIFFERENTIATE", "DYNAMIC", "UNKNOWN"}
PROMPT_VERSION = "heuristic-extraction-v1"
OUTPUT_FIELDS = [
    "issue_ID", "eligibility_classification", "extraction_status", "rpc_context", "caller",
    "callee", "rpc_operation", "timeout_changes", "setting_factors", "rationale",
    "final_heuristic", "heuristic_evidence", "implementation_status", "evidence",
    "needs_manual_review", "uncertainties", "processing_status", "error_message",
    "model_name", "prompt_version",
]


def nullable_string() -> dict[str, Any]:
    return {"type": "string", "nullable": True}


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extraction_status": {"type": "string", "enum": sorted(EXTRACTION_STATUSES)},
        "rpc_context": {"type": "string"},
        "caller": nullable_string(),
        "callee": nullable_string(),
        "rpc_operation": nullable_string(),
        "timeout_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "change_type": {"type": "string", "enum": sorted(CHANGE_TYPES)},
                    "timeout_name": nullable_string(),
                    "old_timeout": nullable_string(),
                    "new_timeout": nullable_string(),
                    "implementation_status": {
                        "type": "string", "enum": sorted(IMPLEMENTATION_STATUSES)
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["change_type", "timeout_name", "old_timeout", "new_timeout",
                             "implementation_status", "evidence"],
                "additionalProperties": False,
            },
        },
        "setting_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": sorted(FACTOR_CATEGORIES)},
                    "description": {"type": "string"},
                    "effect_on_timeout": {"type": "string", "enum": sorted(FACTOR_EFFECTS)},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["category", "description", "effect_on_timeout", "evidence"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
        "final_heuristic": nullable_string(),
        "heuristic_evidence": {"type": "array", "items": {"type": "string"}},
        "implementation_status": {"type": "string", "enum": sorted(IMPLEMENTATION_STATUSES)},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "needs_manual_review": {"type": "boolean"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["extraction_status", "rpc_context", "caller", "callee", "rpc_operation",
                 "timeout_changes", "setting_factors", "rationale", "final_heuristic",
                 "heuristic_evidence", "implementation_status", "evidence",
                 "needs_manual_review", "uncertainties"],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = """You are Agent 2 in a research pipeline about heuristics for setting RPC timeouts in Java Apache server components. Agent 1 has already determined eligibility. Extract only information explicitly supported by the supplied original JIRA row and Agent 1 result. Return JSON only and conform exactly to the response schema.

Extract all relevant timeout configuration changes, including structural changes without numeric values. Preserve timeout values as written; do not convert units or infer missing values. Extract reusable technical factors that influence timeout selection. A factor description and final heuristic may paraphrase the source, but every claim must be grounded in short exact evidence excerpts.

Use null for an unstated caller, callee, operation, timeout name/value, or final heuristic. Do not invent RPC endpoints, values, causal relationships, rationale, or implementation status. A proposal is PROPOSED, not IMPLEMENTED. Issue status or resolution alone is not enough to claim that a specific timeout change was implemented unless the supplied text connects them. Use INSUFFICIENT_EVIDENCE and needs_manual_review=true if Agent 1 says the issue is eligible but the supplied source cannot support a reliable extraction.

The final_heuristic must be a concise, reusable timeout-setting rule derived from stated reasoning—not merely a restatement that a timeout changed. Leave it null when no reusable rule is supported. Evidence must be copied exactly from the supplied source, kept short, and must never include text that appears only in these instructions."""

SOURCE_FIELDS = [
    ("url", "URL"), ("project_key", "Project key"), ("project_name", "Project name"),
    ("summary", "Summary"), ("description", "Description"), ("issue_type", "Issue type"),
    ("status", "Status"), ("resolution", "Resolution"), ("priority", "Priority"),
    ("created", "Created"), ("updated", "Updated"), ("resolution_date", "Resolution date"),
    ("labels", "Labels"), ("components", "Components"), ("fix_versions", "Fix versions"),
    ("comments_text", "Comments"), ("related_issue_keys", "Related issues"),
    ("project", "Project"), ("title", "Title"), ("discussion", "Discussion"),
    ("comments", "Comments"),
]


def allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def issue_id_column(fieldnames: Iterable[str] | None) -> str:
    names = list(fieldnames or [])
    for candidate in ("issue_ID", "issue_id", "key"):
        if candidate in names:
            return candidate
    raise ValueError("CSV must contain issue_ID, issue_id, or key")


def truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    half = max_chars // 2
    return value[:half] + "\n[... middle omitted by extractor ...]\n" + value[-half:]


def collect_source(row: dict[str, str], issue_id: str, max_field_chars: int) -> str:
    parts = [f"Issue ID: {issue_id}"]
    for key, label in SOURCE_FIELDS:
        value = (row.get(key) or "").strip()
        if value:
            parts.append(f"{label}:\n{truncate_text(value, max_field_chars)}")
    return "\n\n".join(parts)


def clean_agent1_row(row: dict[str, str]) -> dict[str, Any]:
    keys = ["classification", "within_research_scope", "contains_timeout_change",
            "contains_timeout_strategy", "reason", "evidence", "needs_manual_review",
            "uncertainties", "processing_status"]
    cleaned: dict[str, Any] = {}
    for key in keys:
        value: Any = row.get(key, "")
        if key in {"evidence", "uncertainties"}:
            try:
                value = json.loads(value) if value else []
            except json.JSONDecodeError:
                value = []
        cleaned[key] = value
    return cleaned


def build_prompt(source_text: str, agent1_row: dict[str, str]) -> str:
    agent1 = clean_agent1_row(agent1_row)
    return ("Extract the timeout heuristic record from the following inputs.\n\n"
            "AGENT 1 RESULT:\n" + json.dumps(agent1, ensure_ascii=False, indent=2) +
            "\n\nORIGINAL JIRA SOURCE:\n" + source_text)


def evidence_strings(result: dict[str, Any]) -> list[str]:
    excerpts = list(result.get("evidence", [])) + list(result.get("heuristic_evidence", []))
    for change in result.get("timeout_changes", []):
        excerpts.extend(change.get("evidence", []))
    for factor in result.get("setting_factors", []):
        excerpts.extend(factor.get("evidence", []))
    return excerpts


def validate_result(result: dict[str, Any], source_text: str, eligibility: str) -> list[str]:
    errors: list[str] = []
    for key in RESPONSE_SCHEMA["required"]:
        if key not in result:
            errors.append(f"missing required field: {key}")
    if errors:
        return errors
    if set(result) != set(RESPONSE_SCHEMA["properties"]):
        errors.append("response contains missing or additional fields")
    if result["extraction_status"] not in EXTRACTION_STATUSES:
        errors.append("invalid extraction_status")
    for key in ("rpc_context", "rationale"):
        if not isinstance(result[key], str):
            errors.append(f"{key} must be a string")
    for key in ("caller", "callee", "rpc_operation", "final_heuristic"):
        if result[key] is not None and not isinstance(result[key], str):
            errors.append(f"{key} must be a string or null")
    if result["implementation_status"] not in IMPLEMENTATION_STATUSES:
        errors.append("invalid implementation_status")
    if type(result["needs_manual_review"]) is not bool:
        errors.append("needs_manual_review must be boolean")
    for key in ("timeout_changes", "setting_factors", "heuristic_evidence", "evidence", "uncertainties"):
        if not isinstance(result[key], list):
            errors.append(f"{key} must be a list")
    if errors:
        return errors
    for index, change in enumerate(result["timeout_changes"]):
        required = {"change_type", "timeout_name", "old_timeout", "new_timeout",
                    "implementation_status", "evidence"}
        if not isinstance(change, dict) or set(change) != required:
            errors.append(f"timeout_changes[{index}] has invalid fields")
            continue
        if change["change_type"] not in CHANGE_TYPES:
            errors.append(f"timeout_changes[{index}] has invalid change_type")
        if change["implementation_status"] not in IMPLEMENTATION_STATUSES:
            errors.append(f"timeout_changes[{index}] has invalid implementation_status")
        for key in ("timeout_name", "old_timeout", "new_timeout"):
            if change[key] is not None and not isinstance(change[key], str):
                errors.append(f"timeout_changes[{index}].{key} must be a string or null")
        if not isinstance(change["evidence"], list) or not change["evidence"]:
            errors.append(f"timeout_changes[{index}] requires evidence")
    for index, factor in enumerate(result["setting_factors"]):
        required = {"category", "description", "effect_on_timeout", "evidence"}
        if not isinstance(factor, dict) or set(factor) != required:
            errors.append(f"setting_factors[{index}] has invalid fields")
            continue
        if factor["category"] not in FACTOR_CATEGORIES:
            errors.append(f"setting_factors[{index}] has invalid category")
        if factor["effect_on_timeout"] not in FACTOR_EFFECTS:
            errors.append(f"setting_factors[{index}] has invalid effect_on_timeout")
        if not isinstance(factor["description"], str) or not isinstance(factor["evidence"], list) or not factor["evidence"]:
            errors.append(f"setting_factors[{index}] requires description and evidence")
    if result["extraction_status"] == "EXTRACTED" and not (result["timeout_changes"] or result["setting_factors"]):
        errors.append("EXTRACTED requires a timeout change or setting factor")
    if result["extraction_status"] == "INSUFFICIENT_EVIDENCE" and result["needs_manual_review"] is not True:
        errors.append("INSUFFICIENT_EVIDENCE requires needs_manual_review=true")
    if eligibility == "TIMEOUT_CHANGE" and result["extraction_status"] == "EXTRACTED" and not result["timeout_changes"]:
        errors.append("TIMEOUT_CHANGE eligibility requires an extracted timeout change")
    if eligibility == "TIMEOUT_STRATEGY" and result["extraction_status"] == "EXTRACTED" and not result["setting_factors"]:
        errors.append("TIMEOUT_STRATEGY eligibility requires an extracted setting factor")
    if result["final_heuristic"] is not None and not result["heuristic_evidence"]:
        errors.append("final_heuristic requires heuristic_evidence")
    if not all(isinstance(item, str) for item in result["uncertainties"]):
        errors.append("uncertainties must contain only strings")
    for excerpt in evidence_strings(result):
        if not isinstance(excerpt, str) or not excerpt.strip():
            errors.append("evidence excerpts must be non-empty strings")
        elif len(excerpt) > 500:
            errors.append("evidence excerpts must be at most 500 characters")
        elif "[... middle omitted by extractor ...]" in excerpt:
            errors.append("evidence cannot cite the truncation marker")
        elif excerpt not in source_text:
            errors.append(f"evidence is not an exact source excerpt: {excerpt[:80]!r}")
    return errors


def is_transient(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in
               ("429", "500", "502", "503", "504", "resource exhausted", "timeout", "temporar", "connection"))


def call_gemini_with_retry(client: Any, model: str, config: Any, prompt: str,
                           source_text: str, eligibility: str, max_attempts: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            parsed = getattr(response, "parsed", None)
            if parsed is None:
                parsed = json.loads(response.text)
            if not isinstance(parsed, dict):
                raise ValueError("Gemini response JSON is not an object")
            validation_errors = validate_result(parsed, source_text, eligibility)
            if validation_errors:
                raise ValueError("invalid Gemini response: " + "; ".join(validation_errors))
            return parsed
        except Exception as exc:
            last_error = exc
            if not is_transient(exc) or attempt == max_attempts - 1:
                break
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"Gemini request failed after {attempt + 1} attempt(s): {last_error}")


def load_agent1(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        id_column = issue_id_column(reader.fieldnames)
        for row in reader:
            issue_id = (row.get(id_column) or "").strip()
            if not issue_id:
                continue
            if row.get("processing_status") == "success" or issue_id not in rows:
                rows[issue_id] = row
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row.get("issue_ID", "") for row in csv.DictReader(handle) if row.get("issue_ID")}


def output_row(issue_id: str, eligibility: str, model: str, prompt_version: str,
               result: dict[str, Any] | None = None, error: Exception | None = None) -> dict[str, Any]:
    result = result or {}
    json_fields = {key: json.dumps(result.get(key, []), ensure_ascii=False)
                   for key in ("timeout_changes", "setting_factors", "heuristic_evidence", "evidence", "uncertainties")}
    return {
        "issue_ID": issue_id, "eligibility_classification": eligibility,
        "extraction_status": result.get("extraction_status", ""),
        "rpc_context": result.get("rpc_context", ""), "caller": result.get("caller", ""),
        "callee": result.get("callee", ""), "rpc_operation": result.get("rpc_operation", ""),
        **json_fields, "rationale": result.get("rationale", ""),
        "final_heuristic": result.get("final_heuristic", ""),
        "implementation_status": result.get("implementation_status", ""),
        "needs_manual_review": result.get("needs_manual_review", ""),
        "processing_status": "error" if error else "success", "error_message": str(error or ""),
        "model_name": model, "prompt_version": prompt_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original JIRA CSV")
    parser.add_argument("--classifications", required=True, type=Path, help="Agent 1 output CSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-field-chars", type=int, default=120_000,
                        help="Per-field prompt cap; 0 disables truncation")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        parser.error("Install the SDK: python3 -m pip install -U google-genai")
    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    allow_large_csv_fields()
    agent1 = load_agent1(args.classifications)
    existing = completed_ids(args.output)
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION, response_mime_type="application/json",
        response_json_schema=RESPONSE_SCHEMA, temperature=0,
    )
    with args.input.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        id_column = issue_id_column(reader.fieldnames)
        rows = list(reader)[args.start_index:]
    mode = "a" if args.output.exists() and args.output.stat().st_size else "w"
    processed = 0
    with args.output.open(mode, newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()
            target.flush()
        for row in rows:
            if args.max_rows is not None and processed >= args.max_rows:
                break
            issue_id = (row.get(id_column) or "").strip()
            classification_row = agent1.get(issue_id)
            if (not issue_id or issue_id in existing or not classification_row or
                    classification_row.get("processing_status") != "success" or
                    classification_row.get("classification") not in ELIGIBLE):
                continue
            eligibility = classification_row["classification"]
            source_text = collect_source(row, issue_id, args.max_field_chars)
            print(f"Extracting {issue_id} ({eligibility})", flush=True)
            try:
                prompt = build_prompt(source_text, classification_row)
                result = call_gemini_with_retry(
                    client, args.model, config, prompt, source_text, eligibility
                )
                writer.writerow(output_row(issue_id, eligibility, args.model, args.prompt_version,
                                           result=result))
            except Exception as exc:
                print(f"{issue_id}: {exc}", file=sys.stderr, flush=True)
                writer.writerow(output_row(issue_id, eligibility, args.model, args.prompt_version,
                                           error=exc))
            target.flush()
            existing.add(issue_id)
            processed += 1
    print(f"Processed {processed} eligible issue(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
