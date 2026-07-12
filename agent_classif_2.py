"""
rpc_timeout_annotation_agent_fixed.py

Robust second-stage annotation agent for RPC server-side timeout heuristic mining.

Main reliability fixes over the original script:
  * Uses a JSON Schema, not only responseMimeType="application/json".
  * Detects and reports empty HTTP bodies and empty Gemini candidates.
  * Preserves finishReason, safetyRatings, promptFeedback, and usageMetadata in errors.
  * Reads HTTP error bodies so 400/429/5xx failures are understandable.
  * Retries transient empty/invalid responses with exponential backoff + jitter.
  * Writes failed rows to an errors JSONL file instead of silently losing them.
  * Tracks both included and excluded rows for correct --resume behavior.
  * Disables Gemini 2.5 Flash thinking for this classification task, leaving more
    output budget for the final JSON response.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

csv.field_size_limit(sys.maxsize)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)
SLEEP_BETWEEN_CALLS = 2.0
MAX_RETRIES = 4

OUTPUT_FIELDNAMES = [
    "issue_id", "rationale_authors", "jira_url", "project_name",
    "variable_edited", "category", "prev_value", "new_value", "time_cat",
    "reason", "github_commit_url", "keywords", "confidence", "needs_review",
]

ANNOTATION_KEYS = [
    "is_relevant", "variable_edited", "category", "prev_value", "new_value",
    "time_cat", "reason", "rationale_authors", "github_commit_url",
    "keywords", "confidence", "needs_review",
]

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_relevant": {"type": "boolean"},
        "variable_edited": {"type": "string"},
        "category": {"type": "string"},
        "prev_value": {"type": "string"},
        "new_value": {"type": "string"},
        "time_cat": {"type": "string"},
        "reason": {"type": "string"},
        "rationale_authors": {"type": "string"},
        "github_commit_url": {"type": "string"},
        "keywords": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", ""],
        },
        "needs_review": {"type": "boolean"},
    },
    "required": ANNOTATION_KEYS,
}

SYSTEM_PROMPT = """You are a research assistant helping mine Apache JIRA issues \
for a software-engineering study on how engineers decide RPC server-side \
timeout values (a mining-software-repositories / heuristic-extraction study).

You will be given one JIRA issue: its summary, description, discussion \
comments, and some pre-computed keyword-based signals from a noisy first-pass \
filter. That first-pass filter over-includes: many issues merely mention a \
timeout value somewhere in the discussion without describing any real timeout \
heuristic or decision.

Decide whether this issue is ACTUALLY relevant to RPC-server-side-timeout \
heuristic research: does it describe a concrete engineering decision about a \
timeout variable, its value, how it is computed, or the mechanism/policy that \
governs it, on the server side of an RPC/messaging path? Not purely \
client-side retry libraries, not JUnit/test-harness timeouts, not GC pauses \
that merely trigger a timeout as a side effect.

Qualifying examples: introducing or splitting a timeout for a specific \
operation type; changing how a timeout value is measured or computed; \
deciding to reuse or explicitly not reuse an existing timeout for a new \
purpose, with reasoning; changing a timeout's unit/precision; hardcoding vs. \
making a timeout configurable, with rationale; a specific numeric value change \
to a server-side timeout with an engineering justification.

Non-qualifying examples: a ticket that just happens to reference "rpc timeout" \
as a symptom without the fix touching a timeout variable or heuristic; \
unrelated features where timeout values appear only incidentally in test data, \
benchmarks, or logs; purely client-side concerns.

Return one JSON object that follows the supplied response schema. Do not add \
markdown, comments, or prose outside the JSON object.

When is_relevant is false, use empty strings for all string fields, set \
needs_review to false, and optionally put a brief exclusion explanation in \
reason. Never invent missing evidence.
"""


def build_user_prompt(row):
    return f"""ISSUE KEY: {row.get('key', '')}
JIRA URL: {row.get('url', '')}
PROJECT: {row.get('project_name', '')} ({row.get('project_key', '')})
ISSUE TYPE: {row.get('issue_type', '')} | STATUS: {row.get('status', '')} | RESOLUTION: {row.get('resolution', '')}
LABELS: {row.get('labels', '')} | COMPONENTS: {row.get('components', '')}

SUMMARY: {row.get('summary', '')}

DESCRIPTION:
{row.get('description', '')[:4000]}

FIRST-PASS SIGNALS (noisy, keyword-based -- do not trust blindly):
  matched_keywords: {row.get('matched_keywords', '')}
  timeout_values: {row.get('timeout_values', '')}
  inclusion_categories: {row.get('inclusion_categories', '')}
  exclusion_categories: {row.get('exclusion_categories', '')}
  selection_score: {row.get('selection_score', '')}
  selection_reason: {row.get('selection_reason', '')}
  has_rpc_timeout_evidence: {row.get('has_rpc_timeout_evidence', '')}
  has_value_change: {row.get('has_value_change', '')}
  has_configuration_reasoning: {row.get('has_configuration_reasoning', '')}
  has_real_fix: {row.get('has_real_fix', '')}

DISCUSSION COMMENTS:
{row.get('comments_text', '')[:8000]}
"""


def _short_json(value, limit=1400):
    """Compact diagnostics without flooding stderr or the errors file."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."


def _parse_json_object(raw_text):
    """Parse JSON, tolerating accidental Markdown fences or leading prose."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text:
        raise ValueError("Gemini candidate contained no text")

    try:
        value = json.loads(text)
    except json.JSONDecodeError as original_error:
        start = text.find("{")
        if start == -1:
            raise ValueError(
                f"Candidate was not JSON. First 500 chars: {text[:500]!r}"
            ) from original_error
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as recovery_error:
            raise ValueError(
                f"Invalid JSON candidate: {recovery_error}; "
                f"first 500 chars: {text[:500]!r}"
            ) from original_error

    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object, got {type(value).__name__}")
    return value


def _validate_annotation(annotation):
    """Validate essentials and normalize model output before use."""
    missing = [key for key in ANNOTATION_KEYS if key not in annotation]
    if missing:
        raise ValueError(f"Annotation is missing required keys: {missing}")

    if not isinstance(annotation["is_relevant"], bool):
        raise ValueError("is_relevant must be a boolean")
    if not isinstance(annotation["needs_review"], bool):
        raise ValueError("needs_review must be a boolean")

    for key in ANNOTATION_KEYS:
        if key not in {"is_relevant", "needs_review"} and not isinstance(annotation[key], str):
            raise ValueError(f"{key} must be a string")

    if annotation["confidence"] not in {"high", "medium", "low", ""}:
        raise ValueError(
            "confidence must be one of: high, medium, low, or empty string"
        )

    return annotation


def _build_request_body(row, model):
    generation_config = {
        "responseMimeType": "application/json",
        "responseJsonSchema": RESPONSE_JSON_SCHEMA,
        "temperature": 0.0,
        "maxOutputTokens": 2048,
    }

    if "gemini-2.5-flash" in model.lower():
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": build_user_prompt(row)}]}
        ],
        "generationConfig": generation_config,
    }


def call_gemini(row, api_key, model):
    url = API_URL_TEMPLATE.format(model=model)
    payload = json.dumps(_build_request_body(row, model)).encode("utf-8")
    last_err = None

    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                http_text = resp.read().decode("utf-8", errors="replace")

            if not http_text.strip():
                raise ValueError("Gemini returned HTTP 200 with an empty body")

            try:
                data = json.loads(http_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Gemini returned a non-JSON HTTP body: {http_text[:500]!r}"
                ) from exc

            candidates = data.get("candidates") or []
            if not candidates:
                raise ValueError(
                    "No candidates in Gemini response; "
                    f"promptFeedback={_short_json(data.get('promptFeedback', {}))}; "
                    f"usageMetadata={_short_json(data.get('usageMetadata', {}))}"
                )

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            parts = candidate.get("content", {}).get("parts") or []

            text_parts = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
                and not part.get("thought", False)
                and isinstance(part.get("text", ""), str)
            ]
            raw = "".join(text_parts).strip()

            if not raw:
                raise ValueError(
                    "Gemini returned an empty candidate; "
                    f"finishReason={finish_reason}; "
                    f"safetyRatings={_short_json(candidate.get('safetyRatings', []))}; "
                    f"promptFeedback={_short_json(data.get('promptFeedback', {}))}; "
                    f"usageMetadata={_short_json(data.get('usageMetadata', {}))}"
                )

            annotation = _parse_json_object(raw)
            return _validate_annotation(annotation)

        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(
                f"HTTP {exc.code} {exc.reason}: {error_body[:1400]}"
            )
            if exc.code == 429 or exc.code >= 500:
                if attempt < MAX_RETRIES - 1:
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    continue
            raise last_err from exc

        except (json.JSONDecodeError, ValueError, urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                continue

    raise RuntimeError(
        f"Failed to get a usable response for {row.get('key')}: {last_err}"
    ) from last_err


def annotate_row(row, api_key, model):
    """Classify and extract one CSV row. Returns None if not relevant."""
    annotation = call_gemini(row, api_key, model)
    if not annotation["is_relevant"]:
        return None

    return {
        "issue_id": row.get("key", ""),
        "rationale_authors": annotation["rationale_authors"],
        "jira_url": row.get("url", ""),
        "project_name": row.get("project_name", row.get("project_key", "")),
        "variable_edited": annotation["variable_edited"],
        "category": annotation["category"],
        "prev_value": annotation["prev_value"],
        "new_value": annotation["new_value"],
        "time_cat": annotation["time_cat"],
        "reason": annotation["reason"],
        "github_commit_url": annotation["github_commit_url"],
        "keywords": annotation["keywords"],
        "confidence": annotation["confidence"],
        "needs_review": annotation["needs_review"],
    }


def _load_processed_ids(processed_path, output_path):
    processed = set()

    if processed_path.exists():
        with processed_path.open(encoding="utf-8") as file:
            processed.update(line.strip() for line in file if line.strip())

    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                issue_id = row.get("issue_id", "").strip()
                if issue_id:
                    processed.add(issue_id)

    return processed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="First-stage filtered CSV")
    parser.add_argument("--output", required=True, help="CSV for qualifying rows")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successfully processed rows, including excluded rows",
    )
    parser.add_argument(
        "--errors",
        default=None,
        help="JSONL error log path (default: <output>.errors.jsonl)",
    )
    parser.add_argument(
        "--processed-file",
        default=None,
        help="Processed-ID checkpoint path (default: <output>.processed.txt)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in your environment before running this.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    processed_path = Path(
        args.processed_file or f"{args.output}.processed.txt"
    )

    with input_path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[:args.limit]

    already_done = set()
    if args.resume:
        already_done = _load_processed_ids(processed_path, output_path)
        print(
            f"Resuming: {len(already_done)} successfully processed issues "
            "will be skipped."
        )

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    kept = 0
    excluded = 0
    failed = 0
    processed_this_run = 0

    with (
        output_path.open("a", newline="", encoding="utf-8") as out_file,
    ):
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
        if write_header:
            writer.writeheader()
            out_file.flush()

        for index, row in enumerate(rows, start=1):
            key = row.get("key", f"row{index}")
            if key in already_done:
                continue

            processed_this_run += 1
            try:
                annotation = annotate_row(row, api_key, args.model)
            except Exception as exc:
                failed += 1
                print(
                    f"[{index}/{len(rows)}] {key}: ERROR ({exc}) -- not marked done",
                    file=sys.stderr,
                )
                
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            if annotation is None:
                excluded += 1
                print(
                    f"[{index}/{len(rows)}] {key}: excluded "
                    "(not relevant to RPC timeout heuristics)"
                )
            else:
                kept += 1
                print(
                    f"[{index}/{len(rows)}] {key}: INCLUDED -- "
                    f"{annotation['category']} ({annotation['confidence']})"
                )
                writer.writerow(annotation)
                out_file.flush()

            already_done.add(key)
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(
        f"\nDone. Included={kept}, excluded={excluded}, failed={failed}, "
        f"attempted={processed_this_run}."
    )
    print(f"Qualifying rows: {output_path}")


if __name__ == "__main__":
    main()
