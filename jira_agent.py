"""
jira_agent.py
=============
Agent that reads your JIRA CSV and auto-fills all columns using Gemini.

Usage:
    python jira_agent.py \
        --input jira_evidence_filtered.csv \
        --examples finding_csv.csv \
        --output jira_annotated.csv

Requirements:
    pip install google-genai pandas

You need a Gemini API key:
    export GEMINI_API_KEY=AIza...
"""

import os
import re
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
import time
from google.api_core import exceptions

MODEL = "gemma-4-31b-it"  

CATEGORIES = [
    "rpc",
    "heartbeat",
    "connection",
    "session",
    "coordination",
    "other",
    "not_rpc"
]

RPC_CATEGORIES=["read","write", "response"]

def build_system_prompt(examples_df) -> str:
    examples_text = ""
    if examples_df is not None and len(examples_df) > 0:
        examples_text = "\n\nHere are real examples of correctly annotated tickets:\n"
        for _, row in examples_df.iterrows():
            if not str(row.get("reason", "")).strip():
                continue
            examples_text += f"""
---
issue_id:   {row.get('issue_id', '')}
rationale_authors: {row.get('rationale_authors', '')}
variable_edited: {row.get('variable_edited', '')}
category:   {row.get('category', '')}
prev_value:{row.get('prev_value', '')}
new_value:{row.get('new_value', '')}
time_cat:   {row.get('time_cat', '')}
reason:     {row.get('reason', '')}
keywords:   {row.get('keywords', '')}
---"""

    return f"""You are a research assistant analyzing Apache JIRA tickets about RPC timeouts in distributed Java systems.

Your job is to extract structured information from each ticket and return ONLY a JSON object — no explanation, no markdown fences, no extra text.

────────────────────────────────────────────────
STEP 1 — DECIDE IF THIS TICKET IS IN SCOPE
────────────────────────────────────────────────
First, decide: is this ticket about a SERVER-TO-SERVER RPC timeout?

IN SCOPE (annotate these):
- RPC timeouts between cluster nodes
- IPC timeouts between server processes  
- Replica response timeouts (coordinator waiting for replicas)
- Server-to-server request/response timeouts

OUT OF SCOPE (set category = "not_rpc", leave all other fields empty):
- Client-to-server timeouts (SDK, driver, JDBC, CQL client)
- Gossip protocol timeouts
- Leader election timeouts
- ZooKeeper session expiration
- Keep-alive / idle connection timeouts
- UI or REST API timeouts
- Test infrastructure timeouts

If OUT OF SCOPE → return immediately with category = "not_rpc" and all other fields as empty strings.

────────────────────────────────────────────────
STEP 2 — EXTRACT FIELDS (only if IN SCOPE)
────────────────────────────────────────────────
- variable_edited
  The Java variable or config property name that was changed.
  Example: "rpc_timeout_in_ms", "ipc.client.rpc-timeout.ms"
  Empty string if not mentioned.

- rationale_authors
  Comma-separated names of comment authors who discussed the timeout value.
  Extract ONLY names inside square brackets at the start of comment blocks.
  Format in discussion: "[First Last]: comment text..."
  Rules:
    • Remove the brackets and colon.
    • Preserve order of first appearance, deduplicate.
    • Do NOT infer names from body text.
    • If no bracketed names exist → return "".
  Example output: "Andrew Kyle Purtell, Yi Liu"

- category
  The type of operation this timeout governs.
  Choose ONE from: {', '.join(CATEGORIES)}
  If the category is RPC, also specify the sub-category in this format:
  "rpc, <sub_category>" where sub_category is one of: {', '.join(RPC_CATEGORIES)}

- prev_value
  The old timeout value that was replaced. Return as a string (e.g. "10000").
  Must be in the same unit as new_value.
  Empty string if not found.

- new_value
  The new timeout value chosen. Return as a string (e.g. "30000").
  Must be in the same unit as prev_value.
  Empty string if not found.

- time_cat
  The time unit for prev_value and new_value.
  One of: milliseconds, seconds, minutes, hours, unknown

- reason
  2–4 sentences explaining WHY this value was chosen or changed.
  Quote the engineer directly when possible.
  Empty string if no reason is given anywhere in the ticket.

- github_commit_url
  A GitHub or git commit URL if one appears in the text.
  Empty string otherwise.

- keywords
  3–10 comma-separated words that identify this issue.
  Should be specific and useful for grouping similar tickets later.
  Example: "GC pause, internode, replica, increased, false positive"

- confidence
  "high"   → clearly an in-scope RPC timeout with a stated reason
  "medium" → relevant but the value or reason is ambiguous
  "low"    → you had to guess

- num_comments
  an integer to express the amount of comments are in the discussion. Exclude the ones from Hadoop QA or Github bots, or bots in general.s

- needs_review
  true  → borderline scope, missing value, or missing reason
  false → confident annotation
{examples_text}
────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────
Return ONLY valid JSON. Example for an in-scope ticket:
{{
  "variable_edited": "rpc_timeout_in_ms",
  "rationale_authors": "Masatake Adam, Eilfred Don",
  "category": "rpc, internode",
  "prev_value": "10000",
  "new_value": "30000",
  "time_cat": "milliseconds",
  "reason": "Increased from 10s to 30s because 10s was too short for clusters with large JVM heaps where GC pauses could exceed 8 seconds.",
  "github_commit_url": "",
  "keywords": "GC pause, stop-the-world, false positive, rpc, increased",
  "confidence": "high",
  "needs_review": false
}}

Example for an out-of-scope ticket:
{{
  "variable_edited": "",
  "rationale_authors": "",
  "category": "not_rpc",
  "prev_value": "",
  "new_value": "",
  "time_cat": "",
  "reason": "",
  "github_commit_url": "",
  "keywords": "",
  "confidence": "high",
  "needs_review": false
}}"""


def build_user_message(row: pd.Series) -> str:
    title       = str(row.get("title", ""))
    description = str(row.get("description", ""))[:800]
    discussion  = str(row.get("discussion", ""))[:3000]
    return f"""Ticket: {row.get('issue_id', row.get('ticket_id', ''))}
Title: {title}

Description:
{description}

Discussion:
{discussion}"""

class JiraAnnotationAgent:
    def __init__(self, examples_df, verbose=True):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client        = genai.Client(api_key=api_key, http_options={"timeout": 60000})
        self.system_prompt = build_system_prompt(examples_df)
        self.verbose       = verbose
        self.stats         = {"processed": 0, "high": 0, "medium": 0,
                              "low": 0, "not_rpc": 0, "errors": 0}

    def annotate(self, row: pd.Series) -> dict:
        try:
            response = self.client.models.generate_content(
                model=MODEL,
                contents=build_user_message(row),
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.0, 
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ],
                ),
            )

            if not response.text:
                raise Exception("API returned an empty response or the content was blocked.")
            
            raw = response.text.strip()
            raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw,
                         flags=re.MULTILINE).strip()

            result = json.loads(raw)

            self.stats["processed"] += 1
            conf = result.get("confidence", "low")
            self.stats[conf] = self.stats.get(conf, 0) + 1
            if result.get("category") == "not_rpc":
                self.stats["not_rpc"] += 1

            return result

        except json.JSONDecodeError as e:
            self.stats["errors"] += 1
            
            if self.verbose:
                print(f"    [JSON error] {row.get('issue_id','?')}: {e}")
                print(f"    Raw response: {raw[:200]}")
            return self._empty_result(str(e))

        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"    [API error] {row.get('issue_id','?')}: {e}")
            return self._empty_result(str(e))

    def _empty_result(self, error="") -> dict:
        return {
            "variable_edited": "", "rationale_authors":"","category": "not_rpc", "prev_value": "","new_value": "",
            "time_cat": "unknown", "reason": "", "github_commit_url": "",
            "keywords": "", "confidence": "low", "needs_review": True,
            "_error": error,
        }



def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate JIRA timeout tickets using Gemini"
    )
    parser.add_argument("--input",    required=True,
                        help="Input CSV (jira_evidence_filtered.csv)")
    parser.add_argument("--examples", default="findings_csv.csv",
                        help="Your manually annotated CSV (recommended)")
    parser.add_argument("--output",   default="jira_annotated.csv")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Only process first N rows (for testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip rows already in output file")
    parser.add_argument("--delay",    type=float, default=0.3,
                        help="Seconds between API calls (avoids rate limits)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)


    if "ticket_id" in df.columns and "issue_id" not in df.columns:
        df = df.rename(columns={"ticket_id": "issue_id"})
    print(f"Loaded {len(df)} tickets from {args.input}")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} rows")

    examples_df = None
    if args.examples and Path(args.examples).exists():
        examples_df = pd.read_csv(args.examples, sep=";")
        examples_df = examples_df[
            examples_df["reason"].notna() & (examples_df["reason"] != "")
        ]
        print(f"Loaded {len(examples_df)} manual examples for few-shot prompting")
        if len(examples_df) < 5:
            print("  ⚠️  Fewer than 5 examples — accuracy improves as you add more")
    else:
        print("No examples file — running zero-shot (lower accuracy)")

    already_done = set()
    if args.resume and Path(args.output).exists():
        done_df = pd.read_csv(args.output)
        already_done = set(done_df["issue_id"].astype(str))
        print(f"Resuming — skipping {len(already_done)} already processed tickets")

    try:
        agent   = JiraAnnotationAgent(examples_df=examples_df)
    except exceptions.ResourceExhausted as e:
            if i == retries - 1:
                raise e  # Out of retries, raise the error
            print(f"Rate limit hit. Waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay *= 2
    results = []
    total   = len(df)

    try:
        for i, (_, row) in enumerate(df.iterrows()):
            issue_id = str(row.get("issue_id", f"row_{i}"))

            if issue_id in already_done:
                continue

            print(f"  [{i+1}/{total}] {issue_id} — {str(row.get('title',''))[:60]}")

            annotation = agent.annotate(row)

            results.append({
                "issue_id":          issue_id,
                "rationale_authors":row.get("rationale_authors", ""),
                "jira_url":          row.get("jira_url", ""),
                "project_name":      row.get("project", row.get("project_name", "")),
                "variable_edited":          annotation.get("variable_edited", ""),
                "category":          annotation.get("category", ""),
                "prev_value":       annotation.get("prev_value", ""),
                "new_value":       annotation.get("new_value", ""),
                "time_cat":          annotation.get("time_cat", ""),
                "reason":            annotation.get("reason", ""),
                "github_commit_url": annotation.get("github_commit_url", ""),
                "keywords":          annotation.get("keywords", ""),
                "confidence":        annotation.get("confidence", ""),
                "needs_review":      annotation.get("needs_review", True),
            })

            if len(results) % 10 == 0 or i == total - 1:
                _save(results, args.output, args.resume)

            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\n🛑 [Proceso interrumpido por el usuario] Guardando progreso acumulado antes de salir...")
        if results:
            _save(results, args.output, args.resume)
            print(f"✓ Guardados {len(results)} tickets en {args.output}. Puedes reanudar luego con --resume")
        else:
            print("No se alcanzaron a procesar nuevos tickets en esta sesión.")
        return 

    _save(results, args.output, args.resume)

    out_df = pd.read_csv(args.output)
    print(f"\n✓ Done — {agent.stats['processed']} processed → {args.output}")
    print(f"  Confidence : high={agent.stats['high']}  "
          f"medium={agent.stats['medium']}  low={agent.stats['low']}")
    print(f"  Not RPC    : {agent.stats['not_rpc']}")
    print(f"  Errors     : {agent.stats['errors']}")
    print(f"  Needs review: {out_df['needs_review'].sum()}")

    rpc_df = out_df[out_df["category"] != "not_rpc"]
    if not rpc_df.empty:
        print(f"\n── Category breakdown ───────────────────")
        print(rpc_df["category"].value_counts().to_string())

        high = rpc_df[
            (rpc_df["confidence"] == "high") & (rpc_df["reason"] != "")
        ][["issue_id", "project_name", "category", "prev_value","new_value", "time_cat", "reason"]]
        if not high.empty:
            print(f"\n── High-confidence findings ─────────────")
            print(high.to_string(index=False))


def _save(results, output_path, resume):
    new_df = pd.DataFrame(results)
    if resume and Path(output_path).exists():
        existing = pd.read_csv(output_path)
        new_df = pd.concat([existing, new_df], ignore_index=True)\
                   .drop_duplicates(subset="issue_id")
    new_df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()