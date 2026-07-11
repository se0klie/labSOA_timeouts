import requests
import pandas as pd
import json
import argparse
from pathlib import Path
import re

def clean_lines(texto):
    if not texto:
        return ""
    lineas_limpias = []
    
    for linea in texto.split('\n'):
        linea_strip = linea.strip()
        
        if any(bot in linea_strip.upper() for bot in ["[HADOOP QA]", "AUTOMATICALLY GENERATED", "HUDSON.APACHE.ORG", "JENKINS"]):
            continue
            
        if re.search(r'https?://\S+', linea_strip):
            continue
            
        if re.match(r'^[+-]\d?\s+\w+', linea_strip) or "warnings" in linea_strip.lower():
            continue
            
        linea_limpia = re.sub(r'^[*\s+\-\\/{}]+', '', linea_strip)
        
        if linea_limpia:
            lineas_limpias.append(linea_limpia)
            
    return " ".join(lineas_limpias) 

JIRA_PROJECT_KEYS = {
    "cassandra": "CASSANDRA",
    "hadoop":    "HADOOP",
    "zookeeper": "ZOOKEEPER",
    "kafka":     "KAFKA",
    "hbase":     "HBASE",
    "hive":      "HIVE",
    "spark":     "SPARK",
}

RATIONALE_KEYWORDS = [
    "because", "chosen", "we decided", "rationale", "reason",
    "GC pause", "p99", "p999", "too aggressive", "false positive",
    "network jitter", "too short", "too long", "increased to", "decreased to",
    "changed from", "bumped", "tuned",
]

RPC_KEYWORDS = [
    "rpc",
    "ipc",
    "request timeout",
    "response timeout",
    "read timeout",
    "write timeout",
    "coordinator",
    "internode",
]

HEARTBEAT_KEYWORDS = [
    "heartbeat",
    "gossip",
    "failure detector",
    "node down",
    "lost tracker",
]

CLIENT_KEYWORDS = [
    "client timeout", "driver", "cqlsh", "jdbc", "thrift client",
    "user-facing", "application timeout", "sdk", "rest api",
    "frontend", "browser",
]

NOISE_WORDS=[
    'sms','notification','pagerduty','email', 'cygwin','windows build','bash script'
]

def has_rationale(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in RATIONALE_KEYWORDS)


def fetch_all_issues(jql: str, fields: list, max_results: int) -> list:
    url = "https://issues.apache.org/jira/rest/api/2/search"
    headers = {"Content-Type": "application/json"}
    
    all_issues = []
    start_at = 0
    page_size = min(50, max_results)  

    while len(all_issues) < max_results:
        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": page_size,
            "fields": fields,
        }
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code != 200:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            break

        data = response.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)

        total = data.get("total", 0)
        print(f"  Fetched {len(all_issues)}/{min(total, max_results)}...")

        if len(issues) == 0 or len(all_issues) >= total:
            break
        start_at += len(issues)

    return all_issues


def fetch_timeouts_jira(project_names: list, max_results: int, output_path: str):
    jira_keys = []
    for name in project_names:
        key = JIRA_PROJECT_KEYS.get(name.lower())
        if key:
            jira_keys.append(key)
        else:
            print(f"  [!] Unknown project '{name}', skipping.")
    
    if not jira_keys:
        print("No valid projects. Exiting.")
        return

    search_terms = [
        "timeout", "heartbeat", "rpc", "ipc",
        "internode", 
        "replica", "gossip",
    ]
    comment_terms = ["increased", "decreased", "heuristic", "rpc", "timeout"]

    project_clause  = f"project IN ({', '.join(jira_keys)})"
    text_clause     = "(" + " OR ".join(f"text ~ '{t}'" for t in search_terms) + ")"
    comment_clause  = "(" + " OR ".join(
        f"comment ~ '{t}' OR description ~ '{t}'" for t in comment_terms
    ) + ")"
    change_clause   = "(text ~ 'increase' OR text ~ 'decrease' OR text ~ 'tune' OR text ~ 'change')"

    jql = f"{project_clause} AND {text_clause} AND {comment_clause} AND {change_clause}"
    print(f"\nJQL: {jql}\n")

    fields_to_fetch = [
        "summary", "description", "comment",
        "issuetype", "resolution", "created", "status",
    ]

    print("📡 Connecting to Apache JIRA API...")
    issues = fetch_all_issues(jql, fields_to_fetch, max_results)
    print(f"✅ Got {len(issues)} issues total.")

    rows = []
    for issue in issues:
        key    = issue["key"]
        f      = issue.get("fields", {})

        summary     = f.get("summary", "")
        description = f.get("description", "") or ""
        issuetype   = f.get("issuetype", {}).get("name", "")
        resolution  = f.get("resolution", {}).get("name", "") if f.get("resolution") else ""
        created     = f.get("created", "")[:10]  
        status      = f.get("status", {}).get("name", "")
        raw_comments = f.get("comment", {}).get("comments", [])
        comments_text = "\n---\n".join(
            f"[{c.get('author', {}).get('displayName', '?')}]: "
            f"{clean_lines(c.get('body', ''))}"
            for c in raw_comments
            if clean_lines(c.get('body', ''))
            and "hadoop qa" not in c.get("author", {}).get("displayName", "").lower()
        )

        full_text = description + " " + comments_text
        count = sum(
            1
            for c in raw_comments
            if "hadoop qa" not in c.get("author", {}).get("displayName", "").lower()
        )
        rows.append({
            "project":       key.split("-")[0],  
            "ticket_id":     key,
            "title":         summary,
            "issuetype":     issuetype,
            "status":        status,
            "resolution":    resolution,
            "created":       created,
            "description":   description[:1000], 
            "discussion":    comments_text,
            "has_rationale": has_rationale(full_text),
            "jira_url":      f"https://issues.apache.org/jira/browse/{key}",
            "amount_comments": count
        })

    df = pd.DataFrame(rows)
    
    return df


def matches_any(text: str, keywords: list) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in keywords)

def score_ticket(row) -> int:
    full_text = " ".join([
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("discussion", "")),
    ])
    score = 0
    for kw in RPC_KEYWORDS:
        if re.search(kw, full_text.lower()):
            score+=2
    for kw in HEARTBEAT_KEYWORDS:
        if re.search(kw, full_text.lower()):
            score += 1
    for kw in CLIENT_KEYWORDS:
        if re.search(kw, full_text.lower()):
            score -= 4 
    for kw in NOISE_WORDS:
        if re.search(kw, full_text.lower()):
            score -= 3   
    return score


def main():
    parser = argparse.ArgumentParser(description="Scrape timeout discussions from Apache JIRA")
    parser.add_argument("--repos",       nargs="+", default=["cassandra", "hadoop","hbase"])
    parser.add_argument("--max_results", type=int,  default=400)
    parser.add_argument("--output",      default="jira_evidence.csv")
    args = parser.parse_args()

    df_timeouts = fetch_timeouts_jira(args.repos, args.max_results, args.output)
    df_timeouts["relevance_score"] = df_timeouts.apply(score_ticket, axis=1)
    df_filtered = df_timeouts[~df_timeouts.apply(
        lambda r: matches_any(
            " ".join([str(r["title"]), str(r["description"])]),
            CLIENT_KEYWORDS
        ), axis=1)]
    df_filtered = df_filtered[df_filtered["relevance_score"] > 0]

    df_filtered = df_filtered.sort_values(
        ["has_rationale", "relevance_score"], ascending=False
    )
    df_filtered.to_csv("jira_evidence_filtered.csv", index=False)
    print(f"Before: {len(df_timeouts)}  →  After: {len(df_filtered)} tickets")
    print(f"With rationale: {df_filtered['has_rationale'].sum()}")



if __name__ == "__main__":
    main()