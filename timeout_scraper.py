"""
timeout_scraper.py
==================
Scrapes RPC timeout constants from Apache Java repos (server-side only).

Usage:
    python3 timeout_scraper.py --repos cassandra hadoop hbase zookeeper kafka
    python timeout_scraper.py --repos cassandra --no-clone  # if already cloned
"""

import re
import csv
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, fields, asdict
from collections import Counter

# ── Repos ────────────────────────────────────────────────────────────────────

REPOS = {
    "cassandra": "https://github.com/apache/cassandra.git",
    "hadoop":    "https://github.com/apache/hadoop.git",
    "zookeeper": "https://github.com/apache/zookeeper.git",
    "kafka":     "https://github.com/apache/kafka.git",
    "hbase":     "https://github.com/apache/hbase.git",
}

# ── Path filters ─────────────────────────────────────────────────────────────
# A file must match at least one PATH_INCLUDE token
# AND must not match any PATH_EXCLUDE token.

PATH_INCLUDE = [
    "server", "ipc", "/net/", "gms", "cluster",
    "rpc", "internode", "messaging", "transport", "remote",
]

PATH_EXCLUDE = [
    "/client/", "/driver/", "/test/", "/example/",
    "/benchmark/", "/public_api/", "/binding/", "/wrapper/",
]

# ── Line-level patterns ───────────────────────────────────────────────────────

TIMEOUT_LINE = re.compile(r'(?i)timeout')

SKIP_LINE = re.compile(
    r'^\s*('
    r'import\s'
    r'|package\s'
    r'|/\*'
    r'|\*'
    r'|//\s*@'
    r'|\s*\*\s*@param'
    r')'
)

VALUE_PATTERN = re.compile(
    r'(?<![.\w])'
    r'(\d[\d_]*)'
    r'(?:L|l)?'
    r'(?!\.\d)'
)

TIME_UNIT_PATTERN = re.compile(
    r'TimeUnit\.(MILLISECONDS|SECONDS|MINUTES|HOURS|NANOSECONDS)'
    r'|(millis|milliseconds|seconds|minutes|ms\b|sec\b)',
    re.IGNORECASE,
)

RPC_VAR_PATTERN = re.compile(r"""(?ix)
    rpc.*timeout        |
    ipc.*timeout        |
    internode.*timeout  |
    remote.*timeout     |
    call.*timeout       |
    request.*timeout    |
    response.*timeout   |
    server.*timeout     |
    replica.*timeout    |
    coordinator.*timeout
""")

RPC_METHOD_PATTERN = re.compile(
    r'(?i)(setTimeout|setRpcTimeout|setIpcTimeout|setRequestTimeout'
    r'|withTimeout|setCallTimeout|setResponseTimeout)',
)

CONFIG_PATTERN = re.compile(
    r'(?i)(getConf|getProperty|config\.get|conf\.get|properties\.get'
    r'|DatabaseDescriptor\.get|parameters\.get)',
)

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class TimeoutMatch:
    project:      str
    file:         str
    line_number:  int
    raw_value:    str
    unit:         str
    matched_line: str
    context:      str
    is_constant:  bool
    is_config:    bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def clone_or_update(name: str, url: str, base_dir: Path) -> Path:
    dest = base_dir / name
    if dest.exists():
        print(f"  [=] {name} already cloned — pulling latest...")
        subprocess.run(["git", "-C", str(dest), "pull", "--quiet"], check=False)
    else:
        print(f"  [+] Cloning {name} (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
            check=True,
        )
    return dest


def file_is_in_scope(filepath: Path) -> bool:
    """
    Returns True if the file should be scanned.
    Rules (in order):
      1. Hard-exclude client/test/driver paths.
      2. Must be in a server/RPC-related path OR have an RPC-related filename.
    """
    path_str = str(filepath).lower()

    # Rule 1 — hard exclusions first
    if any(excl in path_str for excl in PATH_EXCLUDE):
        return False

    # Rule 2 — must be server/RPC related
    if any(incl in path_str for incl in PATH_INCLUDE):
        return True

    # Fallback: filename itself suggests RPC/server component
    name = filepath.name.lower()
    rpc_names = ["rpc", "ipc", "server", "handler", "dispatcher",
                 "messaging", "transport", "connection", "channel"]
    return any(tok in name for tok in rpc_names)


def extract_unit(text: str) -> str:
    m = TIME_UNIT_PATTERN.search(text)
    if not m:
        return "unknown"
    matched = (m.group(1) or m.group(2)).upper()
    if matched in ("MS", "MILLIS", "MILLISECONDS"):
        return "MILLISECONDS"
    if matched in ("SEC", "SECONDS"):
        return "SECONDS"
    if matched == "MINUTES":
        return "MINUTES"
    if matched == "HOURS":
        return "HOURS"
    return matched


def extract_values(line: str) -> str:
    nums = VALUE_PATTERN.findall(line)
    filtered = []
    for n in nums:
        val = int(n.replace("_", ""))
        if 1 <= val <= 3_600_000:
            filtered.append(n)
    return ", ".join(filtered) if filtered else ""


def is_constant_decl(line: str) -> bool:
    return bool(re.search(
        r'\bstatic\b.*\bfinal\b|\bfinal\b.*\bstatic\b', line
    ))


# ── File scraper ──────────────────────────────────────────────────────────────

def scrape_file(filepath: Path, project: str) -> list[TimeoutMatch]:
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    matches = []
    for i, line in enumerate(lines):
        # Fast pre-filters
        if not TIMEOUT_LINE.search(line):
            continue
        if SKIP_LINE.match(line):
            continue
        # Skip getter calls unless it's a constant declaration
        # (Python syntax: use 'in', not .includes())
        if '.get' in line and not is_constant_decl(line):
            continue

        ctx_lines = lines[max(0, i - 3):i]
        ctx = " | ".join(l.strip() for l in ctx_lines if l.strip())

        raw_val    = extract_values(line)
        is_rpc_var = bool(RPC_VAR_PATTERN.search(line))
        has_method = bool(RPC_METHOD_PATTERN.search(line))
        has_config = bool(CONFIG_PATTERN.search(line))

        # Must have a value, a setter call, or a config read
        if not (raw_val or has_method or has_config):
            continue

        # If there's a value but no RPC method/config, require the variable
        # name (or context above) to match an RPC pattern
        if raw_val and not has_method and not has_config:
            if not RPC_VAR_PATTERN.search(line) and not RPC_VAR_PATTERN.search(ctx):
                continue

        rel_path = (
            str(filepath).split(f"/{project}/", 1)[-1]
            if f"/{project}/" in str(filepath)
            else str(filepath)
        )

        matches.append(TimeoutMatch(
            project=project,
            file=rel_path,
            line_number=i + 1,
            raw_value=raw_val,
            unit=extract_unit(line + " " + ctx),
            matched_line=line.strip()[:200],
            context=ctx[:300],
            is_constant=is_constant_decl(line),
            is_config=has_config,
        ))

    return matches


# ── Project scraper ───────────────────────────────────────────────────────────

def scrape_project(project_dir: Path, project_name: str) -> list[TimeoutMatch]:
    java_files = list(project_dir.rglob("*.java"))
    in_scope   = [f for f in java_files if file_is_in_scope(f)]
    skipped    = len(java_files) - len(in_scope)

    print(f"  {len(java_files)} .java files — "
          f"{len(in_scope)} in scope, {skipped} skipped")

    all_matches = []
    for f in in_scope:
        all_matches.extend(scrape_file(f, project_name))
    return all_matches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape RPC timeout constants from Apache Java repos"
    )
    parser.add_argument(
        "--repos", nargs="+", default=["cassandra", "hadoop"],
        help=f"Repos to process. Available: {list(REPOS.keys())}",
    )
    parser.add_argument("--output",   default="rpc_timeouts.csv")
    parser.add_argument("--clonedir", default="./repos")
    parser.add_argument(
        "--no-clone", action="store_true",
        help="Skip cloning — repos already present in --clonedir",
    )
    args = parser.parse_args()

    base_dir = Path(args.clonedir)
    base_dir.mkdir(exist_ok=True)

    all_matches: list[TimeoutMatch] = []

    for repo_name in args.repos:
        if repo_name not in REPOS:
            print(f"[!] Unknown repo '{repo_name}'. "
                  f"Available: {list(REPOS.keys())}")
            continue

        print(f"\n[{repo_name.upper()}]")
        project_dir = (
            base_dir / repo_name if args.no_clone
            else clone_or_update(repo_name, REPOS[repo_name], base_dir)
        )
        matches = scrape_project(project_dir, repo_name)
        print(f"  → {len(matches)} RPC timeout references found")
        all_matches.extend(matches)

    # Write CSV
    output    = Path(args.output)
    col_names = [f.name for f in fields(TimeoutMatch)]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        for m in all_matches:
            writer.writerow(asdict(m))

    print(f"\n✓ {len(all_matches)} rows written to {output}")

    # Summary
    print("\n── By project ───────────────────────────────────────────")
    proj_counter = Counter(m.project for m in all_matches)
    for proj, count in sorted(proj_counter.items()):
        constants = sum(1 for m in all_matches
                        if m.project == proj and m.is_constant)
        configs   = sum(1 for m in all_matches
                        if m.project == proj and m.is_config)
        print(f"  {proj:12s}  total={count:4d}  "
              f"constants={constants:3d}  config_refs={configs:3d}")

    print("\n── Constants (best candidates for git blame) ────────────")
    for m in all_matches:
        if m.is_constant:
            print(f"  {m.project:12s}  line {m.line_number:5d}  "
                  f"val={m.raw_value:10s}  {m.file}")


if __name__ == "__main__":
    main()