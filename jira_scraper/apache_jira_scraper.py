#!/usr/bin/env python3
"""
Scraper/minero de issues de Apache Jira.

Busca issues de un proyecto Apache mediante JQL, descarga sus campos
principales, comentarios, relaciones y changelog, y guarda los resultados
en JSON y CSV.

Ejemplos:
    python apache_jira_scraper.py --project CASSANDRA --text timeout
    python apache_jira_scraper.py --project KAFKA --text timeout --limit 100
    python apache_jira_scraper.py --jql 'project = CASSANDRA AND text ~ "read timeout"'
    python apache_jira_scraper.py --project CASSANDRA --text timeout --start 1 --end 100

Búsqueda automática de timeouts RPC:
    python apache_jira_scraper.py --rpc-timeouts CASSANDRA --limit-per-keyword 100
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://issues.apache.org/jira/rest/api/2"
DEFAULT_PAGE_SIZE = 50
REQUEST_TIMEOUT_SECONDS = 30

# Cada palabra genera una consulta JQL independiente. Acotado a términos que
# nombran directamente un timeout de RPC. Términos genéricos como "retry",
# "latency", "throughput", "coordinator", "messaging" o "rpc" solos se
# sacaron: matchean issues sobre cualquier tema (ej. throughput de
# compactación no tiene nada que ver con timeouts de RPC) y diluyen el
# dataset. Si hace falta ese contexto más amplio, es mejor analizarlo sobre
# el texto ya bajado (summary/description/comments) en vez de usarlo como
# criterio de búsqueda.
RPC_TIMEOUT_KEYWORDS = [
    "timeout",
    "rpc timeout",
    "read timeout",
    "write timeout",
    "request timeout",
    "response timeout",
    "socket timeout",
    "network timeout",
    "connection timeout",
    "deadline",
    "deadline exceeded",
    "remaining timeout",
    "ReadRPCTimeout",
    "WriteRPCTimeout",
    "RangeRPCTimeout",
]


# ---------------------------------------------------------------------------
# Criterios de selección para issues que aportan evidencia sobre POR QUÉ se
# configuró un timeout de cierta forma. No basta con mencionar "timeout":
# buscamos justificación, cambio de valor, trade-offs, pruebas o razonamiento.
# ---------------------------------------------------------------------------

INCLUSION_PATTERNS: dict[str, list[str]] = {
    "configuracion_valor": [
        r"\b(?:increase|decrease|raise|lower|reduce|change|set|configure|default)\w*\b.{0,80}\btimeout\b",
        r"\btimeout\b.{0,80}\b(?:increase|decrease|raise|lower|reduce|change|set|configure|default)\w*\b",
        r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec|seconds?|m|min|minutes?)\b",
    ],
    "justificacion": [
        r"\b(?:because|since|therefore|thus|so that|in order to|reason|rationale|why)\b",
        r"\b(?:too low|too high|too short|too long|aggressive|conservative)\b",
        r"\b(?:client expects?|expected by the client|from the client perspective)\b",
    ],
    "rendimiento_tradeoff": [
        r"\b(?:latency|throughput|performance|overhead|load|workload|queue|contention|resource)\b",
        r"\b(?:trade[- ]?off|impact|degrad|regression|slow(?:er)?|fast(?:er)?)\w*\b",
    ],
    "experimento_evidencia": [
        r"\b(?:benchmark|experiment|test result|stress test|load test|measurement|measured|observed|production)\b",
        r"\b(?:p50|p90|p95|p99|percentile|average latency|mean latency)\b",
    ],
    "mecanismo_timeout": [
        r"\b(?:deadline|remaining timeout|timeout budget|propagat\w* timeout|elapsed time)\b",
        r"\b(?:readrpctimeout|writerpctimeout|rangerpctimeout|rpc timeout|request timeout|response timeout)\b",
    ],
}

EXCLUSION_PATTERNS: dict[str, list[str]] = {
    "solo_pruebas_ci": [
        r"\b(?:jenkins|circleci|github actions|buildkite|maven|gradle)\b.{0,80}\btimeout\b",
        r"\b(?:unit test|integration test|test suite|flaky test|dtest)\b.{0,80}\btimeout\b",
    ],
    "solo_logs_o_error": [
        r"\b(?:stack trace|stacktrace|exception log|log message)\b",
    ],
    "ui_o_usuario": [
        r"\b(?:gui|user interface|browser|web ui|screen|user input)\b.{0,80}\btimeout\b",
    ],
    "proceso_local_no_rpc": [
        r"\b(?:local lock|mutex|semaphore|thread join|sleep timeout|process timeout)\b",
    ],
}

TIMEOUT_VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec|seconds?|m|min|minutes?)\b",
    re.IGNORECASE,
)

# Resoluciones que cuentan como "hay un fix real" que analizar. En los
# proyectos Apache relevantes (Cassandra, Kafka, HBase, Hadoop, Ratis, CXF),
# "Fixed" es la ampliamente dominante, pero "Done" e "Implemented" también
# representan cambios reales (a veces usados para features/mejoras en vez
# de bugs). El resto de las resoluciones (Won't Fix, Not A Problem, Cannot
# Reproduce, Duplicate, Invalid, etc.) no tienen un fix que estudiar.
REAL_FIX_RESOLUTIONS = {"Fixed", "Done", "Implemented"}


class JiraScraperError(RuntimeError):
    """Error controlado producido al consultar Jira."""


def nested_name(value: Any) -> str | None:
    """{"id": "5", "name": "Resolved"} -> "Resolved" """
    return value.get("name") if isinstance(value, dict) else None


def user_display_name(value: Any) -> str | None:
    """Nombre visible de un usuario Jira."""
    if not isinstance(value, dict):
        return None
    return value.get("displayName") or value.get("name") or value.get("key")


def list_names(values: Any) -> list[str]:
    """[{"name": "4.0"}, {"name": "4.1"}] -> ["4.0", "4.1"]"""
    if not isinstance(values, list):
        return []
    return [str(v["name"]) for v in values if isinstance(v, dict) and v.get("name")]


def extract_issue_links(issue_links: Any) -> list[dict[str, Any]]:
    """
    Normaliza las relaciones entre issues.

    outwardIssue: la relación sale desde la issue actual hacia otra.
    inwardIssue: la relación entra desde otra issue hacia la actual.
    """
    if not isinstance(issue_links, list):
        return []

    normalized: list[dict[str, Any]] = []
    for link in issue_links:
        if not isinstance(link, dict):
            continue

        link_type = link.get("type") or {}
        if "outwardIssue" in link:
            related, direction, relation = link["outwardIssue"], "outward", link_type.get("outward")
        elif "inwardIssue" in link:
            related, direction, relation = link["inwardIssue"], "inward", link_type.get("inward")
        else:
            continue

        related_fields = related.get("fields") or {}
        normalized.append({
            "direction": direction,
            "relation": relation,
            "key": related.get("key"),
            "summary": related_fields.get("summary"),
            "status": nested_name(related_fields.get("status")),
            "priority": nested_name(related_fields.get("priority")),
            "issue_type": nested_name(related_fields.get("issuetype")),
        })
    return normalized


def extract_comments(comment_field: Any) -> list[dict[str, Any]]:
    """Extrae los comentarios de una issue."""
    if not isinstance(comment_field, dict):
        return []
    return [
        {
            "id": c.get("id"),
            "author": user_display_name(c.get("author")),
            "created": c.get("created"),
            "updated": c.get("updated"),
            "body": c.get("body"),
        }
        for c in comment_field.get("comments", [])
        if isinstance(c, dict)
    ]


def extract_changelog(changelog: Any) -> list[dict[str, Any]]:
    """Extrae el historial de modificaciones (ej. status: Open -> Resolved)."""
    if not isinstance(changelog, dict):
        return []

    changes: list[dict[str, Any]] = []
    for history in changelog.get("histories", []):
        if not isinstance(history, dict):
            continue
        author = user_display_name(history.get("author"))
        created = history.get("created")
        for item in history.get("items", []):
            if not isinstance(item, dict):
                continue
            changes.append({
                "author": author,
                "created": created,
                "field": item.get("field"),
                "from": item.get("fromString"),
                "to": item.get("toString"),
            })
    return changes


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """Convierte la respuesta grande de Jira en un registro más simple."""
    fields = issue.get("fields") or {}
    project = fields.get("project") or {}
    comments = extract_comments(fields.get("comment"))
    issue_key = issue.get("key")

    return {
        "id": issue.get("id"),
        "key": issue_key,
        "url": f"https://issues.apache.org/jira/browse/{issue_key}",
        "project_key": project.get("key"),
        "project_name": project.get("name"),
        "summary": fields.get("summary"),
        "description": fields.get("description"),
        "issue_type": nested_name(fields.get("issuetype")),
        "status": nested_name(fields.get("status")),
        "resolution": nested_name(fields.get("resolution")),
        "priority": nested_name(fields.get("priority")),
        "reporter": user_display_name(fields.get("reporter")),
        "assignee": user_display_name(fields.get("assignee")),
        "creator": user_display_name(fields.get("creator")),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "resolution_date": fields.get("resolutiondate"),
        "labels": fields.get("labels") or [],
        "components": list_names(fields.get("components")),
        "fix_versions": list_names(fields.get("fixVersions")),
        "affected_versions": list_names(fields.get("versions")),
        "comments_count": len(comments),
        "comments": comments,
        "issue_links": extract_issue_links(fields.get("issuelinks")),
        "changelog": extract_changelog(issue.get("changelog")),
        # Se completa después de realizar las búsquedas por palabras clave.
        "matched_keywords": [],
        "attachments": [
            {
                "filename": a.get("filename"),
                "mime_type": a.get("mimeType"),
                "size": a.get("size"),
                "content_url": a.get("content"),
            }
            for a in fields.get("attachment", [])
            if isinstance(a, dict)
        ],
    }


def _issue_text(issue: dict[str, Any]) -> str:
    """Une los campos textuales que se usarán para aplicar los criterios."""
    comments = "\n".join(str(c.get("body") or "") for c in issue.get("comments", []))
    return "\n".join([
        str(issue.get("summary") or ""),
        str(issue.get("description") or ""),
        comments,
    ])


def evaluate_selection_criteria(issue: dict[str, Any]) -> dict[str, Any]:
    """
    Evalúa inclusión/exclusión de manera explicable.

    Regla principal:
    - Debe existir evidencia del mecanismo RPC/timeout.
    - Y además evidencia de al menos uno de estos aspectos:
      configuración de valor, justificación, rendimiento/trade-off o experimento.
    - Y la issue debe tener un fix real (resolution == "Fixed"): el estudio es
      sobre patrones de fix, así que una issue Open, Won't Fix, Not A Problem
      o Cannot Reproduce no aporta un fix que analizar, aunque el texto
      razone sobre el timeout.
    - Las exclusiones fuertes descartan issues cuyo contexto sea únicamente
      pruebas/CI, UI o procesos locales sin relación con RPC.
    """
    text = _issue_text(issue)
    lower_text = text.lower()

    inclusion_hits: dict[str, list[str]] = {}
    exclusion_hits: dict[str, list[str]] = {}

    for category, patterns in INCLUSION_PATTERNS.items():
        hits: list[str] = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                hits.append(pattern)
        if hits:
            inclusion_hits[category] = hits

    for category, patterns in EXCLUSION_PATTERNS.items():
        hits: list[str] = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                hits.append(pattern)
        if hits:
            exclusion_hits[category] = hits

    timeout_values = sorted(set(TIMEOUT_VALUE_PATTERN.findall(text)))

    # Indicadores semánticos principales.
    #
    # has_rpc_timeout solo mira si aparece una keyword de RPC_TIMEOUT_KEYWORDS
    # (sin contar "timeout" sola, para no matchear cualquier cosa). La
    # categoría "mecanismo_timeout" se movió a has_reasoning: describe CÓMO
    # funciona el timeout (deadline, timeout budget, etc.), no si el timeout
    # es de RPC o no, así que es más un tipo de razonamiento que evidencia
    # de "esto es RPC".
    has_rpc_timeout = any(k.lower() in lower_text for k in RPC_TIMEOUT_KEYWORDS if k.lower() != "timeout")

    # configuracion_valor es obligatorio: el estudio es sobre patrones de
    # fix, así que necesitamos evidencia de que se discutió/cambió un valor
    # o configuración de timeout, no solo que se justificó o se midió
    # rendimiento en el aire. Las otras categorías son contexto de apoyo
    # (alcanza con 1 de esas 4 para confirmar que hay razonamiento real).
    has_value_change = "configuracion_valor" in inclusion_hits
    has_reasoning = any(
        category in inclusion_hits
        for category in (
            "justificacion",
            "rendimiento_tradeoff",
            "experimento_evidencia",
            "mecanismo_timeout",
        )
    )

    # Puntuación transparente para ordenar candidatos.
    score = 0
    score += 3 if has_rpc_timeout else 0
    score += 3 if "configuracion_valor" in inclusion_hits else 0
    score += 3 if "justificacion" in inclusion_hits else 0
    score += 2 if "rendimiento_tradeoff" in inclusion_hits else 0
    score += 2 if "experimento_evidencia" in inclusion_hits else 0
    score += 1 if timeout_values else 0

    strong_exclusion = any(
        category in exclusion_hits
        for category in ("solo_pruebas_ci", "ui_o_usuario", "proceso_local_no_rpc")
    )

    # Cuenta como válida si resolution está en REAL_FIX_RESOLUTIONS (ya se
    # aplicó un fix) o si todavía no tiene resolution (issue.get("resolution")
    # is None): en Jira resolution solo se setea al resolver/cerrar, así que
    # "sin resolución" cubre cualquier estado activo (Open, Triage Needed,
    # In Progress, Patch Available, etc.) sin tener que enumerar cada nombre
    # de estado a mano -- y como el workflow de estados varía por proyecto
    # (CXF cierra como "Closed" en vez de "Resolved", por ejemplo), resolution
    # es el campo consistente entre todos. Won't Fix, Not A Problem, Cannot
    # Reproduce, Duplicate, Invalid, etc. quedan afuera: ahí se cerró sin que
    # haya un fix ni la posibilidad de que lo haya en el futuro.
    has_real_fix = issue.get("resolution") in REAL_FIX_RESOLUTIONS or issue.get("resolution") is None

    included = has_rpc_timeout and has_value_change and has_reasoning and has_real_fix and not strong_exclusion

    if included:
        decision_reason = (
            f"Incluida: contiene evidencia de timeout RPC, un cambio de configuración/valor, "
            f"razonamiento de apoyo, y tiene fix o todavía no se resolvió "
            f"(status={issue.get('status')!r}, resolution={issue.get('resolution')!r})."
        )
    elif strong_exclusion:
        decision_reason = "Excluida: el contexto coincide con un criterio fuerte de exclusión."
    elif not has_rpc_timeout:
        decision_reason = "Excluida: no hay evidencia suficiente de que sea un timeout RPC."
    elif not has_value_change:
        decision_reason = "Excluida: no se discute un cambio de valor/configuración del timeout."
    elif not has_reasoning:
        decision_reason = "Excluida: hay cambio de configuración, pero no explica por qué o con qué evidencia."
    else:
        decision_reason = (
            f"Excluida: se cerró sin un fix real "
            f"(status={issue.get('status')!r}, resolution={issue.get('resolution')!r})."
        )

    return {
        "selection_included": included,
        "selection_score": score,
        "selection_reason": decision_reason,
        "inclusion_categories": sorted(inclusion_hits),
        "exclusion_categories": sorted(exclusion_hits),
        "timeout_values": timeout_values,
        "has_rpc_timeout_evidence": has_rpc_timeout,
        "has_value_change": has_value_change,
        "has_configuration_reasoning": has_reasoning,
        "has_real_fix": has_real_fix,
    }


def apply_selection_criteria(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Añade la evaluación a cada issue y devuelve solo las incluidas."""
    filtered: list[dict[str, Any]] = []
    for issue in issues:
        evaluation = evaluate_selection_criteria(issue)
        issue.update(evaluation)
        if evaluation["selection_included"]:
            filtered.append(issue)
    return filtered


class ApacheJiraClient:
    """Cliente para consultar la API REST de Apache Jira."""

    def __init__(self, base_url: str = BASE_URL, delay_seconds: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.delay_seconds = max(delay_seconds, 0.0)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ApacheJiraResearchScraper/1.0",
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Realiza una petición GET a Jira."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise JiraScraperError(f"No se pudo conectar con Jira: {exc}") from exc

        # Jira puede responder 429 cuando se hacen demasiadas peticiones.
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            print(f"Jira limitó las peticiones. Esperando {retry_after} segundos...")
            time.sleep(retry_after)
            return self._get(path, params)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise JiraScraperError(f"Jira respondió HTTP {response.status_code}: {response.text[:500]}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise JiraScraperError("Jira no devolvió JSON válido.") from exc

        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return data

    def search_issue_keys(
        self, jql: str, start_at: int = 0, limit: int | None = None, page_size: int = DEFAULT_PAGE_SIZE
    ) -> list[str]:
        """
        Ejecuta una búsqueda JQL y obtiene las claves de las issues aplicando
        paginación.

        start_at permite arrancar en una posición específica del resultado
        (0 = la primera issue que devuelve la búsqueda), para poder bajar el
        listado en tandas manuales (ej. 1-100, después 101-200) sin repetir
        lo ya descargado. Ejemplo de clave: CASSANDRA-15442
        """
        keys: list[str] = []
        current_start = start_at
        page_size = min(max(page_size, 1), 100)  # Jira normalmente limita el tamaño de página.

        while True:
            remaining = None if limit is None else limit - len(keys)
            if remaining is not None and remaining <= 0:
                break
            current_size = page_size if remaining is None else min(page_size, remaining)

            data = self._get("search", params={
                "jql": jql,
                "startAt": current_start,
                "maxResults": current_size,
                "fields": "key",  # la búsqueda solamente necesita la clave de la issue.
            })

            issues = data.get("issues", [])
            if not issues:
                break

            keys.extend(issue["key"] for issue in issues if issue.get("key"))
            current_start += len(issues)
            total = int(data.get("total", len(keys)))

            print(
                f"\rIssues encontradas: {len(keys)} (posiciones {start_at + 1}-{current_start} de {total})",
                end="", flush=True,
            )

            if current_start >= total:
                break

        print()
        return keys

    def search_by_keywords(
        self, project: str, keywords: list[str], limit_per_keyword: int | None = None
    ) -> dict[str, list[str]]:
        """
        Ejecuta una consulta JQL independiente por cada palabra clave.

        Retorna {"CASSANDRA-15442": ["timeout", "ReadRPCTimeout", "deadline"]}
        para que una issue encontrada por varias consultas solamente se
        descargue una vez.
        """
        matches: dict[str, list[str]] = {}
        project = project.strip().upper()

        for index, keyword in enumerate(keywords, start=1):
            safe_keyword = keyword.replace("\\", "\\\\").replace('"', '\\"')
            jql = f'project = "{project}" AND text ~ "{safe_keyword}" ORDER BY created DESC'

            print(f"\n[{index}/{len(keywords)}] Buscando palabra clave: {keyword!r}")
            print(f"JQL: {jql}")

            for issue_key in self.search_issue_keys(jql=jql, limit=limit_per_keyword):
                matches.setdefault(issue_key, []).append(keyword)

        return matches

    def get_issue(self, issue_key: str) -> dict[str, Any]:
        """Descarga una issue completa. expand=changelog trae el historial."""
        fields = ",".join([
            "project", "summary", "description", "issuetype", "status", "resolution",
            "priority", "reporter", "assignee", "creator", "created", "updated",
            "resolutiondate", "labels", "components", "fixVersions", "versions",
            "comment", "issuelinks", "attachment",
        ])
        return self._get(f"issue/{issue_key}", params={"fields": fields, "expand": "changelog"})


def save_json(issues: list[dict[str, Any]], output_path: Path) -> None:
    """Guarda los resultados completos en JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(issues: list[dict[str, Any]], output_path: Path) -> None:
    """Guarda una versión tabular en CSV. Las listas se unen con ';'."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "key", "url", "project_key", "project_name", "summary", "description",
        "issue_type", "status", "resolution", "priority", "reporter", "assignee",
        "created", "updated", "resolution_date", "labels", "components",
        "fix_versions", "comments_count", "comments_text", "related_issue_keys",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for issue in issues:
            comments_text = "\n\n".join(
                f"[{c.get('author')} | {c.get('created')}]\n{c.get('body') or ''}"
                for c in issue.get("comments", [])
            )
            related_issue_keys = "; ".join(
                str(link.get("key")) for link in issue.get("issue_links", []) if link.get("key")
            )

            writer.writerow({
                "key": issue.get("key"),
                "url": issue.get("url"),
                "project_key": issue.get("project_key"),
                "project_name": issue.get("project_name"),
                "summary": issue.get("summary"),
                "description": issue.get("description"),
                "issue_type": issue.get("issue_type"),
                "status": issue.get("status"),
                "resolution": issue.get("resolution"),
                "priority": issue.get("priority"),
                "reporter": issue.get("reporter"),
                "assignee": issue.get("assignee"),
                "created": issue.get("created"),
                "updated": issue.get("updated"),
                "resolution_date": issue.get("resolution_date"),
                "labels": "; ".join(issue.get("labels", [])),
                "components": "; ".join(issue.get("components", [])),
                "fix_versions": "; ".join(issue.get("fix_versions", [])),
                "comments_count": issue.get("comments_count"),
                "comments_text": comments_text,
                "related_issue_keys": related_issue_keys,
            })


def build_jql(project: str, text: str | None) -> str:
    """
    Construye una consulta JQL sencilla.

    Se ordena por key ASC (CASSANDRA-1, CASSANDRA-2, ...) en vez de por
    fecha: la key de una issue nunca cambia, así que la posición 101-200 de
    hoy es la misma posición 101-200 de la semana que viene, aunque se hayan
    creado issues nuevas mientras tanto. Con "ORDER BY created DESC" las
    issues nuevas se cuelan al principio y corren todos los rangos ya
    bajados.
    """
    project = project.strip().upper()
    if not project:
        raise ValueError("El proyecto no puede estar vacío.")

    jql = f'project = "{project}"'
    if text:
        safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
        jql += f' AND text ~ "{safe_text}"'

    return jql + " ORDER BY key ASC"


def guess_project_folder(args: argparse.Namespace) -> str:
    """
    Determina en qué subcarpeta de --output-dir van los resultados, para que
    cada proyecto tenga la suya y no se mezclen archivos de proyectos
    distintos en el mismo directorio.
    """
    if args.rpc_timeouts:
        return args.rpc_timeouts.strip().upper()
    if args.project:
        return args.project.strip().upper()

    # Modo --jql: no hay un --project explícito, pero la mayoría de las
    # queries igual filtran por "project = X". Si lo encontramos, lo usamos
    # como nombre de carpeta; si no, todo va a una carpeta genérica.
    match = re.search(r'project\s*=\s*"?([A-Za-z0-9_]+)"?', args.jql or "", re.IGNORECASE)
    return match.group(1).upper() if match else "custom_query"


def parse_args() -> argparse.Namespace:
    """Define los argumentos disponibles desde la terminal."""
    parser = argparse.ArgumentParser(description="Descarga issues desde Apache Jira y las guarda en JSON y CSV.")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--project", help="Clave del proyecto, por ejemplo CASSANDRA, KAFKA o HADOOP.")
    query_group.add_argument("--jql", help='Consulta JQL completa. Ejemplo: project=CASSANDRA AND text~"timeout"')
    query_group.add_argument(
        "--rpc-timeouts", metavar="PROJECT",
        help="Busca automáticamente issues relacionadas con timeouts RPC usando varias palabras clave. "
             "Ejemplo: --rpc-timeouts CASSANDRA",
    )

    parser.add_argument("--text", help="Texto que debe aparecer en la issue. Solo se utiliza junto con --project.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Número máximo final de issues. Sin este argumento se descargan todas. "
             "No combinar con --start/--end (son formas alternativas de acotar la búsqueda).",
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Posición (1-indexado) de la primera issue a descargar dentro del resultado de la búsqueda. "
             "Usalo junto con --end para bajar en tandas manuales: primero --start 1 --end 100, después "
             "--start 101 --end 200, etc. Solo aplica a --project o --jql (no a --rpc-timeouts). Predeterminado: 1",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Posición (1-indexado, inclusive) de la última issue a descargar. Sin este argumento, descarga "
             "desde --start hasta el final del resultado.",
    )
    parser.add_argument(
        "--limit-per-keyword", type=int, default=None,
        help="Máximo de resultados por palabra clave cuando se utiliza --rpc-timeouts.",
    )
    parser.add_argument(
        "--filter-criteria", action="store_true",
        help="Aplica criterios de inclusión/exclusión y genera archivos *_filtered con solo issues relevantes.",
    )
    parser.add_argument("--output-dir", default="jira_output", help="Directorio de salida. Predeterminado: jira_output")
    parser.add_argument("--delay", type=float, default=0.2, help="Pausa entre peticiones en segundos. Predeterminado: 0.2")

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> str | None:
    """Devuelve un mensaje de error si los argumentos son inconsistentes, o None si están OK."""
    if args.limit is not None and args.limit <= 0:
        return "Error: --limit debe ser mayor que cero."
    if args.limit_per_keyword is not None and args.limit_per_keyword <= 0:
        return "Error: --limit-per-keyword debe ser mayor que cero."
    if args.start <= 0:
        return "Error: --start debe ser mayor que cero (es 1-indexado)."
    if args.end is not None and args.end < args.start:
        return "Error: --end debe ser mayor o igual que --start."
    if args.end is not None and args.limit is not None:
        return "Error: no combines --limit con --start/--end, son formas alternativas de acotar la búsqueda."
    if args.rpc_timeouts and (args.start != 1 or args.end is not None):
        return (
            "Error: --start/--end no está soportado junto con --rpc-timeouts (cada palabra clave arma su "
            "propio resultado, no hay una numeración única para acotar). Usá --limit-per-keyword y/o --limit "
            "en ese modo."
        )
    return None


def _build_query(args: argparse.Namespace) -> tuple[str | None, str]:
    """Arma el JQL (si aplica) y el prefijo de nombre de archivo según el modo elegido."""
    if args.rpc_timeouts:
        return None, f"{args.rpc_timeouts.upper()}_rpc_timeouts"

    if args.jql:
        jql, base_prefix = args.jql, "jira_custom_query"
    else:
        jql = build_jql(args.project, args.text)
        suffix = f"_{args.text}" if args.text else ""
        safe_suffix = "".join(c if c.isalnum() or c in "-_" else "_" for c in suffix)
        base_prefix = f"{args.project.upper()}{safe_suffix}"

    # Si se usó --start/--end, el rango va en el nombre del archivo para que
    # las tandas (1-100, 101-200, ...) no se pisen entre sí.
    if args.start != 1 or args.end is not None:
        range_suffix = f"_{args.start}-{args.end if args.end is not None else 'fin'}"
    else:
        range_suffix = ""

    return jql, f"{base_prefix}{range_suffix}"


def main() -> int:
    args = parse_args()

    error = _validate_args(args)
    if error:
        print(error, file=sys.stderr)
        return 2

    # start_at es 0-indexado (lo que espera la API de Jira); --start/--end
    # son 1-indexados de cara al usuario.
    start_at = args.start - 1
    range_limit = (args.end - args.start + 1) if args.end is not None else args.limit

    jql, file_prefix = _build_query(args)
    keyword_matches: dict[str, list[str]] = {}
    client = ApacheJiraClient(delay_seconds=args.delay)

    try:
        if args.rpc_timeouts:
            keyword_matches = client.search_by_keywords(
                project=args.rpc_timeouts, keywords=RPC_TIMEOUT_KEYWORDS, limit_per_keyword=args.limit_per_keyword,
            )
            # Las claves del diccionario son únicas, ya se eliminaron duplicados.
            issue_keys = list(keyword_matches)[: args.limit] if args.limit is not None else list(keyword_matches)
            print(f"\nIssues únicas después de eliminar duplicados: {len(issue_keys)}")
        else:
            print(f"JQL: {jql}")
            issue_keys = client.search_issue_keys(jql=jql, start_at=start_at, limit=range_limit)

        if not issue_keys:
            print("No se encontraron issues.")
            return 0

        output_dir = Path(args.output_dir) / guess_project_folder(args)
        json_path = output_dir / f"{file_prefix}.json"
        csv_path = output_dir / f"{file_prefix}.csv"
        filtered_json_path = output_dir / f"{file_prefix}_filtered.json"
        filtered_csv_path = output_dir / f"{file_prefix}_filtered.csv"
        normalized_issues: list[dict[str, Any]] = []

        def save_results() -> None:
            """
            Vuelca a disco lo descargado hasta el momento. Se llama al salir
            del loop por cualquier motivo (éxito, error o Ctrl+C), para no
            perder el trabajo ya hecho si el proceso se corta a mitad de tanda.
            """
            if args.filter_criteria:
                filtered = apply_selection_criteria(normalized_issues)
                save_json(normalized_issues, json_path)
                save_csv(normalized_issues, csv_path)
                save_json(filtered, filtered_json_path)
                save_csv(filtered, filtered_csv_path)
            else:
                save_json(normalized_issues, json_path)
                save_csv(normalized_issues, csv_path)

        try:
            for index, issue_key in enumerate(issue_keys, start=1):
                print(f"[{index}/{len(issue_keys)}] Descargando {issue_key}...")

                normalized = normalize_issue(client.get_issue(issue_key))
                normalized["matched_keywords"] = keyword_matches.get(issue_key, [])
                normalized_issues.append(normalized)
        finally:
            # Se ejecuta siempre: al terminar bien, al cortar con Ctrl+C, o
            # si get_issue/normalize_issue tira un error. Así el trabajo ya
            # bajado queda guardado pase lo que pase.
            save_results()

        print("\nProceso terminado.")
        print(f"Issues descargadas: {len(normalized_issues)}")
        print(f"JSON: {json_path.resolve()}")
        print(f"CSV:  {csv_path.resolve()}")
        if args.filter_criteria:
            filtered_count = sum(1 for issue in normalized_issues if issue.get("selection_included"))
            print(f"Issues incluidas por criterios: {filtered_count}")
            print(f"JSON filtrado: {filtered_json_path.resolve()}")
            print(f"CSV filtrado:  {filtered_csv_path.resolve()}")
        return 0

    except JiraScraperError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nProceso cancelado.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
