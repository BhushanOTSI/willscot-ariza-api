import os
import math
from datetime import datetime, timezone
from typing import Optional, Any

from arize import ArizeClient
from fastapi import FastAPI, HTTPException
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


SPACE_ID = os.getenv("ARIZE_SPACE_ID", "")
API_KEY = os.getenv("ARIZE_API_KEY", "")
DEFAULT_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "")

SELECTED_COLUMNS = [
    "context.trace_id",
    "context.span_id",
    "attributes.openinference.span.kind",
    "parent_id",
    "name",
    "start_time",
    "end_time",
    "attributes.llm.cost.completion",
    "attributes.llm.cost.total",
    "attributes.llm.token_count.completion",
    "attributes.llm.token_count.total",
    "attributes.llm.model_name",
    "attributes.metadata",
    "attributes.text",
    "attributes.llm.output_messages",
]

app = FastAPI(title="Arize Spans Export API")
EXCLUDED_PROJECTS = {"arize-demo-generative-llm-tracing", "arize-demo-llm-travel-agent"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_client() -> ArizeClient:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Missing ARIZE_API_KEY")
    return ArizeClient(api_key=API_KEY)


def _resolve_project_id(client: ArizeClient, project_name: str) -> str:
    """Resolve project name to project ID via projects.list."""
    response = client.projects.list(space_id=SPACE_ID)
    projects = list(response.projects)
    next_cursor = getattr(response, "next_cursor", None) or getattr(
        getattr(response, "page_info", None), "next_cursor", None
    )
    while next_cursor:
        response = client.projects.list(
            space_id=SPACE_ID, limit=100, cursor=next_cursor
        )
        projects.extend(response.projects)
        next_cursor = getattr(response, "next_cursor", None) or getattr(
            getattr(response, "page_info", None), "next_cursor", None
        )
    for p in projects:
        if p.name == project_name:
            return p.id
    raise HTTPException(
        status_code=404,
        detail=f"Project not found: {project_name!r}. Use GET /getprojects to list projects.",
    )


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    return value


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/getprojects")
def get_projects(limit: int = 100, cursor: Optional[str] = None) -> dict:
    if not SPACE_ID:
        raise HTTPException(status_code=500, detail="Missing ARIZE_SPACE_ID")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be greater than 0")

    try:
        client = _get_client()
        response = client.projects.list(
            space_id=SPACE_ID, limit=limit, cursor=cursor
        )
        projects = list(response.projects)
        next_cursor = getattr(response, "next_cursor", None)
        if not next_cursor and hasattr(response, "pagination"):
            next_cursor = getattr(response.pagination, "next_cursor", None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch projects: {exc}") from exc

    filtered_projects = [project for project in projects if project.name not in EXCLUDED_PROJECTS]

    out = {
        "space_id": SPACE_ID,
        "project_count": len(filtered_projects),
        "projects": [
            {"name": project.name, "id": project.id, "space_id": project.space_id}
            for project in filtered_projects
        ],
    }
    if next_cursor is not None and len(filtered_projects):
        out["next_cursor"] = next_cursor
    return out


@app.post("/spans/export")
def export_spans(
    project_name: str = DEFAULT_PROJECT_NAME,
    start_time_low: Optional[datetime] = None,
    start_time_high: Optional[datetime] = None,
    where: Optional[str] = None,
):
    start_low = _ensure_utc(start_time_low) if start_time_low else datetime(2020, 1, 1, tzinfo=timezone.utc)
    start_high = _ensure_utc(start_time_high) if start_time_high else datetime.now(timezone.utc)

    if start_low > start_high:
        raise HTTPException(status_code=400, detail="start_time_low must be earlier than start_time_high")

    if not project_name or not project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required")

    try:
        client = _get_client()
        where_value = where.strip() if where else ""
        sdk_where = None
    
        base_filter = where_value or None

        time_filter = (
            f"attributes.metadata.start_timestamp >= '{start_low.isoformat()}' "
            f"AND attributes.metadata.start_timestamp <= '{start_high.isoformat()}'"
        )
        sdk_where = time_filter if base_filter is None else f"({base_filter}) AND ({time_filter})"
        # print(sdk_where)

        export_kwargs = {
            "space_id": SPACE_ID,
            "project_name": project_name.strip(),
            # Use only low bound in export call; enforce full range locally below.
            "start_time": start_low,
            "end_time": datetime.now(timezone.utc),
            "columns": SELECTED_COLUMNS
        }
        if sdk_where:
            export_kwargs["where"] = sdk_where

        spans_df = client.spans.export_to_df(
            **export_kwargs
        )

        # Apply reliable time filtering locally.
        # if "start_time" in spans_df.columns:
        #     start_series = pd.to_datetime(spans_df["start_time"], errors="coerce", utc=True)
        #     start_bound = pd.Timestamp(start_low)
        #     end_bound = pd.Timestamp(start_high)
        #     spans_df = spans_df[start_series.between(start_bound, end_bound, inclusive="both")]

        spans_df = spans_df.reindex(columns=SELECTED_COLUMNS)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch spans: {exc}") from exc

    spans_records = spans_df.to_dict(orient="records")
    spans_records = _sanitize_json(spans_records)

    return {
        "project_name": project_name.strip(),
        "row_count": len(spans_records),
        "columns": SELECTED_COLUMNS,
        "start_time_low": start_low.isoformat(),
        "start_time_high": start_high.isoformat(),
        "where": sdk_where,
        "data": spans_records,
    }


@app.post("/spans/export_new")
def export_new(
    project_name: str = DEFAULT_PROJECT_NAME,
    start_time_low: Optional[datetime] = None,
    start_time_high: Optional[datetime] = None,
    filter: Optional[str] = None,
    limit_per_page: int = 100,
    cursor: Optional[str] = None,
):
    """Export spans using client.spans.list (paginated). Pass cursor from previous response for next page."""
    start_low = _ensure_utc(start_time_low) if start_time_low else datetime(2020, 1, 1, tzinfo=timezone.utc)
    start_high = _ensure_utc(start_time_high) if start_time_high else datetime.now(timezone.utc)

    if start_low > start_high:
        raise HTTPException(status_code=400, detail="start_time_low must be earlier than start_time_high")

    if not project_name or not project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required")

    if limit_per_page <= 0 or limit_per_page > 100:
        raise HTTPException(status_code=400, detail="limit_per_page must be between 1 and 100")

    try:
        client = _get_client()
        project_id = _resolve_project_id(client, project_name.strip())
        
        raw_filter = (filter or "").strip()
        base_filter = raw_filter or None

        time_filter = (
            f"attributes.metadata.start_timestamp >= '{start_low.isoformat()}' "
            f"AND attributes.metadata.start_timestamp <= '{start_high.isoformat()}'"
        )
        sdk_filter = time_filter if base_filter is None else f"({base_filter}) AND ({time_filter})"
        # print(sdk_filter)
        response = client.spans.list(
            project_id=project_id,
            # you can keep or drop these; they filter by span.start_time, not metadata:
            # start_time=start_low,
            # end_time=start_high,
            filter=sdk_filter,
            limit=limit_per_page,
            cursor=cursor,
        )
        # print(response)
        spans_df = response.to_df(json_normalize=True)
        if spans_df is None or spans_df.empty:
            spans_df = pd.DataFrame()
        else:
            existing = [c for c in SELECTED_COLUMNS if c in spans_df.columns]
            spans_df = spans_df.reindex(columns=existing)

        spans_records = spans_df.to_dict(orient="records")
        spans_records = _sanitize_json(spans_records)

        next_cursor = getattr(
            getattr(response, "pagination", None), "next_cursor", None
        )

        out = {
            "project_name": project_name.strip(),
            "project_id": project_id,
            "row_count": len(spans_records),
            "columns": SELECTED_COLUMNS,
            "start_time_low": start_low.isoformat(),
            "start_time_high": start_high.isoformat(),
            "filter": filter,
            "data": spans_records,
        }
        if next_cursor is not None and len(spans_records):
            out["next_cursor"] = next_cursor
        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch spans: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
