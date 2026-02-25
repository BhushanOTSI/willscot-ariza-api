import os
import math
from datetime import datetime, timezone
from typing import Optional, Any

from arize import ArizeClient
from fastapi import FastAPI, HTTPException
import pandas as pd

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


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_client() -> ArizeClient:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Missing ARIZE_API_KEY")
    return ArizeClient(api_key=API_KEY)


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
def get_projects(limit: int = 100) -> dict:
    if not SPACE_ID:
        raise HTTPException(status_code=500, detail="Missing ARIZE_SPACE_ID")
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be greater than 0")

    try:
        client = _get_client()
        response = client.projects.list(space_id=SPACE_ID, limit=limit)
        projects = list(response.projects)
        next_cursor = getattr(response, "next_cursor", None)
        if not next_cursor and hasattr(response, "page_info"):
            next_cursor = getattr(response.page_info, "next_cursor", None)

        while next_cursor:
            response = client.projects.list(
                space_id=SPACE_ID,
                limit=limit,
                cursor=next_cursor,
            )
            projects.extend(response.projects)
            next_cursor = getattr(response, "next_cursor", None)
            if not next_cursor and hasattr(response, "page_info"):
                next_cursor = getattr(response.page_info, "next_cursor", None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch projects: {exc}") from exc

    filtered_projects = [project for project in projects if project.name not in EXCLUDED_PROJECTS]

    return {
        "space_id": SPACE_ID,
        "project_count": len(filtered_projects),
        "projects": [
            {"name": project.name, "id": project.id, "space_id": project.space_id}
            for project in filtered_projects
        ],
    }


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
        if where_value:
            sdk_where = where_value

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
        if "start_time" in spans_df.columns:
            start_series = pd.to_datetime(spans_df["start_time"], errors="coerce", utc=True)
            start_bound = pd.Timestamp(start_low)
            end_bound = pd.Timestamp(start_high)
            spans_df = spans_df[start_series.between(start_bound, end_bound, inclusive="both")]

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
