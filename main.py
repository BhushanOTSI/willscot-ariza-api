import os
import math
from datetime import datetime, timezone
from typing import Optional, Any

from arize import ArizeClient
from fastapi import FastAPI, HTTPException

SPACE_ID = os.getenv("ARIZE_SPACE_ID", "U3BhY2U6Mzg0MzQ6bWs0bA==")
API_KEY = os.getenv("ARIZE_API_KEY", "ak-7cf65008-8f13-454e-8254-c7916752ec56-yuWcPT5sUnnUuw5AZSxFPgUkqcq9I5P8")
DEFAULT_PROJECT_NAME = os.getenv("ARIZE_PROJECT_NAME", "CopilotStudio4")

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


@app.post("/spans/export")
def export_spans(
    project_name: str = DEFAULT_PROJECT_NAME,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    start = _ensure_utc(start_time) if start_time else datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = _ensure_utc(end_time) if end_time else datetime.now(timezone.utc)

    if start > end:
        raise HTTPException(status_code=400, detail="start_time must be earlier than end_time")

    if not project_name or not project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required")

    try:
        client = _get_client()
        spans_df = client.spans.export_to_df(
            space_id=SPACE_ID,
            project_name=project_name.strip(),
            start_time=start,
            end_time=end,
            columns=SELECTED_COLUMNS,
            # where="parent_id is null"
        )
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
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "data": spans_records,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
