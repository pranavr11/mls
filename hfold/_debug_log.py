from __future__ import annotations

import json
import os
import time
from typing import Any

_LOG_PATH = os.environ.get(
    "HFOLD_DEBUG_LOG_PATH",
    ".cursor/debug.log",
)
_SESSION_ID = os.environ.get("HFOLD_DEBUG_SESSION_ID")


def debug_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any], run_id: str = "run1") -> None:
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    if _SESSION_ID:
        payload["sessionId"] = _SESSION_ID
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
