"""Display identifiers and timestamps for the printed eye screening report."""
from __future__ import annotations

from base64 import b32encode
from datetime import datetime, timedelta, timezone
from hashlib import blake2s


BEIJING_TIME = timezone(timedelta(hours=8))


def format_report_metadata(created_utc: str | None, request_id: str | None) -> tuple[str, str]:
    """Convert the recorded time to Beijing time and derive a stable display number."""
    check_time = "时间待核对"
    report_date = ""
    if created_utc:
        timestamp = created_utc.strip()
        if timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"
        try:
            recorded = datetime.fromisoformat(timestamp)
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            local = recorded.astimezone(BEIJING_TIME)
        except (ValueError, OverflowError):
            pass
        else:
            check_time = local.strftime("%Y-%m-%d %H:%M:%S")
            report_date = local.strftime("%Y%m%d-")

    report_number = "编号待核对"
    if request_id:
        digest = blake2s(request_id.encode("utf-8"), digest_size=5).digest()
        suffix = b32encode(digest).decode("ascii")
        report_number = f"RET-{report_date}{suffix}"
    return check_time, report_number
