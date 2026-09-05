"""Conservative accounting after independently retained metering verification."""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
import math
import hashlib
import json
from pathlib import Path


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid budget amount")
    return result


def reconciliation_upper(row, proof):
    reserved = money(row["reserved_usd"])
    app = proof["lifecycle"]
    times = [proof["interval_start"], proof["interval_end"], *proof["observed_at"],
             app["created_at"], app["stopped_at"]]
    if len(proof["observed_at"]) != 2 or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("invalid billing timestamps")
    if not proof["interval_start"] <= app["created_at"] <= app["stopped_at"]:
        raise ValueError("invalid application lifetime")
    if (app["app_id"] != row["app_id"] or app["job"] != row["id"] or
        app["state"] != "APP_STATE_STOPPED" or app["stopped_at"] <= 0):
        raise ValueError("billing reconciliation requires the matching stopped app")
    if (proof["interval_start"] > app["created_at"] or
        proof["interval_end"] < app["stopped_at"] + 3600 or
        proof["observed_at"][0] < proof["interval_end"] or
        proof["observed_at"][1] - proof["observed_at"][0] < 600):
        raise ValueError("billing interval or collection buffer is incomplete")
    first, second = proof["readings"]
    if not first or sorted(map(lambda x: json.dumps(x, sort_keys=True), first)) != sorted(
        map(lambda x: json.dumps(x, sort_keys=True), second)
    ):
        raise ValueError("metering changed between independent reads")
    if any(item["object_id"] != row["app_id"] for item in first):
        raise ValueError("metering contains another app")
    for item in first:
        stamp = datetime.fromisoformat(item["interval_start"])
        stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
        start = stamp.timestamp()
        if not proof["interval_start"] <= start or start + 3600 > proof["interval_end"]:
            raise ValueError("hourly metering row lies outside the collected interval")
    resources = {item["resource"] for item in first}
    if not {row["gpu"], "CPU", "Memory"} <= resources:
        raise ValueError("metering resource coverage is incomplete")
    keys = [(item["interval_start"], item["resource"]) for item in first]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate metering rows")
    measured = sum((money(item["cost"]) for item in first), Decimal(0))
    if measured <= 0 or measured > reserved:
        raise ValueError("recorded charges are zero or exceed the reserved bound")
    # Never release on elapsed time alone. Retain a 50% plus $0.25 buffer,
    # capped by the original full-job bound, and keep that original record.
    return min(reserved, measured * Decimal("1.5") + Decimal("0.25"))


def accounted(row, work):
    reserved = money(row["reserved_usd"])
    if row["status"] in ("running", "reserved") or not row.get("billing_reconciliation_sha256"):
        return reserved
    path = Path(work) / "results" / row["id"] / "billing-reconciliation.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["billing_reconciliation_sha256"]:
        raise ValueError("billing evidence hash mismatch")
    upper = reconciliation_upper(row, json.loads(data))
    if money(row["accounting_upper_usd"]) != upper:
        raise ValueError("accounted amount differs from verified metering")
    return upper
