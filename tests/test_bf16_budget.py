import copy
import hashlib
import json
from decimal import Decimal

import pytest

from benchmarks.bf16_budget import accounted, reconciliation_upper


def fixture():
    row = {"id": "job", "app_id": "ap-test", "gpu": "H100", "status": "complete", "reserved_usd": 10}
    readings = [{"object_id": "ap-test", "resource": resource,
                 "interval_start": "1970-01-01T00:00:00", "cost": cost}
                for resource, cost in (("H100", "3"), ("CPU", ".5"), ("Memory", ".5"))]
    proof = {"lifecycle": {"job": "job", "app_id": "ap-test", "state": "APP_STATE_STOPPED",
                           "created_at": 1, "stopped_at": 100},
             "interval_start": 0, "interval_end": 4000, "observed_at": [5000, 6000],
             "readings": [readings, copy.deepcopy(readings)]}
    return row, proof


def test_verified_metering_keeps_margin_and_original_reservation(tmp_path):
    row, proof = fixture()
    data = json.dumps(proof).encode()
    path = tmp_path / "results/job/billing-reconciliation.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    row.update(billing_reconciliation_sha256=hashlib.sha256(data).hexdigest(), accounting_upper_usd="6.25")
    assert accounted(row, tmp_path) == Decimal("6.25")
    assert row["reserved_usd"] == 10
    row["status"] = "running"
    assert accounted(row, tmp_path) == 10
    row["status"] = "complete"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        accounted(row, tmp_path)


@pytest.mark.parametrize("fault", ["late", "unstopped", "new_bill", "foreign", "resource", "duplicate", "negative", "overflow", "early_read", "outside", "nan"])
def test_incomplete_or_inconsistent_billing_never_releases_a_reservation(fault):
    row, proof = fixture()
    if fault == "late": proof["lifecycle"]["stopped_at"] = 1000
    elif fault == "unstopped": proof["lifecycle"]["state"] = "APP_STATE_RUNNING"
    elif fault == "nan": proof["interval_end"] = float("nan")
    elif fault == "early_read": proof["observed_at"][1] = 5050
    else:
        for readings in proof["readings"]:
            if fault == "outside": readings[0]["interval_start"] = "1970-01-02T00:00:00"
            elif fault == "foreign": readings[0]["object_id"] = "another-app"
            elif fault == "resource": readings.pop()
            elif fault == "duplicate": readings.append(dict(readings[0]))
            elif fault == "negative": readings[0]["cost"] = "-1"
            elif fault == "overflow": readings[0]["cost"] = "11"
        if fault == "new_bill": proof["readings"][1][0]["cost"] = "3.01"
    with pytest.raises(ValueError): reconciliation_upper(row, proof)
