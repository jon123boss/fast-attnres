"""Format recorded BF16 results; statistical selection stays in bf16_report."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

STAGES = {"baseline": 80, "experiments": 220, "confirmation": 140, "reserve": 60}


def rows(data, key):
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def text(value):
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def number(value):
    return f"{value:.5g}" if type(value) in (int, float) and math.isfinite(value) else "unknown"


def interval(row):
    value = row.get("ci95_simultaneous", [])
    return f"[{number(value[0])}, {number(value[1])}]" if isinstance(value, (list, tuple)) and len(value) == 2 else "unknown"


def reserved(jobs):
    values = [row.get("reserved_usd") for row in jobs]
    return sum(values) if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values) else None


def accounting(jobs):
    try:
        values = [float(row.get("accounting_upper_usd", row["reserved_usd"]))
                  if row.get("status") not in ("running", "reserved") else float(row["reserved_usd"])
                  for row in jobs]
        return sum(values) if all(math.isfinite(v) and v >= 0 for v in values) else None
    except (ValueError, TypeError, KeyError):
        return None


def render(summary, ledger, *, summary_link="primary-summary.json", ledger_link="ledger.json"):
    summary = summary if isinstance(summary, dict) else {}
    ledger = ledger if isinstance(ledger, dict) else {}
    cells, missing = rows(summary, "cells"), rows(summary, "missing")
    failures, admissions = rows(summary, "failures"), rows(summary, "admission_failures")
    admitted = len(cells) if not admissions else "not established; see admission records"
    lines = ["# BF16 campaign report", "",
             f"[Primary summary](<{summary_link}>) · [Budget ledger](<{ledger_link}>)", "",
             f"Candidate: `{text(summary.get('candidate_identity'))}`. "
             f"Primary target passed: **{text(summary.get('primary_pass'))}**.", "",
             f"Recorded comparisons: {len(cells)}; admitted comparisons: {admitted}; "
             f"missing entries: {len(missing)}; failure records: {len(failures)}; "
             f"admission/inconclusive records: {len(admissions)}.", "",
             "Observed-cell geometric mean speedup: **"
             f"{number(summary.get('geometric_mean_speedup_observed_cells'))}×**. "
             "This copies the recorded statistic; incomplete or failed coverage does not establish the primary target.", "",
             "## Complete-step comparisons", "",
             "Ratios are candidate/alternative latency; lower is faster. All confidence intervals are simultaneous 95% intervals from the summary.", "",
             "| Configuration | Strongest correct alternative | Candidate ms | Alternative ms | Ratio | 95% CI | Non-regression gate |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in sorted(cells, key=lambda x: x.get("cell", "")):
        lines.append(f"| {text(row.get('cell'))} | {text(row.get('strongest_correct_alternative'))} | "
                     f"{number(row.get('candidate_ms'))} | {number(row.get('alternative_ms'))} | "
                     f"{number(row.get('ratio'))} | {interval(row)} | {text(row.get('nonregression_pass'))} |")
    if not cells:
        lines.append("| No completed comparisons | | | | | | unknown |")
    gates = rows(summary, "adjacent_ranks")
    failed = [row for row in gates if row.get("pass") is not True]
    lines += ["", "## Adjacent ranks", "",
              f"Recorded comparisons: {len(gates)}; failed or unestablished gates: {len(failed)}. "
              "Every interval, including passing comparisons, is retained in the primary summary.", ""]
    for row in sorted(failed, key=lambda x: x.get("comparison", "")):
        lines.append(f"- {text(row.get('comparison'))}: ratio {number(row.get('ratio'))}, "
                     f"95% CI {interval(row)}, passed={text(row.get('pass'))}.")
    for title, records in (("Missing entries", missing), ("Failure records", failures),
                           ("Admission and inconclusive records", admissions)):
        lines += ["", f"## {title}", ""]
        for record in records:
            if isinstance(record, dict):
                key = record.get("cell", record.get("path", "campaign"))
                detail = {k: record[k] for k in ("status", "reason", "reasons", "error") if k in record}
                if "arms" in record:
                    detail["arms"] = {k: v.get("classification", v.get("status")) for k, v in record["arms"].items()}
                record = f"{key}: {json.dumps(detail, sort_keys=True, default=str)}"
            lines.append(f"- {text(record)} ([details](<{summary_link}>))")
        if not records:
            lines.append("None recorded.")
    lines += ["", "## Budget reservations", "",
              "Original reservations remain recorded. The current accounting bound uses full reservations for unsettled jobs and verified metering plus a 50% and $0.25 cushion for reconciled stopped apps. These bounds are not actual bills.", "",
              "| Stage | Jobs | Original reserved USD | Current bound USD | Ceiling USD |", "| --- | ---: | ---: | ---: | ---: |"]
    jobs = rows(ledger, "jobs")
    for stage in [*STAGES, *sorted({row.get("stage", "unknown") for row in jobs} - STAGES.keys())]:
        group = [row for row in jobs if row.get("stage", "unknown") == stage]
        amount = reserved(group)
        lines.append(f"| {text(stage)} | {len(group)} | {number(amount)} | {number(accounting(group))} | {number(STAGES.get(stage))} |")
    lines += ["", f"Total reserved: **${number(reserved(jobs))}** "
              f"(historical reservations); current accounting bound: **${number(accounting(jobs))}** "
              f"of **${number(ledger.get('cap_usd'))}**.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.summary.resolve(), args.ledger.resolve()):
        raise ValueError("output must not overwrite input evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    links = [os.path.relpath(path.resolve(), args.output.parent.resolve()).replace(">", "%3E")
             for path in (args.summary, args.ledger)]
    result = render(json.loads(args.summary.read_text()), json.loads(args.ledger.read_text()),
                    summary_link=links[0], ledger_link=links[1])
    args.output.write_text(result)
    print(args.output)


if __name__ == "__main__":
    main()
