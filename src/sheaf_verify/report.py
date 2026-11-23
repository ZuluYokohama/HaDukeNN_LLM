"""Markdown review report."""

from __future__ import annotations

from sheaf_verify.verify import VerifyReport


def render(report: VerifyReport) -> str:
    science = report.ok and all(row["status"] == "pass" for row in report.gate_rows)
    objections = [item for item in report.checklist if not item["answered"]]
    lines = [
        "# Review report",
        "",
        f"Integrity: {'pass' if report.ok else 'fail'}",
        f"Gates: {'pass' if science else 'fail'}",
        f"Open objections: {len(objections)}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Record |",
        "| --- | --- | --- |",
    ]
    for row in report.gate_rows:
        record = row["record_sha256"] or ""
        lines.append(f"| {row['gate_id']} {row['title']} | {row['status']} | `{record}` |")
    lines.extend(["", "## Checklist", "", "| Id | Answered | Detail |", "| --- | --- | --- |"])
    for item in report.checklist:
        lines.append(f"| {item['id']} | {'yes' if item['answered'] else 'no'} | {item['detail']} |")
    lines.extend(["", "## Notes", ""])
    if report.notes:
        lines.extend(f"- {note}" for note in report.notes)
    else:
        lines.append("None.")
    lines.extend(["", "## Problems", ""])
    if report.problems:
        lines.extend(f"- {problem}" for problem in report.problems)
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
