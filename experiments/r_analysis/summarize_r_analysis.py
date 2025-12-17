#!/usr/bin/env python3
"""
Summarize R-analysis scripts by executing them, capturing output, and writing
structured reports (JSON, CSV, Markdown).
"""

import json
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_SCRIPTS: Tuple[str, ...] = (
    "01_time_series_granger.py",
    "02_frequency_decomposition.py",
    "03_spatial_patterns.py",
    "04_phase_relationships.py",
    "05_dynamic_metastability.py",
    "06_multivariate_regression.py",
    "07_experimental_manipulation.py",
)


def run_scripts_and_summarize(
    scripts: Iterable[str] = DEFAULT_SCRIPTS,
    output_dir: Path | None = None,
    print_live: bool = True,
) -> List[dict]:
    """Run scripts, capture outputs, and write summary artifacts."""

    script_dir = Path(__file__).parent
    out_dir = Path(output_dir) if output_dir else script_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    records: List[dict] = []

    if print_live:
        print("=" * 80)
        print("RUNNING R ANALYSIS WITH SUMMARY")
        print("=" * 80)

    for idx, script in enumerate(scripts, 1):
        if print_live:
            print(f"\n{'#' * 80}")
            print(f"# [{idx}/{len(tuple(scripts))}] {script}")
            print(f"{'#' * 80}\n")

        result = subprocess.run(
            [python, str(script_dir / script)],
            capture_output=True,
            text=True,
        )

        if print_live:
            sys.stdout.write(result.stdout)
            sys.stdout.flush()
            if result.stderr:
                sys.stderr.write(result.stderr)
                sys.stderr.flush()

        records.append(
            {
                "script": script,
                "returncode": result.returncode,
                "status": "ok" if result.returncode == 0 else "fail",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

    timestamp = datetime.now().isoformat(timespec="seconds")

    # JSON report
    json_path = out_dir / "r_analysis_runs.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "records": records}, f, indent=2)

    # CSV summary
    csv_path = out_dir / "r_analysis_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["script", "status", "returncode"])
        for rec in records:
            writer.writerow([rec["script"], rec["status"], rec["returncode"]])

    # Markdown summary (truncate long outputs)
    md_path = out_dir / "r_analysis_summary.md"
    lines = [
        "# R Analysis Run Summary",
        "",
        f"Run at {timestamp}",
        "",
    ]

    max_chars = 2000
    for rec in records:
        lines.append(f"## {rec['script']}")
        lines.append(f"- Status: {rec['status']} (code {rec['returncode']})")
        stdout_trim = (rec["stdout"] or "").strip()
        if len(stdout_trim) > max_chars:
            stdout_trim = stdout_trim[:max_chars] + "\n... [truncated]"
        if stdout_trim:
            lines.append("\n```")
            lines.append(stdout_trim)
            lines.append("```")
        if rec["stderr"]:
            err_trim = rec["stderr"].strip()
            if len(err_trim) > max_chars:
                err_trim = err_trim[:max_chars] + "\n... [truncated]"
            lines.append("\n<details><summary>stderr</summary>")
            lines.append("")
            lines.append("```")
            lines.append(err_trim)
            lines.append("```")
            lines.append("</details>")
        lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if print_live:
        print("\n" + "=" * 80)
        print("SUMMARY WRITTEN")
        print(f"JSON: {json_path}")
        print(f"CSV : {csv_path}")
        print(f"MD  : {md_path}")
        print("=" * 80)

    return records


if __name__ == "__main__":
    run_scripts_and_summarize()
