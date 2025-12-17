#!/usr/bin/env python3
"""Run all R (synchronization) analysis scripts with summary artifacts."""

from pathlib import Path

from summarize_r_analysis import DEFAULT_SCRIPTS, run_scripts_and_summarize


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "reports"
    run_scripts_and_summarize(DEFAULT_SCRIPTS, output_dir=output_dir, print_live=True)

