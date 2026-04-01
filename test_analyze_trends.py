from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_trends import (  # noqa: E402
    build_groups,
    compute_group_metrics,
    normalize_columns,
    plot_sheet_groups,
)


def test_normalize_columns_supports_multiple_header_styles():
    df = pd.DataFrame({"X": [0, 1], "Y-A ": [1, 2], "Y_B": [3, 4]})

    normalized = normalize_columns(df)

    assert list(normalized.columns) == ["X", "Y_A", "Y_B"]


def test_build_groups_merges_short_segments():
    df = pd.DataFrame(
        {
            "X": list(range(8)),
            "Y_A": [1, 2, 3, 2, 3, 4, 5, 6],
            "Y_B": [10, 9, 8, 9, 8, 7, 6, 5],
        }
    )

    groups = build_groups(df, min_group_len=3)

    assert len(groups) == 1
    assert groups[0]["start"] == 0
    assert groups[0]["end"] == len(df) - 1


def test_compute_group_metrics_handles_constant_segment_correlation():
    df = pd.DataFrame(
        {
            "X": [0, 1, 2],
            "Y_A": [5.0, 5.0, 5.0],
            "Y_B": [3.0, 3.0, 3.0],
        }
    )

    metrics = compute_group_metrics(df, "Sheet1", 1)

    assert metrics["trend_relation"] == "A\u5e73B\u5e73"
    assert metrics["pearson_corr"] is None
    assert metrics["similarity_level"] == "medium"


def test_cli_generates_summary_and_csv_outputs(tmp_path: Path):
    excel_path = tmp_path / "input.xlsx"
    output_dir = tmp_path / "out"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "X": list(range(6)),
                "Y_A": [1, 2, 3, 4, 3, 2],
                "Y_B": [2, 3, 4, 5, 4, 3],
            }
        ).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame(
            {
                "X": list(range(6)),
                "Y-A ": [5, 4, 3, 2, 3, 4],
                "Y-B ": [4, 3, 2, 1, 2, 3],
            }
        ).to_excel(writer, sheet_name="Sheet2", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "analyze_trends.py"),
            "--input",
            str(excel_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Sheet: Sheet1" in completed.stdout
    assert "Sheet: Sheet2" in completed.stdout
    assert (output_dir / "group_details.csv").exists()
    assert (output_dir / "sheet_summary.csv").exists()
    assert (output_dir / "Sheet1_groups.png").exists()
    assert (output_dir / "Sheet2_groups.png").exists()


def test_plot_sheet_groups_writes_png(tmp_path: Path):
    df = pd.DataFrame(
        {
            "X": list(range(6)),
            "Y_A": [1, 2, 3, 4, 3, 2],
            "Y_B": [2, 3, 4, 5, 4, 3],
        }
    )

    groups = build_groups(df, min_group_len=2)
    output_path = tmp_path / "plot.png"

    plot_sheet_groups(df, groups, "Sheet1", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
