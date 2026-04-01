from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_trends import (  # noqa: E402
    align_groups,
    analyze_sheet,
    build_groups,
    build_series_groups,
    compute_group_metrics,
    normalize_columns,
    plot_sheet_groups,
)


def test_normalize_columns_supports_multiple_header_styles():
    df = pd.DataFrame({"X": [0, 1], "Y-A ": [1, 2], "Y_B": [3, 4]})

    normalized = normalize_columns(df)

    assert list(normalized.columns) == ["X", "Y_A", "Y_B"]


def test_build_series_groups_merges_short_segments():
    series = pd.Series([1, 2, 3, 2, 3, 4, 5, 6])

    groups = build_series_groups(series, min_group_len=3)

    assert len(groups) == 1
    assert groups[0]["start"] == 0
    assert groups[0]["end"] == len(series) - 1


def test_align_groups_uses_a_and_b_boundaries():
    df = pd.DataFrame(
        {
            "X": list(range(10)),
            "Y_A": [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            "Y_B": [10, 9, 8, 7, 8, 9, 10, 9, 8, 7],
        }
    )

    a_groups = build_series_groups(df["Y_A"], min_group_len=2)
    b_groups = build_series_groups(df["Y_B"], min_group_len=2)
    groups = align_groups(a_groups, b_groups, len(df))

    assert len(groups) >= 2
    assert groups[0]["start"] == 0
    assert groups[-1]["end"] == len(df) - 1
    assert "a_group_id" in groups[0]
    assert "b_group_id" in groups[0]


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


def test_analyze_sheet_returns_separate_a_b_group_ids():
    df = pd.DataFrame(
        {
            "X": list(range(6)),
            "Y_A": [1, 2, 3, 2, 1, 1],
            "Y_B": [2, 2, 3, 4, 3, 2],
        }
    )

    details, summary = analyze_sheet(df, "Sheet1", min_group_len=2)

    assert "A_group_id" in details.columns
    assert "B_group_id" in details.columns
    assert summary["aligned_group_count"] == len(details)
    assert summary["A_group_count"] >= 1
    assert summary["B_group_count"] >= 1


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
    assert "A groups:" in completed.stdout
    assert "B groups:" in completed.stdout
    assert (output_dir / "group_details.csv").exists()
    assert (output_dir / "sheet_summary.csv").exists()
    assert (output_dir / "Sheet1_Y_A_groups.png").exists()
    assert (output_dir / "Sheet1_Y_B_groups.png").exists()
    assert (output_dir / "Sheet2_Y_A_groups.png").exists()
    assert (output_dir / "Sheet2_Y_B_groups.png").exists()


def test_plot_sheet_groups_writes_png(tmp_path: Path):
    df = pd.DataFrame(
        {
            "X": list(range(6)),
            "Y_A": [1, 2, 3, 4, 3, 2],
            "Y_B": [2, 3, 4, 5, 4, 3],
        }
    )

    groups = build_groups(df, min_group_len=2)
    output_dir = tmp_path / "plots"

    plot_sheet_groups(df, groups, "Sheet1", output_dir)

    a_plot = output_dir / "Sheet1_Y_A_groups.png"
    b_plot = output_dir / "Sheet1_Y_B_groups.png"
    assert a_plot.exists()
    assert b_plot.exists()
    assert a_plot.stat().st_size > 0
    assert b_plot.stat().st_size > 0
