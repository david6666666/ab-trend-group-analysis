from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_trends import (  # noqa: E402
    analyze_sheet,
    build_groups,
    compute_jump_thresholds,
    compute_group_metrics,
    get_series_labels,
    normalize_columns,
    plot_sheet_groups,
)


def test_normalize_columns_uses_second_and_third_columns():
    df = pd.DataFrame({"时间": [0, 1], "压力值": [1, 2], "温度值": [3, 4]})

    normalized = normalize_columns(df)

    assert list(normalized.columns) == ["X", "Y_A", "Y_B"]
    assert normalized["Y_A"].tolist() == [1, 2]
    assert normalized["Y_B"].tolist() == [3, 4]


def test_get_series_labels_returns_original_b_c_names():
    df = pd.DataFrame({"时间": [0, 1], "压力值": [1, 2], "温度值": [3, 4]})

    labels = get_series_labels(df)

    assert labels == ("压力值", "温度值")


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


def test_build_groups_splits_large_jump_inside_same_trend():
    df = pd.DataFrame(
        {
            "时间": list(range(12)),
            "压力值": [1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21],
            "温度值": [2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22],
        }
    )

    groups = build_groups(df, min_group_len=3)

    assert len(groups) == 2
    assert groups[0]["start"] == 0
    assert groups[0]["end"] == 5
    assert groups[1]["start"] == 6
    assert groups[1]["end"] == 11


def test_compute_jump_thresholds_uses_sheet_local_distribution():
    df = pd.DataFrame(
        {
            "时间": list(range(8)),
            "压力值": [1, 2, 3, 4, 5, 15, 16, 17],
            "温度值": [2, 3, 4, 5, 6, 16, 17, 18],
        }
    )

    thresholds = compute_jump_thresholds(df)

    assert thresholds["Y_A"] > 1
    assert thresholds["Y_B"] > 1


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


def test_analyze_sheet_keeps_original_b_c_labels():
    df = pd.DataFrame(
        {
            "时间": [0, 1, 2, 3],
            "压力值": [5.0, 5.0, 6.0, 7.0],
            "温度值": [3.0, 3.0, 3.0, 4.0],
        }
    )

    details, _ = analyze_sheet(df, "Sheet1", min_group_len=2)

    assert set(details["series_a_name"]) == {"压力值"}
    assert set(details["series_b_name"]) == {"温度值"}


def test_analyze_sheet_marks_jump_split_in_details():
    df = pd.DataFrame(
        {
            "时间": list(range(12)),
            "压力值": [1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21],
            "温度值": [2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22],
        }
    )

    details, _ = analyze_sheet(df, "Sheet1", min_group_len=3)

    assert len(details) == 2
    assert details.iloc[0]["end_x"] == 5
    assert details.iloc[1]["start_x"] == 6


def test_cli_generates_summary_and_csv_outputs(tmp_path: Path):
    excel_path = tmp_path / "input.xlsx"
    output_dir = tmp_path / "out"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "时间": list(range(6)),
                "压力值": [1, 2, 3, 4, 3, 2],
                "温度值": [2, 3, 4, 5, 4, 3],
            }
        ).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame(
            {
                "采样点": list(range(6)),
                "流量": [5, 4, 3, 2, 3, 4],
                "转速": [4, 3, 2, 1, 2, 3],
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
            "时间": list(range(6)),
            "压力值": [1, 2, 3, 4, 3, 2],
            "温度值": [2, 3, 4, 5, 4, 3],
        }
    )

    groups = build_groups(df, min_group_len=2)
    output_path = tmp_path / "plot.png"

    plot_sheet_groups(df, groups, "Sheet1", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
