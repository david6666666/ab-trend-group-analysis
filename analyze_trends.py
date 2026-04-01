from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TREND_LABELS = {1: "\u5347", -1: "\u964d", 0: "\u5e73"}
SIMILARITY_HIGH_SCORE = 0.75
SIMILARITY_MEDIUM_SCORE = 0.45


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = str(column).strip().replace("-", "_").replace(" ", "")
        if normalized.lower() == "x":
            rename_map[column] = "X"
        elif normalized.upper() == "Y_A":
            rename_map[column] = "Y_A"
        elif normalized.upper() == "Y_B":
            rename_map[column] = "Y_B"
        else:
            rename_map[column] = normalized

    normalized_df = df.rename(columns=rename_map).copy()
    required = {"X", "Y_A", "Y_B"}
    missing = required - set(normalized_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return normalized_df.loc[:, ["X", "Y_A", "Y_B"]]


def sign_of_delta(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def fill_zero_signs(signs: Iterable[int]) -> list[int]:
    filled = list(signs)
    last_non_zero = 0
    for index, value in enumerate(filled):
        if value == 0:
            filled[index] = last_non_zero
        else:
            last_non_zero = value

    next_non_zero = 0
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] == 0:
            filled[index] = next_non_zero
        else:
            next_non_zero = filled[index]
    return filled


def build_series_state(series: pd.Series) -> list[int]:
    return fill_zero_signs(sign_of_delta(delta) for delta in series.diff().fillna(0.0))


def build_initial_groups(states: list[int]) -> list[dict[str, object]]:
    if not states:
        return []

    groups: list[dict[str, object]] = []
    start = 0
    current_state = states[0]
    for index in range(1, len(states)):
        if states[index] != current_state:
            groups.append({"start": start, "end": index - 1, "trend": current_state})
            start = index
            current_state = states[index]
    groups.append({"start": start, "end": len(states) - 1, "trend": current_state})
    return groups


def group_length(group: dict[str, object]) -> int:
    return int(group["end"]) - int(group["start"]) + 1


def merge_short_groups(
    groups: list[dict[str, object]], min_group_len: int
) -> list[dict[str, object]]:
    if min_group_len <= 1 or len(groups) <= 1:
        return [dict(group) for group in groups]

    merged = [dict(group) for group in groups]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        next_groups: list[dict[str, object]] = []
        index = 0
        while index < len(merged):
            current = dict(merged[index])
            if group_length(current) >= min_group_len:
                next_groups.append(current)
                index += 1
                continue

            previous = next_groups[-1] if next_groups else None
            following = dict(merged[index + 1]) if index + 1 < len(merged) else None

            if previous and following and previous["trend"] == following["trend"]:
                previous["end"] = following["end"]
                index += 2
                changed = True
                continue

            if previous and following:
                if group_length(previous) >= group_length(following):
                    previous["end"] = current["end"]
                else:
                    following["start"] = current["start"]
                    next_groups.append(following)
                    index += 1
                index += 1
                changed = True
                continue

            if previous:
                previous["end"] = current["end"]
                index += 1
                changed = True
                continue

            if following:
                following["start"] = current["start"]
                next_groups.append(following)
                index += 2
                changed = True
                continue

            next_groups.append(current)
            index += 1

        merged = next_groups
    return merged


def build_series_groups(series: pd.Series, min_group_len: int = 3) -> list[dict[str, object]]:
    initial_groups = build_initial_groups(build_series_state(series.astype(float)))
    merged_groups = merge_short_groups(initial_groups, min_group_len=min_group_len)
    numbered_groups = []
    for group_id, group in enumerate(merged_groups, start=1):
        item = dict(group)
        item["group_id"] = group_id
        numbered_groups.append(item)
    return numbered_groups


def locate_group(index: int, groups: list[dict[str, object]]) -> dict[str, object]:
    for group in groups:
        if int(group["start"]) <= index <= int(group["end"]):
            return group
    raise ValueError(f"Index {index} not covered by groups")


def align_groups(
    a_groups: list[dict[str, object]],
    b_groups: list[dict[str, object]],
    length: int,
) -> list[dict[str, object]]:
    boundaries = {0, length}
    boundaries.update(int(group["start"]) for group in a_groups)
    boundaries.update(int(group["start"]) for group in b_groups)
    sorted_boundaries = sorted(boundaries)

    aligned: list[dict[str, object]] = []
    for index in range(len(sorted_boundaries) - 1):
        start = sorted_boundaries[index]
        end = sorted_boundaries[index + 1] - 1
        if start > end:
            continue
        a_group = locate_group(start, a_groups)
        b_group = locate_group(start, b_groups)
        aligned.append(
            {
                "start": start,
                "end": end,
                "a_group_id": int(a_group["group_id"]),
                "b_group_id": int(b_group["group_id"]),
                "a_group_trend": int(a_group["trend"]),
                "b_group_trend": int(b_group["trend"]),
            }
        )
    return aligned


def build_groups(df: pd.DataFrame, min_group_len: int = 3) -> list[dict[str, object]]:
    normalized = normalize_columns(df)
    a_groups = build_series_groups(normalized["Y_A"], min_group_len=min_group_len)
    b_groups = build_series_groups(normalized["Y_B"], min_group_len=min_group_len)
    return align_groups(a_groups, b_groups, len(normalized))


def safe_pearson(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 2:
        return None
    if a.nunique(dropna=False) <= 1 or b.nunique(dropna=False) <= 1:
        return None
    corr = a.corr(b)
    if pd.isna(corr):
        return None
    return float(corr)


def compute_similarity_score(
    direction_same: bool,
    slope_gap_norm: float,
    pearson_corr: float | None,
) -> float:
    direction_score = 1.0 if direction_same else 0.0
    slope_score = max(0.0, 1.0 - slope_gap_norm)
    corr_score = ((pearson_corr + 1.0) / 2.0) if pearson_corr is not None else 0.5
    return round(0.45 * direction_score + 0.35 * slope_score + 0.20 * corr_score, 4)


def classify_similarity_level(score: float) -> str:
    if score >= SIMILARITY_HIGH_SCORE:
        return "high"
    if score >= SIMILARITY_MEDIUM_SCORE:
        return "medium"
    return "low"


def classify_trend_relation(a_trend: int, b_trend: int) -> str:
    if a_trend == 1 and b_trend == 1:
        return "A\u5347B\u5347"
    if a_trend == -1 and b_trend == -1:
        return "A\u964dB\u964d"
    if a_trend == 1 and b_trend == -1:
        return "A\u5347B\u964d"
    if a_trend == -1 and b_trend == 1:
        return "A\u964dB\u5347"
    return f"A{TREND_LABELS[a_trend]}B{TREND_LABELS[b_trend]}"


def compute_group_metrics(
    segment: pd.DataFrame,
    sheet_name: str,
    group_id: int,
    a_group_id: int | None = None,
    b_group_id: int | None = None,
) -> dict[str, object]:
    normalized = normalize_columns(segment).reset_index(drop=True)
    a_start = float(normalized.loc[0, "Y_A"])
    a_end = float(normalized.loc[len(normalized) - 1, "Y_A"])
    b_start = float(normalized.loc[0, "Y_B"])
    b_end = float(normalized.loc[len(normalized) - 1, "Y_B"])

    length = len(normalized)
    step_count = max(length - 1, 1)
    a_delta = a_end - a_start
    b_delta = b_end - b_start
    a_trend = sign_of_delta(a_delta)
    b_trend = sign_of_delta(b_delta)
    a_slope = a_delta / step_count
    b_slope = b_delta / step_count
    scale = max(abs(a_end - a_start), abs(b_end - b_start), 1.0)
    slope_gap_norm = round(abs(a_slope - b_slope) / scale, 4)
    pearson_corr = safe_pearson(normalized["Y_A"], normalized["Y_B"])
    score = compute_similarity_score(a_trend == b_trend, slope_gap_norm, pearson_corr)
    if a_trend == 0 and b_trend == 0:
        score = min(score, 0.6)

    return {
        "sheet": sheet_name,
        "group_id": group_id,
        "A_group_id": a_group_id,
        "B_group_id": b_group_id,
        "start_x": normalized.loc[0, "X"],
        "end_x": normalized.loc[length - 1, "X"],
        "length": length,
        "A_trend": TREND_LABELS[a_trend],
        "B_trend": TREND_LABELS[b_trend],
        "trend_relation": classify_trend_relation(a_trend, b_trend),
        "A_start": a_start,
        "A_end": a_end,
        "B_start": b_start,
        "B_end": b_end,
        "A_slope": round(a_slope, 4),
        "B_slope": round(b_slope, 4),
        "slope_gap_norm": slope_gap_norm,
        "pearson_corr": None if pearson_corr is None else round(pearson_corr, 4),
        "similarity_score": score,
        "similarity_level": classify_similarity_level(score),
    }


def analyze_sheet(
    df: pd.DataFrame, sheet_name: str, min_group_len: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    normalized = normalize_columns(df)
    a_groups = build_series_groups(normalized["Y_A"], min_group_len=min_group_len)
    b_groups = build_series_groups(normalized["Y_B"], min_group_len=min_group_len)
    aligned_groups = align_groups(a_groups, b_groups, len(normalized))

    records: list[dict[str, object]] = []
    for group_id, group in enumerate(aligned_groups, start=1):
        segment = normalized.iloc[int(group["start"]) : int(group["end"]) + 1]
        records.append(
            compute_group_metrics(
                segment,
                sheet_name,
                group_id,
                a_group_id=int(group["a_group_id"]),
                b_group_id=int(group["b_group_id"]),
            )
        )

    details = pd.DataFrame(records)
    same_direction_mask = details["trend_relation"].isin(["A\u5347B\u5347", "A\u964dB\u964d"])
    high_group = details.sort_values("similarity_score", ascending=False).iloc[0]
    low_group = details.sort_values("similarity_score", ascending=True).iloc[0]
    summary = {
        "sheet": sheet_name,
        "aligned_group_count": int(len(details)),
        "A_group_count": int(len(a_groups)),
        "B_group_count": int(len(b_groups)),
        "same_direction_groups": int(same_direction_mask.sum()),
        "opposite_direction_groups": int((~same_direction_mask).sum()),
        "high_count": int((details["similarity_level"] == "high").sum()),
        "medium_count": int((details["similarity_level"] == "medium").sum()),
        "low_count": int((details["similarity_level"] == "low").sum()),
        "avg_similarity_score": round(float(details["similarity_score"].mean()), 4),
        "top_group_id": int(high_group["group_id"]),
        "top_group_relation": high_group["trend_relation"],
        "top_group_score": float(high_group["similarity_score"]),
        "low_group_id": int(low_group["group_id"]),
        "low_group_relation": low_group["trend_relation"],
        "low_group_score": float(low_group["similarity_score"]),
    }
    return details, summary


def sanitize_sheet_name(sheet_name: str) -> str:
    safe = []
    for char in sheet_name:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "sheet"


def plot_single_series_groups(
    df: pd.DataFrame,
    groups: list[dict[str, object]],
    sheet_name: str,
    output_path: Path,
    series_name: str,
    series_groups: list[dict[str, object]],
    boundary_color: str,
) -> None:
    normalized = normalize_columns(df).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 7))

    x_values = normalized["X"]
    ax.plot(x_values, normalized[series_name], label=series_name, color=boundary_color, linewidth=2.2)

    span_colors = ["#eef4fb", "#fff5e8"]
    for index, group in enumerate(groups):
        start_x = normalized.iloc[int(group["start"])]["X"]
        end_x = normalized.iloc[int(group["end"])]["X"]
        ax.axvspan(start_x, end_x, color=span_colors[index % 2], alpha=0.22, zorder=0)
        midpoint = (start_x + end_x) / 2
        ax.text(
            midpoint,
            1.01,
            f"S{index + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    for group in series_groups:
        start = int(group["start"])
        if start > 0:
            start_x = normalized.iloc[start]["X"]
            ax.axvline(start_x, color=boundary_color, linestyle="--", linewidth=1.4, alpha=0.95)

    ax.set_title(f"{sheet_name} {series_name} Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    legend_items = [
        Line2D([0], [0], color=boundary_color, linewidth=2.2, label=series_name),
        Line2D([0], [0], color=boundary_color, linestyle="--", linewidth=1.4, label=f"{series_name} group boundary"),
        Line2D([0], [0], color="#999999", linewidth=8, alpha=0.22, label="Aligned analysis segment"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sheet_groups(
    df: pd.DataFrame,
    groups: list[dict[str, object]],
    sheet_name: str,
    output_dir: Path,
    a_groups: list[dict[str, object]] | None = None,
    b_groups: list[dict[str, object]] | None = None,
) -> None:
    normalized = normalize_columns(df).reset_index(drop=True)
    if a_groups is None:
        a_groups = build_series_groups(normalized["Y_A"])
    if b_groups is None:
        b_groups = build_series_groups(normalized["Y_B"])

    safe_name = sanitize_sheet_name(sheet_name)
    plot_single_series_groups(
        normalized,
        groups,
        sheet_name,
        output_dir / f"{safe_name}_Y_A_groups.png",
        "Y_A",
        a_groups,
        "#1f77b4",
    )
    plot_single_series_groups(
        normalized,
        groups,
        sheet_name,
        output_dir / f"{safe_name}_Y_B_groups.png",
        "Y_B",
        b_groups,
        "#d62728",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend grouping and A/B similarity analysis.")
    parser.add_argument("--input", required=True, help="Path to the Excel file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for group_details.csv, sheet_summary.csv, and plot PNGs. Defaults to ./output",
    )
    parser.add_argument(
        "--min-group-len",
        type=int,
        default=3,
        help="Merge groups shorter than this length into adjacent major trends.",
    )
    return parser.parse_args()


def format_summary(summary: dict[str, object]) -> str:
    return (
        f"Sheet: {summary['sheet']}\n"
        f"  A groups: {summary['A_group_count']}\n"
        f"  B groups: {summary['B_group_count']}\n"
        f"  Aligned groups: {summary['aligned_group_count']}\n"
        f"  Same direction: {summary['same_direction_groups']}\n"
        f"  Opposite direction: {summary['opposite_direction_groups']}\n"
        f"  Similarity levels: high={summary['high_count']}, "
        f"medium={summary['medium_count']}, low={summary['low_count']}\n"
        f"  Avg similarity score: {summary['avg_similarity_score']}\n"
        f"  Top similar group: #{summary['top_group_id']} "
        f"({summary['top_group_relation']}, score={summary['top_group_score']})\n"
        f"  Lowest similar group: #{summary['low_group_id']} "
        f"({summary['low_group_relation']}, score={summary['low_group_score']})"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    excel = pd.ExcelFile(input_path)
    all_details: list[pd.DataFrame] = []
    all_summaries: list[dict[str, object]] = []
    for sheet_name in excel.sheet_names:
        sheet_df = pd.read_excel(input_path, sheet_name=sheet_name)
        normalized = normalize_columns(sheet_df)
        a_groups = build_series_groups(normalized["Y_A"], min_group_len=args.min_group_len)
        b_groups = build_series_groups(normalized["Y_B"], min_group_len=args.min_group_len)
        aligned_groups = align_groups(a_groups, b_groups, len(normalized))
        details, summary = analyze_sheet(normalized, sheet_name, min_group_len=args.min_group_len)
        plot_sheet_groups(
            normalized,
            aligned_groups,
            sheet_name,
            output_dir,
            a_groups=a_groups,
            b_groups=b_groups,
        )
        all_details.append(details)
        all_summaries.append(summary)

    details_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    details_path = output_dir / "group_details.csv"
    summary_path = output_dir / "sheet_summary.csv"
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Trend analysis completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Details CSV: {details_path}")
    print(f"Summary CSV: {summary_path}")
    for summary in all_summaries:
        print()
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
