from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


TREND_LABELS = {1: "\u5347", -1: "\u964d", 0: "\u5e73"}
SIMILARITY_HIGH_SCORE = 0.75
SIMILARITY_MEDIUM_SCORE = 0.45
MAX_GROUP_COUNT = 20


def compute_jump_thresholds(df: pd.DataFrame) -> dict[str, float]:
    normalized = normalize_columns(df)
    thresholds: dict[str, float] = {}
    for column in ["Y_A", "Y_B"]:
        diffs = normalized[column].astype(float).diff().abs().dropna()
        if diffs.empty:
            thresholds[column] = float("inf")
            continue
        q25 = float(diffs.quantile(0.25))
        q50 = float(diffs.quantile(0.50))
        q75 = float(diffs.quantile(0.75))
        q95 = float(diffs.quantile(0.95))
        iqr = q75 - q25
        # 跳变阈值按当前 sheet 当前列自己的波动水平自动计算，
        # 只把明显高于日常波动的点当成“组内跳变”。
        thresholds[column] = max(q95, q50 + 3 * iqr)
    return thresholds


def get_series_labels(df: pd.DataFrame) -> tuple[str, str]:
    if df.shape[1] < 3:
        raise ValueError("Each sheet must have at least three columns; B and C are required.")
    return str(df.columns[1]).strip(), str(df.columns[2]).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 不再依赖固定列名，而是把每个 sheet 的 A/B/C 三列分别映射成 X、Y_A、Y_B。
    # 这样无论原始表头叫“压力值/温度值”还是别的名字，都能自动处理。
    if df.shape[1] < 3:
        raise ValueError("Each sheet must have at least three columns; B and C are required.")
    normalized_df = df.iloc[:, :3].copy()
    normalized_df.columns = ["X", "Y_A", "Y_B"]
    return normalized_df


def sign_of_delta(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def fill_zero_signs(signs: Iterable[int]) -> list[int]:
    filled = list(signs)
    last_non_zero = 0
    # 第一遍向前填充：把“持平”点视为延续上一段趋势，避免平点把 group 切得过碎。
    for index, value in enumerate(filled):
        if value == 0:
            filled[index] = last_non_zero
        else:
            last_non_zero = value

    next_non_zero = 0
    # 第二遍向后补齐：处理序列开头连续持平、前面没有可继承趋势的情况。
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] == 0:
            filled[index] = next_non_zero
        else:
            next_non_zero = filled[index]
    return filled


def build_state_series(df: pd.DataFrame) -> list[tuple[int, int]]:
    # 先分别计算 A/B 的涨跌方向，再把它们拼成 (A趋势, B趋势) 的联合状态。
    a_signs = fill_zero_signs(sign_of_delta(delta) for delta in df["Y_A"].diff().fillna(0.0))
    b_signs = fill_zero_signs(sign_of_delta(delta) for delta in df["Y_B"].diff().fillna(0.0))
    return list(zip(a_signs, b_signs))


def build_initial_groups(states: list[tuple[int, int]]) -> list[dict[str, object]]:
    if not states:
        return []

    # 只要联合状态变化，就切出一个新的原始 group。
    groups: list[dict[str, object]] = []
    start = 0
    current_state = states[0]
    for index in range(1, len(states)):
        if states[index] != current_state:
            groups.append({"start": start, "end": index - 1, "state": current_state})
            start = index
            current_state = states[index]
    groups.append({"start": start, "end": len(states) - 1, "state": current_state})
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

            # 短段合并规则：
            # 1. 前后状态一致时直接桥接；
            # 2. 前后都存在但状态不同，就并到更长的邻居；
            # 3. 只剩一侧邻居时，并到那一侧。
            if previous and following and previous["state"] == following["state"]:
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


def split_groups_on_jumps(
    groups: list[dict[str, object]],
    df: pd.DataFrame,
    jump_thresholds: dict[str, float],
) -> list[dict[str, object]]:
    normalized = normalize_columns(df)
    refined_groups: list[dict[str, object]] = []

    for group in groups:
        start = int(group["start"])
        end = int(group["end"])
        split_points = [start]
        for index in range(max(start + 1, 1), end + 1):
            a_jump = abs(float(normalized.iloc[index]["Y_A"]) - float(normalized.iloc[index - 1]["Y_A"])) > jump_thresholds["Y_A"]
            b_jump = abs(float(normalized.iloc[index]["Y_B"]) - float(normalized.iloc[index - 1]["Y_B"])) > jump_thresholds["Y_B"]
            if a_jump or b_jump:
                split_points.append(index)

        split_points.append(end + 1)
        for left, right in zip(split_points, split_points[1:]):
            refined_groups.append(
                {
                    "start": left,
                    "end": right - 1,
                    "state": group["state"],
                }
            )
    return refined_groups


def merge_groups_to_max_count(
    groups: list[dict[str, object]], max_group_count: int
) -> list[dict[str, object]]:
    if max_group_count <= 0:
        raise ValueError("max_group_count must be greater than 0.")

    merged = [dict(group) for group in groups]
    while len(merged) > max_group_count:
        best_index = min(
            range(len(merged) - 1),
            key=lambda index: (
                merged[index]["state"] != merged[index + 1]["state"],
                group_length(merged[index]) + group_length(merged[index + 1]),
                index,
            ),
        )
        merged[best_index]["end"] = merged[best_index + 1]["end"]
        if merged[best_index]["state"] != merged[best_index + 1]["state"]:
            merged[best_index]["state"] = (0, 0)
        del merged[best_index + 1]
    return merged


def build_groups(
    df: pd.DataFrame,
    min_group_len: int = 3,
    max_group_count: int = MAX_GROUP_COUNT,
) -> list[dict[str, object]]:
    normalized = normalize_columns(df)
    states = build_state_series(normalized)
    initial_groups = build_initial_groups(states)
    merged_groups = merge_short_groups(initial_groups, min_group_len=min_group_len)
    jump_thresholds = compute_jump_thresholds(normalized)
    jumped_groups = split_groups_on_jumps(merged_groups, normalized, jump_thresholds)
    return merge_groups_to_max_count(jumped_groups, max_group_count=max_group_count)


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
    direction_relative_error: float,
    slope_gap_norm: float,
    pearson_corr: float | None,
) -> float:
    # 相似度由四部分组成：
    # 方向是否一致 + 方向变化幅度相对误差 + 斜率是否接近 + 组内形状相关性。
    # 权重偏向“趋势方向”和“方向幅度/斜率差”，相关系数作为辅助解释。
    direction_score = 1.0 if direction_same else 0.0
    relative_error_score = max(0.0, 1.0 - direction_relative_error)
    slope_score = max(0.0, 1.0 - slope_gap_norm)
    corr_score = ((pearson_corr + 1.0) / 2.0) if pearson_corr is not None else 0.5
    return round(
        0.40 * direction_score
        + 0.25 * relative_error_score
        + 0.25 * slope_score
        + 0.10 * corr_score,
        4,
    )


def compute_direction_relative_error(a_delta: float, b_delta: float) -> float:
    if a_delta == 0 and b_delta == 0:
        return 0.0
    scale = max(abs(a_delta), abs(b_delta))
    return round(min(abs(a_delta - b_delta) / scale, 1.0), 4)


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
    series_a_name: str | None = None,
    series_b_name: str | None = None,
) -> dict[str, object]:
    normalized = normalize_columns(segment).reset_index(drop=True)
    if series_a_name is None or series_b_name is None:
        series_a_name, series_b_name = get_series_labels(segment)
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
    # 用两条曲线在该段内的总体变化幅度做归一化，避免不同量纲下斜率差失真。
    scale = max(abs(a_end - a_start), abs(b_end - b_start), 1.0)
    slope_gap_norm = round(abs(a_slope - b_slope) / scale, 4)
    direction_relative_error = compute_direction_relative_error(a_delta, b_delta)
    pearson_corr = safe_pearson(normalized["Y_A"], normalized["Y_B"])
    score = compute_similarity_score(
        a_trend == b_trend,
        direction_relative_error,
        slope_gap_norm,
        pearson_corr,
    )
    # A、B 都完全不变时，不把这类“静止段”评成过高相似，压到中档上限。
    if a_trend == 0 and b_trend == 0:
        score = min(score, 0.6)

    return {
        "sheet": sheet_name,
        "group_id": group_id,
        "series_a_name": series_a_name,
        "series_b_name": series_b_name,
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
        "direction_relative_error": direction_relative_error,
        "slope_gap_norm": slope_gap_norm,
        "pearson_corr": None if pearson_corr is None else round(pearson_corr, 4),
        "similarity_score": score,
        "similarity_level": classify_similarity_level(score),
    }


def analyze_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    min_group_len: int,
    max_group_count: int = MAX_GROUP_COUNT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    series_a_name, series_b_name = get_series_labels(df)
    normalized = normalize_columns(df)
    groups = build_groups(
        normalized,
        min_group_len=min_group_len,
        max_group_count=max_group_count,
    )
    records: list[dict[str, object]] = []
    for group_id, group in enumerate(groups, start=1):
        # 每个 group 都对应原始序列上的一个连续区间，区间内再计算趋势和相似度指标。
        segment = normalized.iloc[int(group["start"]) : int(group["end"]) + 1]
        records.append(
            compute_group_metrics(
                segment,
                sheet_name,
                group_id,
                series_a_name=series_a_name,
                series_b_name=series_b_name,
            )
        )

    details = pd.DataFrame(records)
    same_direction_mask = details["trend_relation"].isin(["A\u5347B\u5347", "A\u964dB\u964d"])
    high_group = details.sort_values("similarity_score", ascending=False).iloc[0]
    low_group = details.sort_values("similarity_score", ascending=True).iloc[0]
    summary = {
        "sheet": sheet_name,
        "group_count": int(len(details)),
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


def plot_sheet_groups(
    df: pd.DataFrame,
    groups: list[dict[str, object]],
    sheet_name: str,
    output_path: Path,
) -> None:
    normalized = normalize_columns(df).reset_index(drop=True)
    series_a_name, series_b_name = get_series_labels(df)
    fig, ax = plt.subplots(figsize=(16, 7))

    x_values = normalized["X"]
    ax.plot(x_values, normalized["Y_A"], label=series_a_name, color="#1f77b4", linewidth=2.0)
    ax.plot(x_values, normalized["Y_B"], label=series_b_name, color="#d62728", linewidth=2.0)

    span_colors = ["#eef4fb", "#fff5e8"]
    for index, group in enumerate(groups):
        start = int(group["start"])
        end = int(group["end"])
        start_x = normalized.iloc[start]["X"]
        end_x = normalized.iloc[end]["X"]
        # 底色块表示当前分组覆盖的区间，竖线表示分组起点，文字是 group 编号。
        ax.axvspan(start_x, end_x, color=span_colors[index % len(span_colors)], alpha=0.35, zorder=0)
        if start > 0:
            ax.axvline(start_x, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
        midpoint = (start_x + end_x) / 2
        ax.text(
            midpoint,
            1.01,
            f"G{index + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    ax.set_title(f"{sheet_name} Trend Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--max-groups",
        type=int,
        default=MAX_GROUP_COUNT,
        help=f"Maximum groups per sheet after all splits and merges. Defaults to {MAX_GROUP_COUNT}.",
    )
    return parser.parse_args()


def format_summary(summary: dict[str, object]) -> str:
    return (
        f"Sheet: {summary['sheet']}\n"
        f"  Groups: {summary['group_count']}\n"
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
        groups = build_groups(
            sheet_df,
            min_group_len=args.min_group_len,
            max_group_count=args.max_groups,
        )
        details, summary = analyze_sheet(
            sheet_df,
            sheet_name,
            min_group_len=args.min_group_len,
            max_group_count=args.max_groups,
        )
        plot_sheet_groups(
            sheet_df,
            groups,
            sheet_name,
            output_dir / f"{sanitize_sheet_name(sheet_name)}_groups.png",
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
