# Trend Group A/B Analysis

This repository contains a small analysis tool for splitting paired `Y_A` / `Y_B` time series into their own trend groups, aligning those group boundaries, measuring A/B similarity inside each aligned segment, and exporting both tabular results and visualization charts.

## Files

- `analyze_trends.py`: main script for reading Excel, grouping trends, scoring similarity, and generating outputs.
- `test_analyze_trends.py`: regression tests for normalization, grouping, scoring, CLI export, and plot generation.
- `数据.xlsx`: source workbook used in this analysis.
- `output/`: generated CSV summaries and per-sheet PNG charts.

## What the script does

For each sheet in the Excel workbook:

1. Normalizes headers so both `Y_A/Y_B` and `Y-A /Y-B ` styles are accepted.
2. Computes first-order differences for A and B separately.
3. Treats flat points as continuing the nearest non-zero trend.
4. Builds independent trend groups for `Y_A` and `Y_B`.
5. Merges short groups shorter than `--min-group-len` into adjacent major trends.
6. Uses the union of A-group and B-group boundaries to form aligned analysis segments.
7. Computes, for each aligned segment:
   - trend relation such as `A升B升`, `A降B降`, `A升B降`, `A降B升`
   - `A_group_id` and `B_group_id`
   - start/end values
   - slope for A and B
   - normalized slope gap
   - Pearson correlation
   - similarity score and `high/medium/low` label
8. Writes summary CSVs and per-series group charts.

## Usage

Install dependencies temporarily with `uv` and run:

```powershell
uv run --with pandas --with openpyxl --with matplotlib python .\analyze_trends.py --input .\数据.xlsx --output-dir .\output
```

Optional arguments:

```powershell
uv run --with pandas --with openpyxl --with matplotlib python .\analyze_trends.py `
  --input .\数据.xlsx `
  --output-dir .\output `
  --min-group-len 4
```

## Outputs

After running, the script generates:

- `output/group_details.csv`: one row per aligned analysis segment
- `output/sheet_summary.csv`: one row per sheet summary
- `output/Sheet1_Y_A_groups.png`, `output/Sheet1_Y_B_groups.png`, ...: separate A/B charts with that series' own group boundaries and aligned segment shading

## Test

Run tests with:

```powershell
uv run --with pandas --with openpyxl --with matplotlib --with pytest python -m pytest test_analyze_trends.py -q
```

## Notes

- The grouping strategy is heuristic and designed for readability rather than globally optimal segmentation.
- Increasing `--min-group-len` will reduce short-term fragmentation and usually produce fewer, longer groups.
- The output charts are intended to help review whether each series' split boundaries are reasonable before changing thresholds.
