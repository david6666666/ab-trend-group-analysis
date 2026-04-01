# Trend Group A/B Analysis

This repository contains a small analysis tool for splitting paired `Y_A` / `Y_B` time series into trend groups, measuring A/B similarity inside each group, and exporting both tabular results and visualization charts.

## Files

- `analyze_trends.py`: main script for reading Excel, grouping trends, scoring similarity, and generating outputs.
- `test_analyze_trends.py`: regression tests for normalization, grouping, scoring, CLI export, and plot generation.
- `数据.xlsx`: source workbook used in this analysis.
- `output/`: generated CSV summaries and per-sheet PNG charts.

## What the script does

For each sheet in the Excel workbook:

1. Normalizes headers so both `Y_A/Y_B` and `Y-A /Y-B ` styles are accepted.
2. Computes first-order differences for A and B.
3. Treats flat points as continuing the nearest non-zero trend.
4. Splits the sequence whenever the `(A trend, B trend)` state changes.
5. Merges short groups shorter than `--min-group-len` into adjacent major trends.
6. Computes, for each group:
   - trend relation such as `A升B升`, `A降B降`, `A升B降`, `A降B升`
   - start/end values
   - slope for A and B
   - normalized slope gap
   - Pearson correlation
   - similarity score and `high/medium/low` label
7. Writes summary CSVs and group boundary charts.

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

- `output/group_details.csv`: one row per trend group
- `output/sheet_summary.csv`: one row per sheet summary
- `output/Sheet1_groups.png`, `output/Sheet2_groups.png`, ...: original A/B curves with group boundaries and shaded segments

## Test

Run tests with:

```powershell
uv run --with pandas --with openpyxl --with matplotlib --with pytest python -m pytest test_analyze_trends.py -q
```

## Notes

- The grouping strategy is heuristic and designed for readability rather than globally optimal segmentation.
- Increasing `--min-group-len` will reduce short-term fragmentation and usually produce fewer, longer groups.
- The output charts are intended to help review whether the split boundaries are reasonable before changing thresholds.
