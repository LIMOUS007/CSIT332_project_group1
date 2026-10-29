"""
build_register_xlsx.py
======================
Builds docs/dataset_register.xlsx from the table in docs/dataset_register.md,
so the submitted register can never drift from the one in the repository.
The workbook holds the register table only: one row per dataset, eight columns.

Run
---
    python docs/build_register_xlsx.py
"""

import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

DOCS_DIR = Path(__file__).resolve().parent
MD_PATH  = DOCS_DIR / "dataset_register.md"
OUT_XLSX = DOCS_DIR / "dataset_register.xlsx"

COLUMNS = [
    "Name", "Source and URL", "Rows", "Features",
    "Target and binarisation rule", "Class prevalence", "Missingness", "Licence",
]
WIDTHS = [30, 45, 12, 10, 70, 12, 22, 45]

LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def read_table(md_text):
    """Return the rows of the table under '## Register' as lists of cells."""
    section = md_text.split("## Register", 1)[1].split("\n## ", 1)[0]
    lines = [l.strip() for l in section.splitlines() if l.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert header == COLUMNS, f"Unexpected register columns: {header}"
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    for r in rows:
        assert len(r) == len(COLUMNS), f"Row has {len(r)} cells: {r[0]}"
    return rows


def plain(cell):
    """Strip markdown: links become 'label: url', backticks are removed."""
    return LINK.sub(r"\1: \2", cell).replace("`", "").replace(" / ", "\n")


rows = read_table(MD_PATH.read_text(encoding="utf-8"))

wb = Workbook()
ws = wb.active
ws.title = "Register"
ws.append(COLUMNS)

for r in rows:
    name, source, n_rows, n_feat, target, prev, miss, licence = r
    ws.append([
        name,
        plain(source),
        int(n_rows.replace(",", "")),
        int(n_feat),
        target.replace("`", ""),
        float(prev),
        miss,
        licence,
    ])

header_fill = PatternFill("solid", start_color="D9E1F2")
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = header_fill
    cell.alignment = Alignment(vertical="center", wrap_text=True)

for row in ws.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    row[2].number_format = "#,##0"
    row[5].number_format = "0.0000"

for i, w in enumerate(WIDTHS):
    ws.column_dimensions[chr(ord("A") + i)].width = w

ws.freeze_panes = "B2"
ws.auto_filter.ref = ws.dimensions

wb.save(OUT_XLSX)
print(f"Saved {OUT_XLSX.name}: {len(rows)} datasets x {len(COLUMNS)} columns")
