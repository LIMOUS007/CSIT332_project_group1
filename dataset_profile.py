"""Fast triage profiler for unfamiliar tabular datasets.

Usage:
    from dataset_profile import describe_dataset
    report = describe_dataset(df, target=y, name="OpenML 45548")

Returns a dict of DataFrames so you can also use it programmatically:
    report["overview"], report["numeric"], report["categorical"],
    report["warnings"], report["meta"]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pdt

__all__ = ["quick_look", "describe_dataset", "profile_columns", "compare_datasets"]


# ------------------------------------------------------------ quick look ---

def quick_look(df: pd.DataFrame, target=None) -> None:
    """The short version: one table, one verdict per column.

    Use this for a first glance; use describe_dataset() when a dataset looks
    worth committing to. Assumes plain scalar columns (no list/dict cells).
    """
    n = len(df)
    print(f"{n:,} rows x {df.shape[1]} cols | {df.duplicated().sum():,} dup rows "
          f"| {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    def verdict(s: pd.Series) -> str:
        uniq, null = s.nunique(), s.isna().mean()
        if uniq <= 1:
            return "DROP: constant"
        if uniq == s.count() and n > 20:
            return "DROP: id-like"
        if null > 0.5:
            return "DROP?: mostly missing"
        if pdt.is_numeric_dtype(s):
            return "skewed -> log1p" if abs(s.skew()) > 2 else ""
        return "hi-card -> target/freq encode" if uniq > 50 else ""

    with pd.option_context("display.max_rows", 300, "display.width", 160):
        print(pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "null_%": (df.isna().mean() * 100).round(1),
            "unique": df.nunique(),
            "most_common": [df[c].mode().iloc[0] if df[c].notna().any() else ""
                            for c in df.columns],
            "verdict": [verdict(df[c]) for c in df.columns],
        }).to_string())

    if target is not None:
        y = df[target] if isinstance(target, str) else pd.Series(target)
        print(f"\nTARGET '{y.name}' | {y.nunique()} unique | {y.isna().sum()} missing")
        print(y.value_counts(normalize=True).mul(100).round(1).head(10)
              if y.nunique() <= 20 else y.describe())


# ---------------------------------------------------------------- helpers ---

def _hdr(title: str, width: int = 78) -> str:
    return f"\n{'=' * width}\n {title}\n{'=' * width}"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


def _kind(s: pd.Series) -> str:
    """Coarse column kind: numeric / bool / datetime / cat."""
    if pdt.is_bool_dtype(s):
        return "bool"
    if pdt.is_numeric_dtype(s):
        return "numeric"
    if pdt.is_datetime64_any_dtype(s) or pdt.is_timedelta64_dtype(s):
        return "datetime"
    return "cat"


def _looks_numeric(s: pd.Series, sample: int = 1000) -> bool:
    """Text column that is really numbers ('3.4', '1,200', '12%')."""
    vals = s.dropna()
    if vals.empty:
        return False
    vals = vals.head(sample).astype(str).str.strip()
    vals = vals.str.replace(",", "", regex=False).str.rstrip("%")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coerced = pd.to_numeric(vals, errors="coerce")
    return coerced.notna().mean() > 0.95


def _looks_datetime(s: pd.Series, sample: int = 500) -> bool:
    vals = s.dropna()
    if vals.empty:
        return False
    vals = vals.head(sample).astype(str)
    # Cheap gate: dates almost always contain a separator + a 4-digit year.
    if not vals.str.contains(r"\d{4}|[/\-:]", regex=True).any():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            coerced = pd.to_datetime(vals, errors="coerce", format="mixed")
        except Exception:
            return False
    return coerced.notna().mean() > 0.95


def _outlier_pct(s: pd.Series) -> float:
    """Share of values outside the 1.5*IQR fence."""
    v = s.dropna()
    if v.empty:
        return 0.0
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return 0.0
    return float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean() * 100)


def _safe(fn, default=np.nan):
    """Run ``fn`` with numpy/pandas noise muted; fall back to ``default`` on error.

    Constant or all-null columns make ``corr`` divide by a zero stddev, which is
    a warning we do not want printed in the middle of a report.
    """
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return fn()
    except Exception:
        return default


def _dup_column_groups(df: pd.DataFrame) -> list[list[str]]:
    """Columns that are exact copies of each other (hash first, then verify)."""
    buckets: dict[int, list[str]] = {}
    for c in df.columns:
        h = _safe(
            lambda c=c: int(pd.util.hash_pandas_object(df[c], index=False).sum()),
            default=None,
        )
        if h is not None:
            buckets.setdefault(h, []).append(c)
    groups = []
    for cols in buckets.values():
        if len(cols) < 2:
            continue
        base = cols[0]
        same = [c for c in cols[1:] if _safe(lambda c=c: df[base].equals(df[c]), False)]
        if same:
            groups.append([base, *same])
    return groups


# ------------------------------------------------------------ column pass ---

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missingness, cardinality and quality flags."""
    n = len(df)
    rows = []

    for col in df.columns:
        s = df[col]
        kind = _kind(s)
        nulls = int(s.isna().sum())
        null_pct = (nulls / n * 100) if n else 0.0
        uniq = int(_safe(lambda s=s: s.nunique(dropna=True), default=-1))
        card = (uniq / (n - nulls) * 100) if (n - nulls) and uniq > 0 else np.nan

        flags: list[str] = []
        if uniq < 0:
            flags.append("UNHASHABLE")
        elif uniq == 0:
            flags.append("ALL-NULL")
        elif uniq == 1:
            flags.append("CONSTANT")
        elif uniq == 2:
            flags.append("binary")

        if uniq > 1 and n - nulls:
            top_share = float(_safe(
                lambda s=s: s.value_counts(normalize=True, dropna=True).iloc[0], 0.0))
            if top_share >= 0.99:
                flags.append("near-const")
        if null_pct >= 50:
            flags.append("NULLS>50%")
        elif null_pct >= 20:
            flags.append("nulls>20%")

        if kind != "numeric" and uniq == n - nulls and n > 20:
            flags.append("ID?")
        if kind == "numeric" and pdt.is_integer_dtype(s) and uniq == n - nulls and n > 20:
            flags.append("ID?")

        if kind == "cat":
            if uniq > 50 and (card or 0) < 90:
                flags.append("hi-card")
            if _looks_numeric(s):
                flags.append("NUM-AS-TEXT")
            elif _looks_datetime(s):
                flags.append("DATE-AS-TEXT")
        if kind == "numeric":
            sk = _safe(lambda s=s: float(s.skew()), np.nan)
            if np.isfinite(sk) and abs(sk) > 2:
                flags.append("skewed")
            if _outlier_pct(s) > 5:
                flags.append("outliers")
            if n - nulls and float(_safe(lambda s=s: (s == 0).mean(), 0.0)) > 0.5:
                flags.append("zero-heavy")

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "kind": kind,
                "non_null": n - nulls,
                "null_%": round(null_pct, 1),
                "unique": uniq,
                "card_%": round(card, 1) if np.isfinite(card) else np.nan,
                "flags": ", ".join(flags),
            }
        )

    return pd.DataFrame(rows).set_index("column")


def _numeric_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna()
        if v.empty:
            continue
        q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        rows.append(
            {
                "column": c,
                "mean": v.mean(),
                "std": v.std(),
                "min": v.min(),
                "p1": q.loc[0.01],
                "p25": q.loc[0.25],
                "median": q.loc[0.50],
                "p75": q.loc[0.75],
                "p99": q.loc[0.99],
                "max": v.max(),
                "skew": _safe(lambda v=v: v.skew(), np.nan),
                "zero_%": (v == 0).mean() * 100,
                "neg_%": (v < 0).mean() * 100,
                "outlier_%": _outlier_pct(v),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("column").round(3)


def _categorical_table(df: pd.DataFrame, cols: list[str], sample: int = 5000) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        # Columns holding lists/dicts/arrays are unhashable -> value_counts blows up.
        vc = _safe(lambda s=s: s.value_counts(), default=None)
        if vc is None or vc.empty:
            rows.append({"column": c, "unique": np.nan, "top": "<unhashable>",
                         "top_%": np.nan, "2nd": "", "2nd_%": np.nan,
                         "len_avg": np.nan, "len_max": np.nan,
                         "samples": " | ".join(str(x)[:18] for x in s.head(3))})
            continue
        total = len(s)
        top, top_n = vc.index[0], vc.iloc[0]
        second = vc.index[1] if len(vc) > 1 else None
        second_n = vc.iloc[1] if len(vc) > 1 else 0
        lens = _safe(lambda s=s: s.head(sample).astype(str).str.len(), None)
        rows.append(
            {
                "column": c,
                "unique": len(vc),
                "top": str(top)[:24],
                "top_%": round(top_n / total * 100, 1),
                "2nd": str(second)[:24] if second is not None else "",
                "2nd_%": round(second_n / total * 100, 1),
                "len_avg": round(float(lens.mean()), 1) if lens is not None else np.nan,
                "len_max": int(lens.max()) if lens is not None else np.nan,
                "samples": " | ".join(str(x)[:18] for x in vc.index[:4]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("column")


# ----------------------------------------------------------------- target ---

def _target_report(y: pd.Series, n_rows: int) -> tuple[list[str], pd.DataFrame]:
    lines, tbl = [], pd.DataFrame()
    nulls = int(y.isna().sum())
    uniq = int(y.nunique(dropna=True))
    numeric = pdt.is_numeric_dtype(y) and not pdt.is_bool_dtype(y)
    task = "regression" if (numeric and uniq > 20) else "classification"

    lines.append(f"name        : {y.name}")
    lines.append(f"dtype       : {y.dtype}   (inferred task: {task})")
    lines.append(f"missing     : {nulls:,} ({nulls / n_rows * 100:.1f}%)"
                 + ("   <-- drop or impute these rows" if nulls else ""))

    if task == "regression":
        v = y.dropna()
        q = v.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        lines.append(
            f"range       : min={v.min():,.4g}  p1={q.loc[0.01]:,.4g}  "
            f"med={q.loc[0.5]:,.4g}  p99={q.loc[0.99]:,.4g}  max={v.max():,.4g}"
        )
        sk = float(_safe(lambda: v.skew(), np.nan))
        lines.append(f"mean/std    : {v.mean():,.4g} / {v.std():,.4g}   skew={sk:.2f}")
        if np.isfinite(sk) and abs(sk) > 1:
            lines.append("            -> heavy skew: consider log1p / target transform")
        if (v <= 0).any() and (v > 0).mean() > 0.9:
            lines.append("            -> contains <=0 values: log1p unsafe without a shift")
    else:
        vc = y.value_counts(dropna=False)
        tbl = pd.DataFrame({"count": vc, "share_%": (vc / len(y) * 100).round(2)})
        lines.append(f"classes     : {uniq}")
        if len(vc) > 1:
            ratio = vc.iloc[0] / max(vc.iloc[-1], 1)
            lines.append(f"imbalance   : {ratio:,.1f}:1 (largest vs smallest)")
            if ratio > 10:
                lines.append("            -> strong imbalance: stratify splits, "
                             "use class_weight / PR-AUC not accuracy")
            rare = vc[vc < 10]
            if len(rare):
                lines.append(f"            -> {len(rare)} class(es) with <10 rows: "
                             "merge or drop before stratifying")
    return lines, tbl


# ------------------------------------------------------------------- main ---

def describe_dataset(
    df: pd.DataFrame,
    target: pd.Series | str | None = None,
    name: str = "dataset",
    corr_threshold: float = 0.95,
    max_corr_cols: int = 300,
    preview: int = 3,
    show: bool = True,
) -> dict:
    """Profile a tabular dataset and print a triage report.

    Parameters
    ----------
    df      : feature frame (target may be included or passed separately).
    target  : column name in ``df``, a standalone Series, or None.
    preview : number of head rows to print (0 disables).

    Returns a dict of DataFrames: overview / numeric / categorical /
    corr_pairs / target_dist / warnings / meta.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    y = None
    if isinstance(target, str):
        if target in df.columns:
            y = df[target]
            df = df.drop(columns=[target])
        else:
            raise KeyError(f"target column {target!r} not in dataframe")
    elif target is not None:
        y = pd.Series(target)
        if y.name is None:
            y.name = "target"
        if len(y) != len(df):
            raise ValueError(f"target length {len(y)} != dataframe length {len(df)}")

    n_rows, n_cols = df.shape
    overview = profile_columns(df) if n_cols else pd.DataFrame()

    num_cols = [c for c in df.columns if _kind(df[c]) == "numeric"]
    cat_cols = [c for c in df.columns if _kind(df[c]) in ("cat", "bool")]
    dt_cols = [c for c in df.columns if _kind(df[c]) == "datetime"]

    numeric_tbl = _numeric_table(df, num_cols) if num_cols else pd.DataFrame()
    cat_tbl = _categorical_table(df, cat_cols) if cat_cols else pd.DataFrame()

    dup_rows = int(_safe(lambda: df.duplicated().sum(), -1))
    dup_groups = _dup_column_groups(df) if n_cols else []
    mem = float(_safe(lambda: df.memory_usage(deep=True).sum(), np.nan))
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum()) if n_cols else 0

    # --- redundant / leaky numeric pairs -----------------------------------
    corr_pairs = pd.DataFrame()
    if 1 < len(num_cols) <= max_corr_cols:
        cm = _safe(lambda: df[num_cols].corr(numeric_only=True).abs(), None)
        if cm is not None:
            mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            pairs = cm.where(mask).stack()
            pairs = pairs[pairs >= corr_threshold].sort_values(ascending=False)
            if len(pairs):
                corr_pairs = (
                    pairs.head(25).rename("abs_corr").round(3).reset_index()
                    .rename(columns={"level_0": "col_a", "level_1": "col_b"})
                )

    # --- target -------------------------------------------------------------
    target_lines: list[str] = []
    target_dist = pd.DataFrame()
    leaks = pd.DataFrame()
    if y is not None:
        target_lines, target_dist = _target_report(y, n_rows)
        if pdt.is_numeric_dtype(y) and not pdt.is_bool_dtype(y) and num_cols:
            c = _safe(lambda: df[num_cols].corrwith(y, numeric_only=True).abs(), None)
            if c is not None:
                c = c[c >= 0.98].sort_values(ascending=False)
                if len(c):
                    leaks = c.round(4).rename("abs_corr_with_target").to_frame()

    # --- warnings -----------------------------------------------------------
    warns: list[tuple[str, str]] = []

    def _cols_with(flag: str) -> list[str]:
        if overview.empty:
            return []
        return overview.index[overview["flags"].str.contains(flag, regex=False)].tolist()

    if n_rows == 0:
        warns.append(("CRITICAL", "dataset has 0 rows"))
    if dup_rows > 0:
        warns.append(("HIGH", f"{dup_rows:,} duplicate rows "
                              f"({dup_rows / max(n_rows, 1) * 100:.1f}%) -> drop_duplicates()"))
    for grp in dup_groups:
        warns.append(("HIGH", f"identical columns: {grp} -> keep one"))
    for c in _cols_with("ALL-NULL"):
        warns.append(("HIGH", f"'{c}' is entirely null -> drop"))
    for c in _cols_with("CONSTANT"):
        warns.append(("HIGH", f"'{c}' is constant -> drop (no signal)"))
    for c in _cols_with("UNHASHABLE"):
        warns.append(("HIGH", f"'{c}' holds lists/dicts/arrays -> flatten or drop; "
                              "most sklearn steps will reject it"))
    for c in _cols_with("NUM-AS-TEXT"):
        warns.append(("HIGH", f"'{c}' is numbers stored as text -> pd.to_numeric()"))
    for c in _cols_with("DATE-AS-TEXT"):
        warns.append(("MED", f"'{c}' looks like dates stored as text -> pd.to_datetime()"))
    for c in _cols_with("ID?"):
        warns.append(("MED", f"'{c}' is unique per row (identifier) -> drop from features"))
    for c in _cols_with("NULLS>50%"):
        warns.append(("MED", f"'{c}' is >50% missing -> drop or add a missingness flag"))
    for c in _cols_with("near-const"):
        warns.append(("MED", f"'{c}' is >=99% one value -> near-zero variance"))
    hi = _cols_with("hi-card")
    if hi:
        warns.append(("MED", f"{len(hi)} high-cardinality categorical(s) "
                             f"(one-hot will explode): {hi[:6]}"
                             f"{' ...' if len(hi) > 6 else ''}"))
    if len(corr_pairs):
        warns.append(("MED", f"{len(corr_pairs)} numeric pair(s) with |r| >= "
                             f"{corr_threshold} -> redundant features"))
    if len(leaks):
        warns.append(("CRITICAL", f"possible target leakage: {list(leaks.index)} "
                                  "correlate >=0.98 with the target"))
    if missing_cells and total_cells:
        pct = missing_cells / total_cells * 100
        if pct > 10:
            warns.append(("MED", f"{pct:.1f}% of all cells are missing"))

    warn_df = pd.DataFrame(warns, columns=["severity", "issue"]) if warns else pd.DataFrame(
        columns=["severity", "issue"]
    )

    meta = pd.Series(
        {
            "name": name,
            "rows": n_rows,
            "cols": n_cols,
            "numeric": len(num_cols),
            "categorical": len(cat_cols),
            "datetime": len(dt_cols),
            "memory": _human_bytes(mem) if np.isfinite(mem) else "n/a",
            "duplicate_rows": dup_rows,
            "missing_cells_%": round(missing_cells / total_cells * 100, 2) if total_cells else 0.0,
            "clean_columns": int((overview["flags"] == "").sum()) if not overview.empty else 0,
        }
    )

    if show:
        _print_report(meta, overview, numeric_tbl, cat_tbl, corr_pairs,
                      target_lines, target_dist, warn_df, df, preview)

    return {
        "meta": meta,
        "overview": overview,
        "numeric": numeric_tbl,
        "categorical": cat_tbl,
        "corr_pairs": corr_pairs,
        "target_dist": target_dist,
        "leakage": leaks,
        "warnings": warn_df,
    }


def _print_report(meta, overview, numeric_tbl, cat_tbl, corr_pairs,
                  target_lines, target_dist, warn_df, df, preview):
    opts = ("display.max_rows", 500, "display.max_columns", 100,
            "display.width", 200, "display.max_colwidth", 40)
    with pd.option_context(*opts):
        print(_hdr(f"DATASET: {meta['name']}"))
        print(
            f" {meta['rows']:,} rows x {meta['cols']:,} cols   |   {meta['memory']}\n"
            f" types: {meta['numeric']} numeric, {meta['categorical']} categorical, "
            f"{meta['datetime']} datetime\n"
            f" duplicate rows: {meta['duplicate_rows']:,}   |   "
            f"missing cells: {meta['missing_cells_%']}%   |   "
            f"clean columns: {meta['clean_columns']}/{meta['cols']}"
        )

        if target_lines:
            print(_hdr("TARGET"))
            for line in target_lines:
                print(" " + line)
            if not target_dist.empty:
                print()
                print(target_dist.head(20).to_string())

        if not warn_df.empty:
            print(_hdr(f"ISSUES ({len(warn_df)})"))
            order = {"CRITICAL": 0, "HIGH": 1, "MED": 2}
            wd = warn_df.sort_values("severity", key=lambda s: s.map(order))
            for sev, issue in wd.itertuples(index=False):
                print(f" [{sev:<8}] {issue}")
        else:
            print(_hdr("ISSUES"))
            print(" none detected")

        if not overview.empty:
            print(_hdr("COLUMNS"))
            print(overview.to_string())

        if not numeric_tbl.empty:
            print(_hdr("NUMERIC DETAIL"))
            print(numeric_tbl.to_string())

        if not cat_tbl.empty:
            print(_hdr("CATEGORICAL DETAIL"))
            print(cat_tbl.to_string())

        if not corr_pairs.empty:
            print(_hdr("HIGHLY CORRELATED NUMERIC PAIRS"))
            print(corr_pairs.to_string(index=False))

        if preview:
            print(_hdr(f"HEAD ({preview})"))
            print(df.head(preview).to_string())
        print()


# --------------------------------------------------------- multi-dataset ---

def compare_datasets(datasets: dict[str, pd.DataFrame], show: bool = True) -> pd.DataFrame:
    """One row per dataset - for picking which of many to work on first."""
    rows = []
    for name, d in datasets.items():
        rep = describe_dataset(d, name=name, preview=0, show=False)
        m = rep["meta"]
        rows.append(
            {
                "dataset": name,
                "rows": m["rows"],
                "cols": m["cols"],
                "numeric": m["numeric"],
                "categorical": m["categorical"],
                "missing_%": m["missing_cells_%"],
                "dup_rows": m["duplicate_rows"],
                "clean_cols": m["clean_columns"],
                "issues": len(rep["warnings"]),
                "memory": m["memory"],
            }
        )
    out = pd.DataFrame(rows).set_index("dataset")
    if show:
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(_hdr(f"COMPARING {len(out)} DATASETS"))
            print(out.to_string())
    return out
