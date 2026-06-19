"""
Dataset validation module.
Checks a CSV or Excel building file before Agent 1 processes it.
"""

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema: columns Agent 1 expects
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {
    "yearOfConstructionMin": "Building era classification",
    "constructionType":      "Construction material type",
    "finishAreaSqft":        "Building size (primary)",
    "numOfBedrooms":         "Building size (fallback)",
    "yearOfConstruction":    "Alternative year column",
    "propertyClass":         "Property type",
    "sizeSqft":              "Total size in sqft",
    "wallType":              "Wall construction type",
    "roofing":               "Roof material type",
    "numOfFloors":           "Number of floors",
    "occupancy":             "Building occupancy type",
}

YEAR_COLUMNS  = {"yearOfConstructionMin", "yearOfConstruction"}
SIZE_COLUMNS  = {"finishAreaSqft", "sizeSqft", "numOfBedrooms"}
CONST_COLUMNS = {"constructionType"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(name: str) -> str:
    """Lowercase + strip all non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_similar(required_col: str, available_cols: list) -> str | None:
    """
    Return the best-matching available column name, or None.
    Matches on normalised substring overlap (min 4 chars to avoid false hits).
    """
    norm_req = _normalize(required_col)
    best = None
    best_score = 0
    for col in available_cols:
        norm_col = _normalize(col)
        if norm_col == norm_req:
            return col
        if len(norm_req) >= 4 and len(norm_col) >= 4:
            if norm_req in norm_col or norm_col in norm_req:
                score = len(set(norm_req) & set(norm_col))
                if score > best_score:
                    best_score = score
                    best = col
    return best


def _valid_numeric_count(series: pd.Series, lo=None, hi=None) -> int:
    """Count values that coerce to numeric and optionally fall within [lo, hi]."""
    nums = pd.to_numeric(series, errors="coerce")
    if lo is not None and hi is not None:
        return int(nums.between(lo, hi).sum())
    return int(nums.notna().sum())


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------
def validate_dataset(filepath: str) -> tuple[bool, dict, list]:
    """
    Validate a building CSV/Excel dataset before Agent 1 runs on it.

    Returns
    -------
    passed         – True if the dataset meets minimum requirements
    mapped_columns – dict mapping required column name → actual column name
    issues         – list of warning/error strings for the caller
    """
    issues: list[str] = []
    mapped_columns: dict[str, str] = {}

    # ── 1. Load file ─────────────────────────────────────────────────────────
    print("\n=== DATASET VALIDATION REPORT ===")
    print(f"File: {os.path.basename(filepath)}")

    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as exc:
        msg = f"FATAL: Cannot read file — {exc}"
        issues.append(msg)
        print(f"❌ {msg}")
        return False, {}, issues

    available_cols = list(df.columns)
    total = len(df)
    print(f"Total buildings: {total}\n")

    # ── 2. Minimum building count ─────────────────────────────────────────────
    if total < 10:
        msg = f"Only {total} buildings — results may be unreliable (recommend ≥ 10)"
        issues.append(f"WARNING: {msg}")
        print(f"⚠️  {msg}\n")

    # ── 3. Column presence and mapping ───────────────────────────────────────
    print("REQUIRED COLUMNS:")
    missing_critical: list[str] = []

    for req_col, description in REQUIRED_COLUMNS.items():
        if req_col in available_cols:
            mapped_columns[req_col] = req_col
            missing_pct = df[req_col].isna().mean() * 100
            tag = "⚠️ " if missing_pct > 50 else "✅"
            print(f"  {tag} {req_col:<28} - found ({missing_pct:.1f}% missing)")
            if missing_pct > 50:
                issues.append(f"WARNING: {req_col} has {missing_pct:.0f}% missing values")
        else:
            similar = _find_similar(req_col, available_cols)
            if similar:
                mapped_columns[req_col] = similar
                missing_pct = df[similar].isna().mean() * 100
                issues.append(f"MAPPED: {req_col} → {similar}")
                print(f"  ⚠️  {req_col:<28} → mapped to '{similar}' "
                      f"({missing_pct:.1f}% missing)")
            else:
                issues.append(f"MISSING: {req_col} — {description}")
                missing_critical.append(req_col)
                print(f"  ❌ {req_col:<28} - NOT FOUND")

    # ── 4. Data quality ───────────────────────────────────────────────────────
    print("\nDATA QUALITY:")

    # Year
    yr_col = mapped_columns.get("yearOfConstruction") or \
             mapped_columns.get("yearOfConstructionMin")
    if yr_col:
        valid_yr = _valid_numeric_count(df[yr_col], lo=1800, hi=2025)
        tag = "✅" if valid_yr >= total * 0.8 else "⚠️ "
        print(f"  {tag} Year data:      {valid_yr}/{total} buildings have valid years")
        if valid_yr < total * 0.5:
            issues.append(f"WARNING: fewer than 50% of {yr_col} values are valid years")
    else:
        print("  ❌ Year data:      no year column found")
        issues.append("MISSING: no year column available")

    # Size
    sz_col = (mapped_columns.get("finishAreaSqft") or
              mapped_columns.get("sizeSqft") or
              mapped_columns.get("numOfBedrooms"))
    if sz_col:
        valid_sz = _valid_numeric_count(df[sz_col])
        tag = "✅" if valid_sz >= total * 0.8 else "⚠️ "
        print(f"  {tag} Size data:      {valid_sz}/{total} buildings have size info")
    else:
        print("  ❌ Size data:      no size column found")
        issues.append("MISSING: no size column available")

    # Construction
    ct_col = mapped_columns.get("constructionType")
    if ct_col:
        valid_ct = int(df[ct_col].notna().sum())
        tag = "✅" if valid_ct >= total * 0.8 else "⚠️ "
        print(f"  {tag} Construction:   {valid_ct}/{total} buildings have construction type")
    else:
        print("  ❌ Construction:   no construction type column found")
        issues.append("MISSING: no construction type column available")

    # ── 5. Archetype readiness ────────────────────────────────────────────────
    print("\nARCHETYPE READINESS:")
    has_era   = any(mapped_columns.get(c) for c in YEAR_COLUMNS)
    has_size  = any(mapped_columns.get(c) for c in SIZE_COLUMNS)
    has_const = any(mapped_columns.get(c) for c in CONST_COLUMNS)

    print(f"  {'✅' if has_era   else '❌'} ERA classification:          "
          f"{'READY' if has_era   else 'MISSING — need a year column'}")
    print(f"  {'✅' if has_size  else '❌'} SIZE classification:         "
          f"{'READY' if has_size  else 'MISSING — need finishAreaSqft or sizeSqft'}")
    print(f"  {'✅' if has_const else '❌'} CONSTRUCTION classification: "
          f"{'READY' if has_const else 'MISSING — need constructionType'}")

    # ── 6. Pass / fail ────────────────────────────────────────────────────────
    # Fatal only when a column needed for readiness has no mapping at all
    fatal_missing = [c for c in missing_critical
                     if c in YEAR_COLUMNS | SIZE_COLUMNS | CONST_COLUMNS]
    passed = has_era and has_size and has_const and len(fatal_missing) == 0

    print(f"\n{'─' * 46}")
    if passed:
        print("VALIDATION RESULT: PASSED ✅")
        print("Dataset is ready for archetype identification.")
    else:
        print("VALIDATION RESULT: FAILED ❌")
        print("Resolve the issues above before running Agent 1.")
    print(f"{'─' * 46}\n")

    return passed, mapped_columns, issues


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "community_data.csv"
    passed, mapped, issues = validate_dataset(path)
    if issues:
        print("Issues log:")
        for issue in issues:
            print(f"  • {issue}")
