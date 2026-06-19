"""
Rule-based building archetype classifier that mirrors the original
archetype_identification.py logic.

Classification uses three independent rule axes:
  ERA          ← yearOfConstruction  (yearOfConstructionMin as fallback)
  SIZE         ← finishAreaSqft      (sizeSqft → numOfBedrooms as fallbacks)
  CONSTRUCTION ← constructionType

The resulting archetype_label is "ERA | SIZE | CONSTRUCTION".
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ERA classification constants
# (update these to match the original script if they differ)
# ---------------------------------------------------------------------------
ERA_BINS = [0, 1900, 1940, 1960, 1979, 1999, 2100]
ERA_LABELS = [
    "Pre-1900",
    "1900-1940",
    "1941-1960",
    "1961-1979",
    "1980-1999",
    "2000+",
]

# ---------------------------------------------------------------------------
# SIZE thresholds (sq ft)
# ---------------------------------------------------------------------------
SIZE_SMALL  =  900   # < 900 sqft  → Small
SIZE_MEDIUM = 1800   # 900–1800    → Medium  |  > 1800 → Large


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numeric_series(df: pd.DataFrame, *cols: str) -> pd.Series:
    """
    Try each column name in order; return the first that yields at least one
    non-NaN numeric value after coercion, otherwise return an all-NaN series.
    """
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index)


def _fix_year(y: float) -> float:
    """
    Coerce 2-digit years (e.g. 78 → 1978) and discard implausible values.
    Anything outside 0-99 or 1800-2025 becomes NaN.
    """
    if pd.isna(y):
        return np.nan
    if 1800 <= y <= 2025:
        return y
    if 0 <= y <= 99:
        return 1900.0 + y   # 78 → 1978, 05 → 1905
    return np.nan           # 1030, 9999, etc.


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------
def classify_size(row: pd.Series) -> str:
    """
    Return 'Small', 'Medium', or 'Large' from the pre-computed _area column.
    Falls back to numOfBedrooms when area is unavailable.
    """
    area = row.get("_area")
    if pd.isna(area):
        beds = row.get("numOfBedrooms", np.nan)
        if not pd.isna(beds):
            if beds <= 2:
                return "Small"
            if beds <= 3:
                return "Medium"
            return "Large"
        return "Unknown"
    if area < SIZE_SMALL:
        return "Small"
    if area <= SIZE_MEDIUM:
        return "Medium"
    return "Large"


def classify_construction(value) -> str:
    """
    Map raw constructionType strings to three broad groups:
    Masonry  |  Wood Frame  |  Other
    """
    if pd.isna(value):
        return "Unknown"
    v = str(value).lower().strip()
    if any(k in v for k in ("masonry", "brick", "concrete")):
        return "Masonry"
    if any(k in v for k in ("wood", "frame", "pole")):
        return "Wood Frame"
    return "Other"


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------
def run_old_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based archetype classification to a DataFrame.

    Returns a DataFrame with columns:
        no.  |  era  |  size_category  |  construction_group  |  archetype_label
    """
    df = df.copy()

    # ── ERA ──────────────────────────────────────────────────────────────────
    yr = _to_numeric_series(df, "yearOfConstruction", "yearOfConstructionMin")
    yr = yr.apply(_fix_year)
    df["era"] = pd.cut(yr, bins=ERA_BINS, labels=ERA_LABELS, right=True)

    # ── SIZE ─────────────────────────────────────────────────────────────────
    area = _to_numeric_series(df, "finishAreaSqft", "sizeSqft")
    df["_area"] = area
    df["size_category"] = df.apply(classify_size, axis=1)
    df.drop(columns=["_area"], inplace=True)

    # ── CONSTRUCTION ─────────────────────────────────────────────────────────
    if "constructionType" in df.columns:
        df["construction_group"] = df["constructionType"].apply(classify_construction)
    else:
        df["construction_group"] = "Unknown"

    # ── ARCHETYPE LABEL ───────────────────────────────────────────────────────
    df["archetype_label"] = (
        df["era"].astype(str) + " | " +
        df["size_category"]   + " | " +
        df["construction_group"]
    )

    id_col = "no." if "no." in df.columns else df.columns[0]
    return df[[id_col, "era", "size_category", "construction_group", "archetype_label"]]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "community_data.csv"
    df = pd.read_csv(path)
    result = run_old_classification(df)
    print(result.to_string())
    print(f"\nArchetype distribution ({result['archetype_label'].nunique()} unique):")
    print(result["archetype_label"].value_counts().to_string())
