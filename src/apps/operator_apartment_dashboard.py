from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ======================================
# Project path setup
# ======================================
def resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd().resolve()]

    for cand in candidates:
        data_ok = (cand / "data" / "processed" / "stage2_recommender_input.csv").exists()
        model_ok = (cand / "src" / "models" / "personality_engine.py").exists()
        if data_ok and model_ok:
            return cand

    # Fallback for the expected repo layout when the file sits in src/apps
    try:
        return here.parents[2]
    except IndexError:
        return here.parent


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.personality_engine import (  # noqa: E402
    build_personality_matrices,
    upper_triangle_values,
)


# ======================================
# App config
# ======================================
st.set_page_config(
    page_title="Monitor current flat compatibility",
    page_icon="🏠",
    layout="wide",
)


# ======================================
# Constants
# ======================================
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stage2_recommender_input.csv"

LIFESTYLE_FEATURES = [
    "sleep_schedule_num",
    "noise_sensitivity_num",
    "vibe_num",
    "cleanliness_num",
    "guests_over_num",
    "alcohol_num",
    "smoking_num",
    "cooking_at_home_num",
    "chores_num",
]

DETAIL_COLS = [
    "unit",
    "age",
    "gender",
    "nationality",
    "occupation",
    "remainingstay_days",
]

RISK_COLOR = "#8c2d1f"
STRONG_COLOR = "#2a6a39"
NEUTRAL_COLOR = "#6f665d"

STRONG_OVERALL_DEFAULT = 84
STRONG_WEAK_LINK_DEFAULT = 82
STRONG_FAIRNESS_DEFAULT = 84


# ======================================
# Styling
# ======================================
st.markdown(
    """
    <style>
        :root {
            --page-bg: #f3f0ea;
            --card-bg: #faf7f2;
            --border: #e4d8cd;
            --text: #1f1f1f;
            --muted: #6a6762;
            --accent: #1099b6;
            --accent-strong: #0d7f96;
            --accent-soft: #e5f5f8;
            --line: #efcbd2;
            --pill-bg: #e5f5f8;
            --pill-text: #0f5f71;
            --risk-bg: #fff1ef;
            --risk-border: #e8b4ae;
            --risk-pill: #8c2d1f;
            --strong-bg: #f1f8f0;
            --strong-border: #bfd7b9;
            --strong-pill: #2a6a39;
            --neutral-bg: #faf7f2;
            --neutral-border: #e4d8cd;
            --neutral-pill: #6f665d;
            --insufficient-bg: #f4f1ec;
            --insufficient-border: #d8d1c7;
            --insufficient-pill: #7e756c;
            --button-text: #ffffff;
        }

        .stApp {
            background: var(--page-bg);
        }

        [data-testid="stHeader"] {
            background: rgba(243, 240, 234, 0.92);
        }

        [data-testid="stSidebar"] {
            background: #f7f4ee;
            border-right: 1px solid var(--border);
            min-width: 430px;
            max-width: 430px;
        }

        section[data-testid="stSidebar"] {
            width: 430px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 430px !important;
        }

        .block-container {
            padding-top: 3.1rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        h1, h2, h3, h4, h5, h6,
        p, div, span, label {
            color: var(--text);
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: var(--text);
        }

        .hero-wrap {
            padding: 0.55rem 0 0.55rem 0;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: clamp(2.45rem, 5vw, 4.8rem);
            line-height: 1.0;
            letter-spacing: -0.035em;
            font-weight: 500;
            color: var(--accent);
            margin: 0.15rem 0 0 0;
            max-width: none;
            white-space: nowrap;
        }

        .hero-kicker {
            margin-top: 1.4rem;
            color: var(--accent-strong);
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        }

        .hero-divider {
            height: 2px;
            width: 100%;
            background: var(--line);
            margin: 1.5rem 0 1rem 0;
            border-radius: 999px;
        }

        .intro-copy {
            max-width: 940px;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.65;
            margin: 0.1rem 0 1.2rem 0;
        }

        .section-title {
            color: var(--text);
            font-size: clamp(1.9rem, 3.2vw, 2.7rem);
            font-weight: 800;
            line-height: 1.08;
            margin: 1.5rem 0 0.3rem 0;
        }

        .section-copy {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.55;
            margin: 0 0 0.95rem 0;
        }

        .status-card {
            border-radius: 24px;
            padding: 1.3rem 1.35rem 1.15rem 1.35rem;
            margin: 0.8rem 0 0.7rem 0;
            border: 1px solid var(--border);
            box-shadow: 0 1px 0 rgba(0,0,0,0.01);
        }

        .status-risk {
            background: var(--risk-bg);
            border-color: var(--risk-border);
        }

        .status-strong {
            background: var(--strong-bg);
            border-color: var(--strong-border);
        }

        .status-neutral {
            background: var(--neutral-bg);
            border-color: var(--neutral-border);
        }

        .status-insufficient {
            background: var(--insufficient-bg);
            border-color: var(--insufficient-border);
        }

        .status-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.9fr) minmax(165px, 0.7fr) auto;
            gap: 1rem;
            align-items: start;
        }

        .status-title {
            font-size: 1.95rem;
            line-height: 1.14;
            font-weight: 750;
            margin: 0;
            color: var(--text);
        }

        .status-subcopy {
            color: var(--muted);
            margin-top: 0.5rem;
            font-size: 0.98rem;
            line-height: 1.5;
        }

        .status-meta {
            margin-top: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem 1rem;
            align-items: center;
        }

        .score-label {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 650;
            margin-bottom: 0.08rem;
        }

        .score-value {
            font-size: 2.95rem;
            line-height: 1;
            font-weight: 520;
            color: var(--text);
        }

        .status-pill {
            display: inline-block;
            padding: 0.28rem 0.78rem;
            border-radius: 999px;
            font-weight: 750;
            font-size: 0.88rem;
            border: 1px solid transparent;
        }

        .pill-risk {
            color: white;
            background: var(--risk-pill);
            border-color: var(--risk-pill);
        }

        .pill-strong {
            color: white;
            background: var(--strong-pill);
            border-color: var(--strong-pill);
        }

        .pill-neutral {
            color: white;
            background: var(--neutral-pill);
            border-color: var(--neutral-pill);
        }

        .pill-insufficient {
            color: white;
            background: var(--insufficient-pill);
            border-color: var(--insufficient-pill);
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(5, minmax(110px, 1fr));
            gap: 0.65rem;
            margin-top: 1.1rem;
        }

        .metric-box {
            background: rgba(255,255,255,0.55);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 16px;
            padding: 0.72rem 0.8rem 0.68rem 0.8rem;
        }

        .metric-box-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.25;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.22rem;
        }

        .metric-box-value {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 750;
            line-height: 1.1;
        }

        .occupancy-wrap {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .occupancy-label {
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 600;
        }

        .occupancy-icons {
            display: inline-flex;
            gap: 0.18rem;
            align-items: center;
        }

        .person-icon {
            width: 16px;
            height: 20px;
            display: inline-flex;
        }

        .person-icon svg {
            width: 100%;
            height: 100%;
        }

        .occupied-icon svg {
            fill: #2c6f7f;
        }

        .free-icon svg {
            fill: #d7dfe3;
        }

        .mini-tag {
            display: inline-block;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.65);
            border: 1px solid rgba(0,0,0,0.06);
            color: var(--text);
            font-size: 0.84rem;
            font-weight: 650;
        }

        .note-list {
            margin: 1rem 0 0 0;
            padding-left: 1.1rem;
        }

        .note-list li {
            color: var(--text);
            line-height: 1.58;
            font-size: 0.98rem;
            margin: 0.16rem 0;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button,
        button[kind="primary"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        .stFormSubmitButton > button *,
        button[kind="primary"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover,
        button[kind="primary"]:hover {
            background: var(--accent-strong) !important;
            border-color: var(--accent-strong) !important;
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--card-bg);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }


        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stNumberInputRootElement"] > div {
            background: #fffdfa !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-testid="stTextInputRootElement"] input,
        [data-testid="stNumberInputRootElement"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }

        [data-baseweb="select"] svg,
        [data-baseweb="input"] svg {
            fill: var(--text) !important;
        }

        div[role="listbox"] {
            background: #fffdfa !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }

        div[role="option"] {
            background: #fffdfa !important;
            color: var(--text) !important;
        }

        div[role="option"]:hover {
            background: #f3ece2 !important;
        }

        @media (max-width: 900px) {
            .status-grid,
            .metric-row {
                grid-template-columns: 1fr;
            }

            .score-value {
                font-size: 2.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================
# Data loading
# ======================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "apartmentsize",
        "age",
        "remainingstay_days",
        *LIFESTYLE_FEATURES,
        "compatibility_overallrating_num",
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "emotional_stability",
        "openness",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_text(value, default: str = "Not specified") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    text = text.replace("_", " ")
    return text[0].upper() + text[1:]


def score_to_100(value: float) -> int | None:
    if pd.isna(value):
        return None
    return int(round(float(value) * 100))


def extract_unit_number(unit_value: str) -> int | None:
    if pd.isna(unit_value):
        return None
    try:
        return int(str(unit_value).split(".")[-1])
    except (ValueError, TypeError):
        return None


# ======================================
# Reused compatibility logic
# ======================================
def build_feature_ranges(df_in: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}

    for col in feature_cols:
        col_min = float(df_in[col].min())
        col_max = float(df_in[col].max())

        if pd.isna(col_min) or pd.isna(col_max) or col_max <= col_min:
            raise ValueError(f"Feature '{col}' has no usable range.")

        ranges[col] = {
            "min": col_min,
            "max": col_max,
            "span": col_max - col_min,
        }

    return ranges


def lifestyle_similarity_component(a: float, b: float, span: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(np.clip(1.0 - (abs(a - b) / span), 0.0, 1.0))


def lifestyle_compatibility_between_rows(
    row_i: pd.Series,
    row_j: pd.Series,
    *,
    feature_cols: list[str],
    feature_ranges: dict,
    weights: dict[str, float] | None = None,
) -> float:
    if weights is None:
        weights = {f: 1.0 for f in feature_cols}

    num = 0.0
    den = 0.0

    for f in feature_cols:
        w = float(weights.get(f, 1.0))
        span = float(feature_ranges[f]["span"])
        s = lifestyle_similarity_component(row_i[f], row_j[f], span=span)

        if pd.notna(s):
            num += w * s
            den += w

    return num / den if den else np.nan


def lifestyle_compatibility_matrix(
    df_in: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_ranges: dict,
    weights: dict[str, float] | None = None,
    id_col: str = "unit",
) -> pd.DataFrame:
    work_df = df_in.copy()

    if id_col in work_df.columns:
        if work_df[id_col].duplicated().any():
            raise ValueError(f"Column '{id_col}' contains duplicate IDs.")
        work_df = work_df.set_index(id_col, drop=False)

    n = len(work_df)
    M = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            s = lifestyle_compatibility_between_rows(
                work_df.iloc[i],
                work_df.iloc[j],
                feature_cols=feature_cols,
                feature_ranges=feature_ranges,
                weights=weights,
            )
            if pd.isna(s):
                s = 0.0
            M[i, j] = s
            M[j, i] = s

    return pd.DataFrame(M, index=work_df.index, columns=work_df.index)


@st.cache_data(show_spinner=False)
def build_personality_matrices_cached(df_in: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return build_personality_matrices(df_in=df_in, id_col="unit")


@st.cache_data(show_spinner=False)
def build_lifestyle_matrix_cached(df_in: pd.DataFrame) -> pd.DataFrame:
    feature_ranges = build_feature_ranges(df_in, LIFESTYLE_FEATURES)
    weights = {f: 1.0 for f in LIFESTYLE_FEATURES}
    return lifestyle_compatibility_matrix(
        df_in=df_in,
        feature_cols=LIFESTYLE_FEATURES,
        feature_ranges=feature_ranges,
        weights=weights,
        id_col="unit",
    )


def build_final_pairwise_matrix(
    C_personality: pd.DataFrame,
    C_lifestyle: pd.DataFrame,
    personality_weight: float,
    lifestyle_weight: float,
) -> pd.DataFrame:
    return personality_weight * C_personality + lifestyle_weight * C_lifestyle


def coalition_member_utilities(submatrix: pd.DataFrame) -> pd.Series:
    utilities = {}
    for member in submatrix.index:
        others = submatrix.columns[submatrix.columns != member]
        utilities[member] = submatrix.loc[member, others].mean() if len(others) else np.nan
    return pd.Series(utilities)


def coalition_metrics(submatrix: pd.DataFrame) -> dict[str, float]:
    utilities = coalition_member_utilities(submatrix)
    pair_values = upper_triangle_values(submatrix)

    return {
        "harmony": float(utilities.mean()),
        "fairness": float(utilities.min()),
        "min_pair": float(pair_values.min()) if len(pair_values) > 0 else np.nan,
        "mean_pair": float(pair_values.mean()) if len(pair_values) > 0 else np.nan,
        "max_pair": float(pair_values.max()) if len(pair_values) > 0 else np.nan,
        "utility_std": float(utilities.std()) if len(utilities) > 1 else np.nan,
    }


def add_structural_score(
    df_in: pd.DataFrame,
    w_fairness: float = 0.45,
    w_harmony: float = 0.35,
    w_min_pair: float = 0.20,
) -> pd.DataFrame:
    out = df_in.copy()
    out["struct_score"] = (
        w_fairness * out["fairness"]
        + w_harmony * out["harmony"]
        + w_min_pair * out["min_pair"]
    )
    return out


# ======================================
# Apartment-level operator dashboard logic
# ======================================
def normalize_flag_value(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    return x if x else None


def homogeneous_group_info(values, min_residents: int = 2) -> tuple[bool, str | None]:
    vals = [normalize_flag_value(v) for v in values]
    vals = [v for v in vals if v is not None]

    if len(vals) < min_residents:
        return False, None

    uniq = set(vals)
    if len(uniq) == 1:
        return True, next(iter(uniq))

    return False, None


def compute_available_rooms(apartment: str, apartment_size: float, units: list[str]) -> list[str]:
    if pd.isna(apartment_size):
        return []

    total = int(round(float(apartment_size)))
    occupied_numbers = [
        n for n in (extract_unit_number(u) for u in units)
        if n is not None
    ]
    missing_numbers = sorted(set(range(1, total + 1)) - set(occupied_numbers))
    return [f"{apartment}.{num}" for num in missing_numbers]


def build_pair_breakdown(submatrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    members = list(submatrix.index)

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            u1 = members[i]
            u2 = members[j]
            rows.append(
                {
                    "Resident A": u1,
                    "Resident B": u2,
                    "Pair fit": float(submatrix.loc[u1, u2]),
                    "Pair fit (0-100)": score_to_100(float(submatrix.loc[u1, u2])),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Resident A", "Resident B", "Pair fit", "Pair fit (0-100)"])

    return pd.DataFrame(rows).sort_values(by=["Pair fit", "Resident A", "Resident B"]).reset_index(drop=True)


def build_resident_snapshot(residents_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DETAIL_COLS if c in residents_df.columns]
    out = residents_df[cols].copy()
    rename_map = {
        "unit": "Resident ID",
        "age": "Age",
        "gender": "Gender",
        "nationality": "Nationality",
        "occupation": "Occupation",
        "remainingstay_days": "Remaining stay (days)",
    }
    out = out.rename(columns=rename_map)
    return out.reset_index(drop=True)


def build_apartment_dashboard(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, float]]:
    personality_matrices = build_personality_matrices_cached(df_in)
    C_lifestyle = build_lifestyle_matrix_cached(df_in)

    # Fixed, non-personalized view aligned with Section 4.1 logic.
    C_personality = personality_matrices["equal"]
    C_combined = build_final_pairwise_matrix(
        C_personality=C_personality,
        C_lifestyle=C_lifestyle,
        personality_weight=0.5,
        lifestyle_weight=0.5,
    )

    rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}

    for apt, group in df_in.groupby("apartment"):
        residents_df = group.copy().sort_values("unit")
        resident_units = residents_df["unit"].dropna().astype(str).unique().tolist()
        resident_units = [u for u in resident_units if u in C_combined.index]

        if not resident_units:
            continue

        apartment_size = residents_df["apartmentsize"].iloc[0]
        n_residents = len(resident_units)
        total_rooms = int(round(float(apartment_size))) if pd.notna(apartment_size) else np.nan
        empty_rooms = max(total_rooms - n_residents, 0) if pd.notna(total_rooms) else np.nan
        available_rooms = compute_available_rooms(apt, apartment_size, resident_units)

        utilities = pd.Series(dtype=float)
        pair_df = pd.DataFrame(columns=["Resident A", "Resident B", "Pair fit", "Pair fit (0-100)"])
        matrix_df = pd.DataFrame(index=resident_units, columns=resident_units, dtype=float)

        if n_residents >= 2:
            submatrix = C_combined.loc[resident_units, resident_units]
            metrics = coalition_metrics(submatrix)
            utilities = coalition_member_utilities(submatrix).sort_values()
            pair_df = build_pair_breakdown(submatrix)
            matrix_df = (submatrix * 100).round(1)
        else:
            metrics = {
                "harmony": np.nan,
                "fairness": np.nan,
                "min_pair": np.nan,
                "mean_pair": np.nan,
                "max_pair": np.nan,
                "utility_std": np.nan,
            }

        flag_gender_homogeneous, gender_group = homogeneous_group_info(residents_df.get("gender", pd.Series(dtype=object)).tolist())
        flag_nationality_homogeneous, nationality_group = homogeneous_group_info(residents_df.get("nationality", pd.Series(dtype=object)).tolist())
        flag_occupation_homogeneous, occupation_group = homogeneous_group_info(residents_df.get("occupation", pd.Series(dtype=object)).tolist())

        age_series = pd.to_numeric(residents_df.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
        age_range = float(age_series.max() - age_series.min()) if len(age_series) >= 2 else np.nan

        remaining_series = pd.to_numeric(
            residents_df.get("remainingstay_days", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        turnover_share_lt_40d = float((remaining_series < 40).mean()) if len(remaining_series) > 0 else np.nan
        flag_high_turnover = bool(pd.notna(turnover_share_lt_40d) and turnover_share_lt_40d >= 0.50)

        row = {
            "apartment": apt,
            "apartmentsize": apartment_size,
            "n_residents": n_residents,
            "empty_rooms": empty_rooms,
            "available_rooms": " | ".join(available_rooms) if available_rooms else "",
            "resident_units": " | ".join(resident_units),
            "harmony": metrics["harmony"],
            "fairness": metrics["fairness"],
            "min_pair": metrics["min_pair"],
            "mean_pair": metrics["mean_pair"],
            "max_pair": metrics["max_pair"],
            "utility_std": metrics["utility_std"],
            "age_range": age_range,
            "turnover_share_lt_40d": turnover_share_lt_40d,
            "flag_high_turnover": flag_high_turnover,
            "flag_gender_homogeneous": flag_gender_homogeneous,
            "gender_group": gender_group or "",
            "flag_nationality_homogeneous": flag_nationality_homogeneous,
            "nationality_group": nationality_group or "",
            "flag_occupation_homogeneous": flag_occupation_homogeneous,
            "occupation_group": occupation_group or "",
        }
        rows.append(row)

        details[apt] = {
            "residents_df": build_resident_snapshot(residents_df.loc[residents_df["unit"].isin(resident_units)].copy()),
            "utilities_df": pd.DataFrame(
                {
                    "Resident ID": utilities.index.tolist(),
                    "Average fit to flat": utilities.round(3).tolist(),
                    "Average fit to flat (0-100)": [score_to_100(v) for v in utilities.tolist()],
                }
            ) if not utilities.empty else pd.DataFrame(columns=["Resident ID", "Average fit to flat", "Average fit to flat (0-100)"]),
            "pair_df": pair_df,
            "matrix_df": matrix_df,
        }

    apartment_df = pd.DataFrame(rows)
    apartment_df = add_structural_score(apartment_df)
    apartment_df = apartment_df.sort_values(
        by=["struct_score", "fairness", "harmony"],
        ascending=False,
    ).reset_index(drop=True)

    meta = {
        "personality_weight": 0.5,
        "lifestyle_weight": 0.5,
    }

    return apartment_df, details, meta


# ======================================
# Card helpers
# ======================================
def classify_apartment(
    row: pd.Series,
    strong_threshold: int,
    risk_threshold: int,
    weak_link_threshold: int,
) -> tuple[str, str]:
    if int(row.get("n_residents", 0)) < 2 or pd.isna(row.get("struct_score")):
        return "Insufficient data", "insufficient"

    overall = score_to_100(row.get("struct_score")) or 0
    min_pair = score_to_100(row.get("min_pair")) or 0
    fairness = score_to_100(row.get("fairness")) or 0

    if min_pair < 30:
        return "Critical weak link", "risk"

    if overall <= risk_threshold or min_pair <= weak_link_threshold:
        return "Potential issue", "risk"

    if overall >= strong_threshold and min_pair >= STRONG_WEAK_LINK_DEFAULT and fairness >= STRONG_FAIRNESS_DEFAULT:
        return "Very strong", "strong"

    return "Stable", "neutral"


def tone_css_class(tone: str) -> str:
    return {
        "risk": "status-risk",
        "strong": "status-strong",
        "neutral": "status-neutral",
        "insufficient": "status-insufficient",
    }.get(tone, "status-neutral")


def pill_css_class(tone: str) -> str:
    return {
        "risk": "pill-risk",
        "strong": "pill-strong",
        "neutral": "pill-neutral",
        "insufficient": "pill-insufficient",
    }.get(tone, "pill-neutral")


def build_person_icon_html(occupied: bool) -> str:
    icon_class = "occupied-icon" if occupied else "free-icon"
    return (
        f'<span class="person-icon {icon_class}">'
        '<svg viewBox="0 0 14 18" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<circle cx="7" cy="3" r="2.5"></circle>'
        '<rect x="5.4" y="6" width="3.2" height="6.1" rx="1.4"></rect>'
        '<rect x="2.6" y="7" width="2" height="5.2" rx="0.95"></rect>'
        '<rect x="9.4" y="7" width="2" height="5.2" rx="0.95"></rect>'
        '<rect x="4.5" y="11.3" width="2.1" height="6.1" rx="0.95"></rect>'
        '<rect x="7.4" y="11.3" width="2.1" height="6.1" rx="0.95"></rect>'
        '</svg>'
        '</span>'
    )


def build_occupancy_html(row: pd.Series) -> str:
    apartment_size = row.get("apartmentsize", np.nan)
    n_residents = row.get("n_residents", np.nan)

    if pd.isna(apartment_size) or pd.isna(n_residents):
        return ""

    total_slots = max(int(round(float(apartment_size))), 0)
    occupied_slots = min(max(int(round(float(n_residents))), 0), total_slots)
    free_slots = max(total_slots - occupied_slots, 0)

    icons = "".join(build_person_icon_html(True) for _ in range(occupied_slots))
    icons += "".join(build_person_icon_html(False) for _ in range(free_slots))

    room_word = "room" if total_slots == 1 else "rooms"
    label = f"{occupied_slots} of {total_slots} {room_word} currently occupied"

    return (
        '<div class="occupancy-wrap">'
        f'<span class="occupancy-label">{escape(label)}</span>'
        f'<span class="occupancy-icons">{icons}</span>'
        '</div>'
    )


def build_note_list(row: pd.Series) -> list[str]:
    notes: list[str] = []

    overall = score_to_100(row.get("struct_score"))
    fairness = score_to_100(row.get("fairness"))
    harmony = score_to_100(row.get("harmony"))
    min_pair = score_to_100(row.get("min_pair"))
    mean_pair = score_to_100(row.get("mean_pair"))

    if pd.notna(row.get("struct_score")):
        if overall is not None and overall >= 85:
            notes.append("Overall composition looks very strong on the combined compatibility score.")
        elif overall is not None and overall <= 65:
            notes.append("Overall composition falls into a weaker zone and may deserve closer attention.")

    if min_pair is not None:
        if min_pair < 30:
            notes.append("A critically weak resident pairing is present despite the broader apartment average.")
        elif min_pair <= 55:
            notes.append("The weakest direct resident pairing is notably low.")
        elif min_pair >= 78:
            notes.append("There is no obvious weak pair inside the current flat.")

    if fairness is not None and fairness <= 65:
        notes.append("At least one resident looks materially worse off than the others on average.")
    elif fairness is not None and fairness >= 80:
        notes.append("Compatibility looks relatively balanced across residents.")

    if harmony is not None and harmony >= 82:
        notes.append("The overall group dynamic appears strong.")
    elif harmony is not None and harmony <= 68:
        notes.append("Average within-flat compatibility is only moderate to weak.")

    if mean_pair is not None and mean_pair >= 80:
        notes.append("Average pairwise fit is strong across the flat.")

    if pd.notna(row.get("empty_rooms")) and int(row["empty_rooms"]) > 0:
        rooms = int(row["empty_rooms"])
        notes.append(f"{rooms} room{'s' if rooms != 1 else ''} currently empty.")

    if row.get("flag_high_turnover", False):
        share = row.get("turnover_share_lt_40d")
        if pd.notna(share):
            notes.append(f"High near-term turnover risk: about {int(round(float(share) * 100))}% of residents have fewer than 40 days remaining.")
        else:
            notes.append("High near-term turnover risk among current residents.")

    age_range = row.get("age_range")
    if pd.notna(age_range) and float(age_range) >= 8:
        notes.append(f"The current age range inside the flat is about {int(round(float(age_range)))} years.")

    if row.get("flag_gender_homogeneous", False) and row.get("gender_group"):
        notes.append(f"Resident gender composition is currently homogeneous ({clean_text(row['gender_group'])}).")

    if row.get("flag_nationality_homogeneous", False) and row.get("nationality_group"):
        notes.append(f"Resident nationality composition is currently homogeneous ({clean_text(row['nationality_group'])}).")

    if row.get("flag_occupation_homogeneous", False) and row.get("occupation_group"):
        notes.append(f"Resident occupation composition is currently homogeneous ({clean_text(row['occupation_group'])}).")

    if not notes:
        notes.append("No major structural or contextual risk signal was triggered by the current thresholds.")

    return notes[:6]


def metric_value_html(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{escape(str(value))}{escape(suffix)}"


def build_card_html(row: pd.Series, status_label: str, tone: str) -> str:
    overall_score = score_to_100(row.get("struct_score"))
    fairness = score_to_100(row.get("fairness"))
    harmony = score_to_100(row.get("harmony"))
    min_pair = score_to_100(row.get("min_pair"))
    mean_pair = score_to_100(row.get("mean_pair"))
    n_residents = int(row.get("n_residents", 0))
    empty_rooms = row.get("empty_rooms", np.nan)
    available_rooms = row.get("available_rooms", "")

    notes_html = "".join(f"<li>{escape(item)}</li>" for item in build_note_list(row))
    occupancy_html = build_occupancy_html(row)

    extra_tags = []
    if pd.notna(empty_rooms):
        extra_tags.append(f"<span class='mini-tag'>Empty rooms: {int(empty_rooms)}</span>")
    if available_rooms:
        extra_tags.append(f"<span class='mini-tag'>Open units: {escape(available_rooms)}</span>")

    tags_html = "".join(extra_tags)

    subtitle = "Current resident composition only · fixed non-personalized compatibility view"
    title_html = escape(f"Apartment {row['apartment']}")

    return (
        f"<div class='status-card {tone_css_class(tone)}'>"
        "<div class='status-grid'>"
        "<div>"
        f"<div class='status-title'>{title_html}</div>"
        f"<div class='status-subcopy'>{escape(subtitle)}</div>"
        f"<div class='status-meta'>{occupancy_html}{tags_html}</div>"
        "</div>"
        "<div>"
        "<div class='score-label'>Overall composition</div>"
        f"<div class='score-value'>{metric_value_html(overall_score, '/100')}</div>"
        "</div>"
        "<div>"
        f"<span class='status-pill {pill_css_class(tone)}'>{escape(status_label)}</span>"
        "</div>"
        "</div>"
        "<div class='metric-row'>"
        f"<div class='metric-box'><div class='metric-box-label'>Balanced fit</div><div class='metric-box-value'>{metric_value_html(fairness, '/100')}</div></div>"
        f"<div class='metric-box'><div class='metric-box-label'>Group fit</div><div class='metric-box-value'>{metric_value_html(harmony, '/100')}</div></div>"
        f"<div class='metric-box'><div class='metric-box-label'>Weakest link</div><div class='metric-box-value'>{metric_value_html(min_pair, '/100')}</div></div>"
        f"<div class='metric-box'><div class='metric-box-label'>Average pair</div><div class='metric-box-value'>{metric_value_html(mean_pair, '/100')}</div></div>"
        f"<div class='metric-box'><div class='metric-box-label'>Residents</div><div class='metric-box-value'>{metric_value_html(n_residents)}</div></div>"
        "</div>"
        f"<ul class='note-list'>{notes_html}</ul>"
        "</div>"
    )


def prepare_operator_table(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in.copy()

    return pd.DataFrame(
        {
            "Apartment": df_in["apartment"],
            "Overall composition": df_in["struct_score"].apply(score_to_100),
            "Balanced fit": df_in["fairness"].apply(score_to_100),
            "Group fit": df_in["harmony"].apply(score_to_100),
            "Weakest link": df_in["min_pair"].apply(score_to_100),
            "Average pair": df_in["mean_pair"].apply(score_to_100),
            "Strongest pair": df_in["max_pair"].apply(score_to_100),
            "Residents": df_in["n_residents"],
            "Empty rooms": df_in["empty_rooms"],
            "Available units": df_in["available_rooms"].replace("", "—"),
            "Utility spread": df_in["utility_std"].round(3),
            "Age range": df_in["age_range"].round(1),
            "High turnover share (<40d)": (df_in["turnover_share_lt_40d"] * 100).round(0),
        }
    )


def render_apartment_block(row: pd.Series, details: dict[str, object], status_label: str, tone: str) -> None:
    st.markdown(build_card_html(row, status_label=status_label, tone=tone), unsafe_allow_html=True)

    with st.expander(f"Open detailed diagnostics · Apartment {row['apartment']}", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Current residents**")
            st.dataframe(details["residents_df"], use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Resident utility breakdown**")
            st.dataframe(details["utilities_df"], use_container_width=True, hide_index=True)

        st.markdown("**Weakest-to-strongest pair breakdown**")
        st.dataframe(details["pair_df"], use_container_width=True, hide_index=True)

        matrix_df = details["matrix_df"]
        if isinstance(matrix_df, pd.DataFrame) and not matrix_df.empty:
            st.markdown("**Pairwise compatibility matrix (0–100)**")
            st.dataframe(matrix_df, use_container_width=True)
        else:
            st.info("At least two current residents are needed for pairwise compatibility diagnostics.")


# ======================================
# UI
# ======================================
if not DATA_PATH.exists():
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

merged_df = load_data(str(DATA_PATH))

required_cols = {
    "unit",
    "apartment",
    "apartmentsize",
    "age",
    "gender",
    "nationality",
    "occupation",
    "remainingstay_days",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "emotional_stability",
    "openness",
    *LIFESTYLE_FEATURES,
}
missing_cols = sorted(c for c in required_cols if c not in merged_df.columns)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

apartment_df, apartment_details, meta = build_apartment_dashboard(merged_df)

if apartment_df.empty:
    st.warning("No apartments with valid resident compositions were found in the input data.")
    st.stop()

with st.sidebar:
    st.header("Dashboard controls")

    with st.form("operator_dashboard_controls"):
        focus_mode = st.radio(
            "What should be surfaced first?",
            options=["All apartments", "Potential issues first", "Strongest first"],
            index=1,
        )

        show_group = st.radio(
            "Which apartments should be shown?",
            options=["All", "Potential issues only", "Strongest only", "With empty rooms", "Fully occupied only"],
            index=0,
        )

        sort_by = st.selectbox(
            "Sort apartments by",
            options=[
                "Overall composition",
                "Weakest link",
                "Balanced fit",
                "Group fit",
                "Average pair",
                "Empty rooms",
                "Resident count",
            ],
            index=1,
        )

        sort_order = st.radio(
            "Sort direction",
            options=["Descending", "Ascending"],
            index=1 if focus_mode == "Potential issues first" else 0,
        )

        max_cards = st.select_slider(
            "How many apartments should be listed?",
            options=[5, 10, 15, 20, 30],
            value=15,
        )

        min_residents_filter = st.radio(
            "Minimum residents to include",
            options=[1, 2],
            index=0,
            format_func=lambda x: "At least 1 current resident" if x == 1 else "At least 2 current residents",
        )

        risk_threshold = st.slider(
            "Highlight poor overall scores at or below",
            min_value=45,
            max_value=80,
            value=65,
            step=1,
        )

        weak_link_threshold = st.slider(
            "Highlight poor weakest-link scores at or below",
            min_value=35,
            max_value=75,
            value=55,
            step=1,
        )

        strong_threshold = st.slider(
            "Highlight strong overall scores at or above",
            min_value=70,
            max_value=95,
            value=STRONG_OVERALL_DEFAULT,
            step=1,
        )

        search_apartment = st.text_input(
            "Filter by apartment ID",
            value="",
            placeholder="e.g. PXC.04",
        )

        submitted = st.form_submit_button("Refresh dashboard")

# Make the form state explicit even though Streamlit re-runs anyway.
_ = submitted

classified_df = apartment_df.copy()
classified_df[["status_label", "status_tone"]] = classified_df.apply(
    lambda row: pd.Series(
        classify_apartment(
            row=row,
            strong_threshold=strong_threshold,
            risk_threshold=risk_threshold,
            weak_link_threshold=weak_link_threshold,
        )
    ),
    axis=1,
)

if min_residents_filter > 1:
    classified_df = classified_df.loc[classified_df["n_residents"] >= min_residents_filter].copy()

if search_apartment.strip():
    needle = search_apartment.strip().lower()
    classified_df = classified_df.loc[
        classified_df["apartment"].astype(str).str.lower().str.contains(needle, na=False)
    ].copy()

if show_group == "Potential issues only":
    classified_df = classified_df.loc[classified_df["status_tone"] == "risk"].copy()
elif show_group == "Strongest only":
    classified_df = classified_df.loc[classified_df["status_tone"] == "strong"].copy()
elif show_group == "With empty rooms":
    classified_df = classified_df.loc[classified_df["empty_rooms"] > 0].copy()
elif show_group == "Fully occupied only":
    classified_df = classified_df.loc[classified_df["empty_rooms"] <= 0].copy()

sort_map = {
    "Overall composition": "struct_score",
    "Weakest link": "min_pair",
    "Balanced fit": "fairness",
    "Group fit": "harmony",
    "Average pair": "mean_pair",
    "Empty rooms": "empty_rooms",
    "Resident count": "n_residents",
}
sort_col = sort_map[sort_by]
ascending = sort_order == "Ascending"
classified_df = classified_df.sort_values(by=[sort_col, "apartment"], ascending=[ascending, True]).reset_index(drop=True)

if max_cards > 0:
    cards_df = classified_df.head(max_cards).copy()
else:
    cards_df = classified_df.copy()

risk_count = int((apartment_df.apply(lambda row: classify_apartment(row, strong_threshold, risk_threshold, weak_link_threshold)[1], axis=1) == "risk").sum())
strong_count = int((apartment_df.apply(lambda row: classify_apartment(row, strong_threshold, risk_threshold, weak_link_threshold)[1], axis=1) == "strong").sum())
scored_df = apartment_df.loc[apartment_df["n_residents"] >= 2].copy()
avg_score = scored_df["struct_score"].mean() if not scored_df.empty else np.nan
avg_empty_rooms = apartment_df["empty_rooms"].mean() if "empty_rooms" in apartment_df.columns else np.nan

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-kicker">Operator dashboard</div>
        <div class="hero-title">Monitor current flatmate compatibility</div>
        <div class="hero-divider"></div>
        <div class="intro-copy">
            This view scores the <strong>current apartment compositions only</strong> using the fixed,
            non-personalized combined compatibility setup. It is meant to surface potentially fragile
            flats early, highlight especially strong compositions, and keep empty-room visibility in the
            same operator view.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Apartments monitored", f"{int(apartment_df['apartment'].nunique())}")
m2.metric("Potential issues", f"{risk_count}")
m3.metric("Very strong flats", f"{strong_count}")
m4.metric("Average composition", f"{score_to_100(avg_score) or 0}/100")

st.caption(
    f"Fixed layer mix: {int(meta['personality_weight'] * 100)}% personality equal-profile + {int(meta['lifestyle_weight'] * 100)}% lifestyle. "
    f"Potential issues are triggered at overall ≤ {risk_threshold}, weakest link ≤ {weak_link_threshold}, or any min link below 30. "
    f"Very strong flats require overall ≥ {strong_threshold}, weakest link ≥ {STRONG_WEAK_LINK_DEFAULT}, and fairness ≥ {STRONG_FAIRNESS_DEFAULT}. "
    f"Average empty rooms across tracked flats: {float(avg_empty_rooms):.1f}."
)

if cards_df.empty:
    st.warning("No apartments remain after the current dashboard filters.")
else:
    if focus_mode == "Potential issues first":
        risk_df = cards_df.loc[cards_df["status_tone"] == "risk"].sort_values(by=["min_pair", "struct_score", "fairness"], ascending=[True, True, True])
        strong_df = cards_df.loc[cards_df["status_tone"] == "strong"].sort_values(by=["struct_score", "min_pair", "fairness"], ascending=False)
        neutral_df = cards_df.loc[~cards_df["status_tone"].isin(["risk", "strong"])].sort_values(by=[sort_col, "apartment"], ascending=[ascending, True])
    elif focus_mode == "Strongest first":
        strong_df = cards_df.loc[cards_df["status_tone"] == "strong"].sort_values(by=["struct_score", "min_pair", "fairness"], ascending=False)
        risk_df = cards_df.loc[cards_df["status_tone"] == "risk"].sort_values(by=["min_pair", "struct_score", "fairness"], ascending=[True, True, True])
        neutral_df = cards_df.loc[~cards_df["status_tone"].isin(["risk", "strong"])].sort_values(by=[sort_col, "apartment"], ascending=[ascending, True])
    else:
        strong_df = pd.DataFrame(columns=cards_df.columns)
        risk_df = pd.DataFrame(columns=cards_df.columns)
        neutral_df = cards_df

    if focus_mode != "All apartments":
        st.markdown('<div class="section-title">Potential issues</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">These flats are highlighted because their overall composition score is weak, their weakest internal pairing falls below the current threshold, or a critically weak link drops below 30 regardless of the broader apartment average.</div>',
            unsafe_allow_html=True,
        )
        if risk_df.empty:
            st.info("No apartments are currently classified as potential issues under the chosen thresholds.")
        else:
            for _, row in risk_df.iterrows():
                render_apartment_block(row, apartment_details[row["apartment"]], row["status_label"], row["status_tone"])

        st.markdown('<div class="section-title">Strongest compositions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">These flats stand out positively on the fixed apartment-composition score, maintain a strong weakest-link score, and remain balanced across residents.</div>',
            unsafe_allow_html=True,
        )
        if strong_df.empty:
            st.info("No apartments are currently classified as especially strong under the chosen thresholds.")
        else:
            for _, row in strong_df.iterrows():
                render_apartment_block(row, apartment_details[row["apartment"]], row["status_label"], row["status_tone"])

        if not neutral_df.empty:
            st.markdown('<div class="section-title">Other monitored apartments</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">These flats sit between the current red and green thresholds.</div>',
                unsafe_allow_html=True,
            )
            for _, row in neutral_df.iterrows():
                render_apartment_block(row, apartment_details[row["apartment"]], row["status_label"], row["status_tone"])
    else:
        st.markdown('<div class="section-title">Apartment overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">All apartments below are scored on the current resident composition only.</div>',
            unsafe_allow_html=True,
        )
        for _, row in cards_df.iterrows():
            render_apartment_block(row, apartment_details[row["apartment"]], row["status_label"], row["status_tone"])

with st.expander("See full apartment summary table", expanded=False):
    operator_table = prepare_operator_table(classified_df)
    st.dataframe(operator_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download apartment summary as CSV",
        data=operator_table.to_csv(index=False).encode("utf-8"),
        file_name="operator_apartment_summary.csv",
        mime="text/csv",
    )
