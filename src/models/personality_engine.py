# ======================================
# Personality Engine
# Reusable Stage 1 -> Stage 2 bridge
# ======================================

import numpy as np
import pandas as pd


# ======================================
# Global settings
# ======================================

SCALE_MIN = 1.0
SCALE_MAX = 5.0
DELTA_MAX = SCALE_MAX - SCALE_MIN

PERSONALITY_FEATURES = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "emotional_stability",
    "openness",
]


# ======================================
# Profile definitions for Stage 2
# Operational default: all traits as similarity
# ======================================

PERSONALITY_PROFILES = {
    "equal": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 1.0,
    },
    "harmony": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 1.0,
    },
    "reliability": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 1.0,
    },
}

# ======================================
# Optional sensitivity specs from Stage 1
# Mirrors the current Stage 1 notebook setup
# ======================================

STAGE1_SENSITIVITY_SPECS = {
    "equal_weights__all_sim": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 0.0,  # ignored when no complementarity traits are used
    },
    "equal_weights__mixed_0.5": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 0.5,
    },
    "equal_weights__mixed_0.8": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 0.8,
    },
    "equal_weights__mixed_1.0": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 1.0,
    },
    "equal_weights__mixed_1.2": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "conscientiousness": 1.0,
            "emotional_stability": 1.0,
            "openness": 1.0,
        },
        "ideal_delta": 1.2,
    },
    "harmony_focus__all_sim": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 0.0,
    },
    "harmony_focus__mixed_0.5": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 0.5,
    },
    "harmony_focus__mixed_0.8": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 0.8,
    },
    "harmony_focus__mixed_1.0": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 1.0,
    },
    "harmony_focus__mixed_1.2": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.8,
            "agreeableness": 1.5,
            "conscientiousness": 1.2,
            "emotional_stability": 1.5,
            "openness": 0.7,
        },
        "ideal_delta": 1.2,
    },
    "reliability_focus__all_sim": {
        "similarity_traits": PERSONALITY_FEATURES,
        "complementarity_traits": [],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 0.0,
    },
    "reliability_focus__mixed_0.5": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 0.5,
    },
    "reliability_focus__mixed_0.8": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 0.8,
    },
    "reliability_focus__mixed_1.0": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 1.0,
    },
    "reliability_focus__mixed_1.2": {
        "similarity_traits": ["agreeableness", "conscientiousness", "openness"],
        "complementarity_traits": ["extraversion", "emotional_stability"],
        "weights": {
            "extraversion": 0.7,
            "agreeableness": 1.2,
            "conscientiousness": 1.7,
            "emotional_stability": 1.2,
            "openness": 0.7,
        },
        "ideal_delta": 1.2,
    },
}

# ======================================
# Core scoring components
# ======================================

def similarity_component(a: float, b: float, delta_max: float = DELTA_MAX) -> float:
    """
    Converts absolute difference to similarity in [0, 1].

    1.0 -> identical values
    0.0 -> maximal difference
    """
    return float(np.clip(1.0 - (abs(a - b) / delta_max), 0.0, 1.0))


def complementarity_component(
    a: float,
    b: float,
    ideal_delta: float,
    delta_max: float = DELTA_MAX,
) -> float:
    """
    Scores how close |a-b| is to an ideal difference.
    Peaks when |a-b| == ideal_delta.
    """
    delta = abs(a - b)
    max_dev = max(ideal_delta, delta_max - ideal_delta)
    return float(np.clip(1.0 - (abs(delta - ideal_delta) / max_dev), 0.0, 1.0))


def compatibility_between_rows(
    row_i: pd.Series,
    row_j: pd.Series,
    *,
    similarity_traits,
    complementarity_traits=None,
    weights=None,
    ideal_delta: float = 1.0,
) -> float:
    """
    Computes weighted compatibility score between two rows.
    """
    if complementarity_traits is None:
        complementarity_traits = []

    if weights is None:
        weights = {t: 1.0 for t in PERSONALITY_FEATURES}

    num = 0.0
    den = 0.0

    for trait in similarity_traits:
        w = float(weights.get(trait, 1.0))
        num += w * similarity_component(row_i[trait], row_j[trait])
        den += w

    for trait in complementarity_traits:
        w = float(weights.get(trait, 1.0))
        num += w * complementarity_component(
            row_i[trait],
            row_j[trait],
            ideal_delta=ideal_delta,
        )
        den += w

    return num / den if den else np.nan


def compatibility_matrix(
    df_in: pd.DataFrame,
    *,
    similarity_traits,
    complementarity_traits=None,
    weights=None,
    ideal_delta: float = 1.0,
) -> pd.DataFrame:
    """
    Builds symmetric pairwise compatibility matrix.
    Uses df_in.index as matrix row/column labels.
    """
    if complementarity_traits is None:
        complementarity_traits = []

    n = len(df_in)
    M = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            s = compatibility_between_rows(
                df_in.iloc[i],
                df_in.iloc[j],
                similarity_traits=similarity_traits,
                complementarity_traits=complementarity_traits,
                weights=weights,
                ideal_delta=ideal_delta,
            )
            M[i, j] = s
            M[j, i] = s

    return pd.DataFrame(M, index=df_in.index, columns=df_in.index)


# ======================================
# Builders
# ======================================

def build_personality_matrices(
    df_in: pd.DataFrame,
    profiles: dict = None,
    id_col: str = "unit",
) -> dict:
    """
    Precompute all personality matrices for a given dataframe.

    Returns
    -------
    dict
        {
            "equal": DataFrame,
            "harmony": DataFrame,
            "reliability": DataFrame,
        }
    """
    if profiles is None:
        profiles = PERSONALITY_PROFILES

    missing = [c for c in PERSONALITY_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing personality columns: {missing}")

    work_df = df_in.copy()

    if id_col in work_df.columns:
        if work_df[id_col].duplicated().any():
            raise ValueError(f"Column '{id_col}' contains duplicate IDs.")
        work_df = work_df.set_index(id_col, drop=False)

    if work_df[PERSONALITY_FEATURES].isna().any().any():
        raise ValueError("Personality columns contain missing values.")

    matrices = {}

    for profile_name, cfg in profiles.items():
        matrices[profile_name] = compatibility_matrix(
            df_in=work_df[PERSONALITY_FEATURES],
            similarity_traits=cfg["similarity_traits"],
            complementarity_traits=cfg.get("complementarity_traits", []),
            weights=cfg.get("weights", None),
            ideal_delta=cfg.get("ideal_delta", 1.0),
        )

    return matrices


def build_stage1_sensitivity_matrices(
    df_in: pd.DataFrame,
    specs: dict = None,
    id_col: str = "unit",
) -> dict:
    """
    Optional helper for rebuilding earlier Stage 1 variants.
    """
    if specs is None:
        specs = STAGE1_SENSITIVITY_SPECS

    missing = [c for c in PERSONALITY_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing personality columns: {missing}")

    work_df = df_in.copy()

    if id_col in work_df.columns:
        if work_df[id_col].duplicated().any():
            raise ValueError(f"Column '{id_col}' contains duplicate IDs.")
        work_df = work_df.set_index(id_col, drop=False)

    if work_df[PERSONALITY_FEATURES].isna().any().any():
        raise ValueError("Personality columns contain missing values.")

    matrices = {}

    for spec_name, cfg in specs.items():
        matrices[spec_name] = compatibility_matrix(
            df_in=work_df[PERSONALITY_FEATURES],
            similarity_traits=cfg["similarity_traits"],
            complementarity_traits=cfg.get("complementarity_traits", []),
            weights=cfg.get("weights", None),
            ideal_delta=cfg.get("ideal_delta", 1.0),
        )

    return matrices


# ======================================
# Personalization helpers
# ======================================

def select_personality_profile(flatmate_priority_value) -> str:
    """
    Maps applicant flatmate priority to one of:
    - 'equal'
    - 'harmony'
    - 'reliability'
    """
    if pd.isna(flatmate_priority_value):
        return "equal"

    x = str(flatmate_priority_value).strip().lower()

    reliability_labels = {
        "reliability in shared spaces",
        "respect for boundaries",
        "similar daily routines",
    }

    harmony_labels = {
        "being social and engaging",
        "shared interests / personality fit",
    }

    equal_labels = {
        "not important / flexible",
    }

    if x in reliability_labels:
        return "reliability"
    if x in harmony_labels:
        return "harmony"
    if x in equal_labels:
        return "equal"

    return "equal"


def get_applicant_personality_matrix(
    applicant_row: pd.Series,
    personality_matrices: dict,
    priority_col: str = "flatmatefactors_priority",
):
    """
    Returns the precomputed personality matrix selected for the applicant.
    """
    profile = select_personality_profile(applicant_row[priority_col])
    return personality_matrices[profile], profile


# ======================================
# Analysis helpers
# ======================================

def top_matches(comp_df: pd.DataFrame, k: int = 5, exclude_self: bool = True) -> dict:
    out = {}

    for idx in comp_df.index:
        scores = comp_df.loc[idx]

        if exclude_self:
            scores = scores.drop(idx, errors="ignore")

        topk = scores.nlargest(k)
        out[idx] = list(zip(topk.index.tolist(), topk.values.tolist()))

    return out


def upper_triangle_values(M) -> np.ndarray:
    A = M.values if hasattr(M, "values") else np.asarray(M)
    return A[np.triu_indices_from(A, k=1)]