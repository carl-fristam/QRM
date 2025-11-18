from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



def create_pipelines(random_state=42):
    """
    Create pipelines for Logistic Regression, Decision Tree, Random Forest, and XGBoost.
    Each model is evaluated with three balancing strategies:
        1. Baseline
        2. SMOTE
        3. Class weights (if supported)
    """

    pipelines = {}

    # ============================
    # DEFINE MODELS
    # ============================
    models = {
        "LR": LogisticRegression(random_state=random_state, max_iter=1000),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state),
        "XGB": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    # ============================
    # WHICH MODELS SUPPORT CLASS WEIGHTS?
    # ============================
    supports_class_weight = {
        "LR": True,
        "DT": True,
        "RF": True,
        "XGB": False  # XGBoost does not use class_weight reliably → use scale_pos_weight instead
    }

    # ============================
    # CONSTRUCT PIPELINES
    # ============================
    for name, model in models.items():

        # ---- 1. Baseline pipeline ----
        pipelines[f"{name}_Baseline"] = ImbPipeline([
            ("classifier", model)
        ])

        # ---- 2. SMOTE pipeline ----
        pipelines[f"{name}_SMOTE"] = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", model)
        ])

        # ---- 3. Class weighting / equivalent ----
        if supports_class_weight[name]:

            weighted_model = None

            if name == "LR":
                weighted_model = LogisticRegression(
                    random_state=random_state, 
                    max_iter=1000,
                    class_weight="balanced"
                )
            elif name == "DT":
                weighted_model = DecisionTreeClassifier(
                    random_state=random_state, 
                    class_weight="balanced"
                )
            elif name == "RF":
                weighted_model = RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced"
                )

            pipelines[f"{name}_ClassWeight"] = ImbPipeline([
                ("classifier", weighted_model)
            ])

        else:
            # XGBoost alternative: scale_pos_weight for imbalance
            pipelines[f"{name}_ScalePosWeight"] = ImbPipeline([
                ("classifier", XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    scale_pos_weight=1  # Replace later with ratio: (non-fraud / fraud)
                ))
            ])

    return pipelines