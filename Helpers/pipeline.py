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



def run_pipeline(df, target_col='Class', test_size=0.3, random_state=42, 
                 preprocess=True):
    """
    Main function to run the complete pipeline.
    
    Parameters:
    -----------
    df : DataFrame
        Your fraud dataset (credit card fraud format)
    target_col : str
        Name of the target column (default: 'Class')
    test_size : float
        Proportion of data for testing (default: 0.3)
    random_state : int
        Random seed for reproducibility
    preprocess : bool
        Whether to scale Amount and Time features (default: True)
        Set to False if you've already preprocessed
    """
    print("="*60)
    print("FRAUD DETECTION PIPELINE: SMOTE vs CLASS WEIGHTING")
    print("="*60)
    
    # Step 1: Prepare data (with preprocessing)
    X_train, X_test, y_train, y_test = prepare_data(
        df, target_col, test_size, random_state, preprocess
    )
    
    # Step 2: Create pipelines
    pipelines = create_pipelines(random_state)
    
    # Step 3: Compare models
    results = compare_models(pipelines, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find best model by PR-AUC (more important for imbalanced data)
    best_idx = np.argmax([r['pr_auc'] for r in results])
    best_model = results[best_idx]['model']
    
    print(f"\nBest Model (by PR-AUC): {best_model}")
    print("\nKey Insights:")
    print("- ROC-AUC: Good for overall performance")
    print("- PR-AUC: Better metric for imbalanced data (focuses on minority class)")
    print("- F1-Score: Balance between precision and recall")
    print("- Recall: Important if catching all frauds is critical")
    print("- Precision: Important if false alarms are costly")
    
    return results, pipelines