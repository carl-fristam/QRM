import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

plt.style.use('default')


#Other functions needed to import
from Helpers.evaluate import evaluate_model


def visualize_results(results, predictions, y_test):
    """
    Visualize metrics, ROC curves, PR curves, and confusion matrices for any number of models.
    The layout scales automatically based on the number of pipelines.
    """

    # Sort results by model name to keep order consistent
    results = sorted(results, key=lambda r: r['model'])

    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)

    # ==========================================
    # 1. METRICS BAR PLOT (Dynamic)
    # ==========================================
    metrics_to_plot = ['roc_auc', 'pr_auc', 'fraud_f1', 'fraud_recall', 'fraud_precision']

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.set_index('model')[metrics_to_plot].plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. ROC CURVES (Dynamic)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curves')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 3. PRECISION RECALL CURVES (Dynamic)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, y_pred_proba in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        ax.plot(recall, precision, label=f'{model_name} (AP={pr_auc:.3f})')

    ax.set_title('Precision Recall Curves')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 4. CONFUSION MATRICES (Dynamic grid)
    # ==========================================
    n_models = len(results)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if idx < n_models:
                result = results[idx]
                cm = result['confusion_matrix']
                ax = axes[r, c]

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Non Fraud', 'Fraud'],
                            yticklabels=['Non Fraud', 'Fraud'])
                ax.set_title(f"{result['model']} Confusion Matrix")
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
            else:
                axes[r, c].axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()

    # ==========================================
    # PRINT SUMMARY TABLE
    # ==========================================
    print("\nSummary Table")
    print(metrics_df[['model', 'roc_auc', 'pr_auc', 'fraud_f1', 'fraud_precision', 'fraud_recall']].round(4))




def compare_models(pipelines, X_train, X_test, y_train, y_test):
    """
    Compare all pipelines and visualize results.
    """
    results = []
    predictions = {}
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    print(f"Total models: {len(pipelines)}\n")

    # Progress bar for models
    for name, pipeline in tqdm(pipelines.items(), desc="Training models"):
        
        print(f"\n→ Training {name}...")
        start = time.time()

        metrics, y_pred_proba = evaluate_model(
            pipeline, X_train, X_test, y_train, y_test, name
        )

        end = time.time()
        duration = end - start
        print(f"✓ Done: {name} ({duration:.2f} seconds)")

        results.append(metrics)
        predictions[name] = y_pred_proba
        
        print(f"\n{name} Results:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  Fraud F1-Score: {metrics['fraud_f1']:.4f}")
        print(f"  Fraud Precision: {metrics['fraud_precision']:.4f}")
        print(f"  Fraud Recall: {metrics['fraud_recall']:.4f}")
    
    print("\nAll models trained. Generating visualizations...\n")

    # Visualize results
    visualize_results(results, predictions, y_test)
    
    return results