from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a single pipeline.
    Returns metrics dictionary.
    """
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Get precision, recall, f1 for each class
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['fraud_precision'] = report['1']['precision']
    metrics['fraud_recall'] = report['1']['recall']
    metrics['fraud_f1'] = report['1']['f1-score']
    
    return metrics, y_pred_proba