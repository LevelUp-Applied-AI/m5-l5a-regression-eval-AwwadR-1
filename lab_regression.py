"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    path = Path(filepath)
    if not path.exists():
        # Support tests that pass a legacy "starter/data/..." path.
        fallback = Path(__file__).resolve().parent / "data" / path.name
        path = fallback if fallback.exists() else path
    df = pd.read_csv(path)
    return df
    

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify_value = y if target_col == "churned" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_value
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Error splitting data: {e}")
        return None


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

        return metrics

    except Exception as e:
        print(f"Error evaluating classifier: {e}")
        return None


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

        return metrics

    except Exception as e:
        print(f"Error evaluating regressor: {e}")
        return None


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    try:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
        print("Fold scores:", scores)
        return scores
    except Exception as e:
        print(f"Error running cross-validation: {e}")
        return None


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")



# Summary of Findings:
# The most important features for churn prediction are tenure, monthly charges, and number of support calls, as these are related to customer status and  often influence a customer's decision to stay or leave.
# By using the Logistic Regression model for predicting whether a customer would churn I noticed that the performance was at average level as the as it achieved an accuracy of approximately 63% and was able to recognize some customers who would actually leave.
# In this problem recall is more important than precision, because failing to identify a customer who might leave is bigger issue than falsely predicting a customer who might not leave.
# However, the low precision shows that the model is still giving many incorrect predictions. To solve this I recommend optimizing used characteristics, adjusting the model settings, or trying threshold tuning.

