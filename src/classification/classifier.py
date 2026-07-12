"""
Fixed-classifier train/test protocol (Algorithm 6).

The Random Forest classifier is trained once on gap-free features and
frozen — never retrained on imputed data. This isolates imputation quality
as the sole variable affecting downstream accuracy.

Also provides:
  - train_classifier()
  - evaluate_classifier()
  - cross_validate_classifier()

Requires: scikit-learn
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

CLASS_NAMES = ["RRLYR", "DSCT", "EB", "GDOR", "SOL", "ROT"]


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    random_state: int = 42,
    cv_folds: int = 5,
    model_path: Optional[Path] = None,
) -> RandomForestClassifier:
    """
    Train the Random Forest classifier on gap-free features.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 35)
        Feature matrix from gap-free training light curves.
    y_train : np.ndarray, shape (n_train,)
        Integer class labels.
    n_estimators : int
        Number of trees (default 500).
    random_state : int
        Random seed (default 42).
    cv_folds : int
        Number of stratified CV folds for hyperparameter validation.
    model_path : Path | None
        If given, save the trained model as a pickle.

    Returns
    -------
    RandomForestClassifier
        Fitted (frozen) classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="accuracy",
    )
    logger.info(
        "RF classifier CV accuracy: %.4f ± %.4f (folds=%d)",
        cv_scores.mean(), cv_scores.std(), cv_folds,
    )

    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Classifier saved to %s", model_path)

    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Evaluate a frozen classifier on a test feature matrix.

    Parameters
    ----------
    clf : RandomForestClassifier
        Pre-trained (frozen) classifier.
    X_test : np.ndarray
        Feature matrix (may be from imputed light curves).
    y_test : np.ndarray
        True class labels.
    class_names : list[str] | None
        Human-readable class names for the report.

    Returns
    -------
    dict with keys:
        accuracy, f1_macro, mcc, confusion_matrix, report, y_pred
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_pred = clf.predict(X_test)

    return {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "f1_macro":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "mcc":              float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report":           classification_report(y_test, y_pred, target_names=class_names,
                                                   zero_division=0),
        "y_pred":           y_pred,
    }


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> SVC:
    """
    Train the secondary SVM classifier (for sensitivity analysis).

    Parameters
    ----------
    X_train, y_train : as above
    random_state : int

    Returns
    -------
    SVC
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=random_state)),
    ])
    clf.fit(X_train, y_train)
    logger.info("SVM classifier trained")
    return clf


def load_classifier(model_path: Path) -> RandomForestClassifier:
    """Load a serialised classifier from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_feature_importance(
    clf: RandomForestClassifier,
    feature_names: list[str],
) -> pd.Series:
    """
    Extract and rank feature importances from the gap-free RF classifier.

    Returns
    -------
    pd.Series indexed by feature name, sorted descending.
    """
    importances = clf.feature_importances_
    return (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
    )
