"""Evaluate a pre-match Manchester City win classifier on a chronological holdout."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path)
    args = parser.parse_args()

    matches = pd.read_csv(args.matches, parse_dates=["Date"]).sort_values("Date")
    if matches["Date"].isna().any() or matches["Date"].duplicated().any():
        parser.error("Expected one match per date and no missing dates.")

    # Shift excludes the current match from each rolling form measure.
    features = matches[["Is_Home"]].copy()
    for column in ("xG", "Opp_xG", "Win_Binary"):
        features[f"prior_five_{column}"] = matches[column].shift().rolling(5).mean()

    usable = features.notna().all(axis=1)
    X = features.loc[usable]
    y = matches.loc[usable, "Win_Binary"]
    if len(X) <= 37 or y.iloc[:-37].nunique() < 2 or y.iloc[-37:].nunique() < 2:
        parser.error("Need at least 38 usable rows and both outcomes in each split.")

    X_train, X_test = X.iloc[:-37], X.iloc[-37:]
    y_train, y_test = y.iloc[:-37], y.iloc[-37:]
    print(f"{len(X_train)} training matches, {len(X_test)} holdout matches")
    print(f"Holdout win rate: {y_test.mean():.3f}")

    models = {
        "Historical win-rate baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    }
    for label, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        print(f"\n{label}")
        print(f"  Brier score: {brier_score_loss(y_test, probabilities):.3f}")
        print(f"  Average precision: {average_precision_score(y_test, probabilities):.3f}")
        print("  Confusion matrix (actual rows, predicted columns; non-win then win):")
        print(confusion_matrix(y_test, probabilities >= 0.5, labels=[0, 1]))


if __name__ == "__main__":
    main()
