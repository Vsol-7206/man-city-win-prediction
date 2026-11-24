# Manchester City win prediction

Can recent form improve a pre-match estimate of Manchester City's chance of winning, compared with using their historical win rate for every fixture? This proof of concept is for a football analyst deciding which upcoming fixtures merit a closer look.

## Approach

- Predict a win versus a draw or loss with logistic regression.
- Use home advantage and the preceding five matches' average expected goals for, expected goals against, and win rate. Expected goals (xG) estimate the likelihood of scoring from the chances created. The calculation excludes the fixture being predicted.
- Train on 148 eligible matches and evaluate on the final 37. The first five matches supply history for the rolling features. The baseline uses the win rate from the training period.

## Holdout results

| Model | Brier score (lower is better) | Average precision (higher is better) |
| --- | ---: | ---: |
| Historical win rate | 0.198 | 0.730 |
| Logistic regression | 0.187 | 0.865 |

The model ranked wins better than the constant baseline and modestly improved probability accuracy. At a 50% threshold both models predicted a win for all 37 matches, including all 10 non-wins. An analyst could use the scores to rank fixtures for review. A larger holdout and a threshold chosen on earlier data would be needed before using the scores to flag non-wins automatically.

## Reproduce

Use Python 3.10 or later and a chronological match dataset with one row per fixture. Required CSV columns are `Date` (ISO date), `Is_Home` (1 for home, 0 for away), `xG`, `Opp_xG`, and `Win_Binary` (1 for win, 0 otherwise). The evaluation used an existing prepared dataset of 190 Manchester City matches. The CSV is excluded pending confirmation of its source and redistribution terms.

```sh
python -m pip install -r requirements.txt
python evaluate.py /path/to/matches.csv
```
