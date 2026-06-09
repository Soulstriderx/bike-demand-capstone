# Model Card: Bike Demand Forecaster

## Purpose
Predicts hourly total bike-rental demand (`cnt`) for a city bike-share system using weather and calendar features. Trained on the UCI Bike Sharing dataset (~17k hourly records, 2011–2012).

## Champion model
**RandomForestRegressor** (30 trees) — selected by lowest test RMSE across three candidates:

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| LinearRegression | 165.63 | 121.75 | 0.436 |
| RandomForest | **71.68** | 47.26 | 0.894 |
| GradientBoosting | 117.63 | 80.16 | 0.715 |

- Test split: chronological 80/20 (no shuffle) to respect time order.
- Features: hour, month, season, weather situation, temperature, humidity, wind speed, weekday, weekend flag, and cyclical sin/cos encodings of hour and month.

## Limitations
- Trained on data from 2011–2012 only — may not generalise to present-day ridership patterns.
- Does not include station-level capacity, dock status, or event data (concerts, festivals).
- Normalised continuous features mean predictions are relative to the training distribution, not absolute counts for a different city scale.
- Model is a plain sklearn pipeline; no deep-learning or sequence-aware architecture.

## Agent workflow reflection
Working with the AI coding agent was efficient for boilerplate and iterative refinement — the scaffold stubs with clear docstring contracts meant less time reading documentation and more time reviewing output. I had to correct the first push (the model file exceeded GitHub's 100 MB limit and later HF's 10 MB limit), which required reducing RandomForest tree count and migrating the model artifact to Hugging Face Hub instead of committing it to git. The agent handled the code changes quickly once the constraint was specified, but the git history clean-up (filter-branch / squash) needed manual intervention because the large blobs persisted in prior commits. Overall the specify-review-run-correct loop worked well, but infrastructure edge cases (file size limits, git history pollution) still required human judgment.
