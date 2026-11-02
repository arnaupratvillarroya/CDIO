
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ts_utils as U


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results_s10" / "shoreline_timeseries_MS.csv"
OUTPUT_DIR = BASE_DIR / "results_s11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 72
N_CYCLES = 1
JUMP_DATE = pd.Timestamp("2020-02-01")
CONFIDENCE_LEVEL = 0.95
Z_VALUE = 1.96


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

dates = df["date"]
values = df["distance_m"].to_numpy(dtype=float)

if len(values) != 96:
    raise ValueError(f"Expected 96 observations, found {len(values)}")

if np.isnan(values).any():
    raise ValueError("The time series contains missing values.")

train_values = values[:TRAIN_SIZE]
test_values = values[TRAIN_SIZE:]
test_dates = dates.iloc[TRAIN_SIZE:]

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(f"Training observations: {len(train_values)}")
print(f"Test observations:     {len(test_values)}")
print(
    f"Training period: {dates.iloc[0].date()} "
    f"to {dates.iloc[TRAIN_SIZE - 1].date()}"
)
print(
    f"Test period: {test_dates.iloc[0].date()} "
    f"to {test_dates.iloc[-1].date()}"
)


# ============================================================
# 3. PARAMETRIC MODEL: STEP + FFT
# ============================================================

jump_positions = np.where(dates == JUMP_DATE)[0]

if len(jump_positions) != 1:
    raise ValueError(
        f"Jump date {JUMP_DATE.date()} was not found exactly once."
    )

t_jump = int(jump_positions[0])

if t_jump >= TRAIN_SIZE:
    raise ValueError("The jump date must belong to the training period.")

# Fit the model using training data only.
param_model = U.fit_model(
    "step",
    train_values,
    N_CYCLES,
    t_jump=t_jump
)

# Predict the test period.
t_test = np.arange(TRAIN_SIZE, len(values))
param_predictions = U.predict_model(param_model, t_test)

param_metrics = U.metrics(test_values, param_predictions)

print("\n" + "=" * 60)
print("PARAMETRIC MODEL: STEP + FFT")
print("=" * 60)

print(f"Number of FFT cycles: {N_CYCLES}")
print(f"Jump index: {t_jump}")
print(f"MAE:  {param_metrics['MAE']:.3f} m")
print(f"RMSE: {param_metrics['RMSE']:.3f} m")
print(f"R2:   {param_metrics['R2']:.3f}")


# ============================================================
# 4. PREDICTION INTERVALS (95%) - STEP + FFT
# ============================================================

print("\n" + "=" * 60)
print("95% PREDICTION INTERVALS")
print("=" * 60)

# Obtain fitted values for the training period.
train_predictions = U.predict_model(
    param_model,
    np.arange(TRAIN_SIZE)
)

# Calculate training residuals.
residuals = train_values - train_predictions

# Estimate residual standard deviation.
sigma = np.std(residuals, ddof=1)

if not np.isfinite(sigma) or sigma <= 0:
    raise ValueError("Invalid residual standard deviation.")

# Forecast horizon: months 1 to 24.
h = np.arange(1, len(test_values) + 1)
n = TRAIN_SIZE

# Calculate prediction interval margin.
prediction_margin = Z_VALUE * sigma * np.sqrt(1 + h / n)

lower_bound = param_predictions - prediction_margin
upper_bound = param_predictions + prediction_margin

print(f"Confidence level: {CONFIDENCE_LEVEL * 100:.0f}%")
print(f"Residual standard deviation: {sigma:.3f} m")
print(f"First month margin: +/- {prediction_margin[0]:.3f} m")
print(f"Last month margin:  +/- {prediction_margin[-1]:.3f} m")

print(
    f"First month interval: "
    f"[{lower_bound[0]:.3f}, {upper_bound[0]:.3f}] m"
)

print(
    f"Last month interval:  "
    f"[{lower_bound[-1]:.3f}, {upper_bound[-1]:.3f}] m"
)

# Check how many observed test values fall inside the intervals.
inside_interval = (
    (test_values >= lower_bound) &
    (test_values <= upper_bound)
)

coverage = np.mean(inside_interval) * 100

print(
    f"Test observations inside intervals: "
    f"{inside_interval.sum()}/{len(test_values)} ({coverage:.1f}%)"
)


# ============================================================
# 5. LOAD LSTM PREDICTIONS
# ============================================================

LSTM_FILE = OUTPUT_DIR / "lstm_predictions.csv"

if not LSTM_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {LSTM_FILE}. "
        "Run s11_forecasting.py first."
    )

lstm_df = pd.read_csv(LSTM_FILE, parse_dates=["date"])

required_columns = {"date", "actual", "lstm_prediction"}

if not required_columns.issubset(lstm_df.columns):
    raise ValueError(
        "lstm_predictions.csv must contain: "
        "date, actual, lstm_prediction"
    )

if len(lstm_df) != len(test_values):
    raise ValueError("The LSTM file must contain 24 test predictions.")

if not np.allclose(lstm_df["actual"].to_numpy(), test_values):
    raise ValueError("LSTM actual values do not match the test data.")

lstm_dates = pd.to_datetime(lstm_df["date"]).reset_index(drop=True)
expected_dates = pd.to_datetime(test_dates).reset_index(drop=True)

if not lstm_dates.equals(expected_dates):
    raise ValueError("LSTM dates do not match the test dates.")
    raise ValueError("LSTM dates do not match the test dates.")

lstm_predictions = lstm_df["lstm_prediction"].to_numpy(dtype=float)

lstm_metrics = U.metrics(test_values, lstm_predictions)


# ============================================================
# 6. METRICS COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Model": ["Step + FFT", "LSTM"],
        "MAE (m)": [
            param_metrics["MAE"],
            lstm_metrics["MAE"]
        ],
        "RMSE (m)": [
            param_metrics["RMSE"],
            lstm_metrics["RMSE"]
        ],
        "R2": [
            param_metrics["R2"],
            lstm_metrics["R2"]
        ],
    }
)

print("\n" + "=" * 60)
print("MODEL COMPARISON ON TEST DATA")
print("=" * 60)

print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "model_comparison_metrics.csv",
    index=False
)


# ============================================================
# 7. SAVE PREDICTIONS AND INTERVALS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "date": test_dates.to_numpy(),
        "actual": test_values,
        "step_fft_prediction": param_predictions,
        "lstm_prediction": lstm_predictions,
        "step_fft_lower_95": lower_bound,
        "step_fft_upper_95": upper_bound,
        "inside_95_interval": inside_interval
    }
)

predictions_df.to_csv(
    OUTPUT_DIR / "model_comparison_predictions.csv",
    index=False
)

# Save interval statistics separately.
interval_summary = pd.DataFrame(
    {
        "confidence_level": [CONFIDENCE_LEVEL],
        "residual_std_m": [sigma],
        "test_observations": [len(test_values)],
        "observations_inside_interval": [int(inside_interval.sum())],
        "empirical_coverage_percent": [coverage],
        "first_month_margin_m": [prediction_margin[0]],
        "last_month_margin_m": [prediction_margin[-1]]
    }
)

interval_summary.to_csv(
    OUTPUT_DIR / "prediction_interval_summary.csv",
    index=False
)


# ============================================================
# 8. PLOT: ACTUAL VS PREDICTED + 95% INTERVAL
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    test_values,
    label="Observed",
    marker="o"
)

plt.plot(
    test_dates,
    param_predictions,
    label="Step + FFT",
    linestyle="--",
    marker="s"
)

plt.fill_between(
    test_dates,
    lower_bound,
    upper_bound,
    alpha=0.2,
    label="Step + FFT 95% prediction interval"
)

plt.plot(
    test_dates,
    lstm_predictions,
    label="LSTM",
    linestyle=":",
    marker="^"
)

plt.title("Forecast comparison: Step + FFT vs LSTM")
plt.xlabel("Date")
plt.ylabel("Shoreline distance (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "s11_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. PLOT: ABSOLUTE ERRORS
# ============================================================

param_errors = np.abs(test_values - param_predictions)
lstm_errors = np.abs(test_values - lstm_predictions)

plt.figure(figsize=(12, 5))

plt.plot(
    test_dates,
    param_errors,
    label="Step + FFT absolute error",
    marker="s"
)

plt.plot(
    test_dates,
    lstm_errors,
    label="LSTM absolute error",
    marker="^"
)

plt.title("Absolute prediction errors")
plt.xlabel("Date")
plt.ylabel("Absolute error (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "s11_absolute_errors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON COMPLETED")
print("=" * 60)

print(f"Results saved in: {OUTPUT_DIR}")

print("\nGenerated files:")
print("- model_comparison_metrics.csv")
print("- model_comparison_predictions.csv")
print("- prediction_interval_summary.csv")
print("- s11_model_comparison.png")
print("- s11_absolute_errors.png")
