# ==========================================================
# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 4: TREND PREDICTION ANALYSIS
# Netflix Movies and TV Shows Dataset
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("TASK 4: TREND PREDICTION ANALYSIS")
print("=" * 60)

# 1. Create output folder
output_folder = "task4_trend_outputs"
os.makedirs(output_folder, exist_ok=True)

# 2. Load dataset
data = pd.read_csv("netflix_cleaned.csv")

print("\nDataset Loaded Successfully!")
print("Original Dataset Shape:", data.shape)

# 3. Prepare release year
data["release_year"] = pd.to_numeric(
    data["release_year"], errors="coerce"
)

data = data.dropna(
    subset=["release_year", "type"]
).copy()

data["release_year"] = data["release_year"].astype(int)

data = data[
    (data["release_year"] >= 2000) &
    (data["release_year"] <= 2021)
].copy()

if data.empty:
    raise ValueError("No valid records found for trend analysis.")

# 4. Year-wise content analysis
yearly_trends = (
    data.groupby(["release_year", "type"])
    .size()
    .unstack(fill_value=0)
)

for col in ["Movie", "TV Show"]:
    if col not in yearly_trends.columns:
        yearly_trends[col] = 0

yearly_trends = yearly_trends[["Movie", "TV Show"]]

all_years = range(
    yearly_trends.index.min(),
    yearly_trends.index.max() + 1
)

yearly_trends = yearly_trends.reindex(
    all_years, fill_value=0
)

yearly_trends.index.name = "Year"

yearly_trends["Total Content"] = (
    yearly_trends["Movie"] + yearly_trends["TV Show"]
)

print("\nYEARLY CONTENT TRENDS")
print(yearly_trends.tail(10))

# OUTPUT 1: Historical trend CSV
yearly_trends.to_csv(
    os.path.join(output_folder, "yearly_content_trends.csv")
)

print("\n1. Yearly content trends CSV saved.")

# 5. Historical trend visualization
plt.figure(figsize=(12, 6))

plt.plot(
    yearly_trends.index,
    yearly_trends["Movie"],
    marker="o",
    label="Movies"
)

plt.plot(
    yearly_trends.index,
    yearly_trends["TV Show"],
    marker="s",
    label="TV Shows"
)

plt.title("Netflix Historical Content Trends")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.legend()
plt.grid(True)
plt.tight_layout()

# OUTPUT 2: Historical trend chart
plt.savefig(
    os.path.join(output_folder, "yearly_content_trend.png"),
    dpi=300
)
plt.close()

print("2. Historical trend chart saved.")

# 6. Prepare Linear Regression model
X = yearly_trends.index.to_numpy().reshape(-1, 1)

last_year = int(yearly_trends.index.max())

future_years = list(range(last_year + 1, last_year + 6))

future_X = pd.DataFrame({
    "Year": future_years
}).to_numpy()

predictions = {}
evaluation_results = {}

# 7. Train separate models
for content_type in ["Movie", "TV Show"]:

    y = yearly_trends[content_type].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    predicted_values = model.predict(future_X)

    predicted_values = [
        max(0, round(value))
        for value in predicted_values
    ]

    predictions[content_type] = predicted_values

    training_predictions = model.predict(X)

    mae = mean_absolute_error(y, training_predictions)
    r2 = r2_score(y, training_predictions)

    evaluation_results[content_type] = {
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 3)
    }

    print(f"\n{content_type} Model Results")
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 3))

# 8. Future prediction table
forecast_data = pd.DataFrame({
    "Year": future_years,
    "Predicted Movies": predictions["Movie"],
    "Predicted TV Shows": predictions["TV Show"]
})

forecast_data["Predicted Total Content"] = (
    forecast_data["Predicted Movies"] +
    forecast_data["Predicted TV Shows"]
)

print("\nFUTURE CONTENT PREDICTIONS")
print(forecast_data)

# OUTPUT 3: Prediction CSV
forecast_data.to_csv(
    os.path.join(output_folder, "trend_predictions.csv"),
    index=False
)

print("\n3. Future prediction CSV saved.")

# 9. Future trend visualization
plt.figure(figsize=(12, 6))

for content_type, column in [
    ("Movie", "Movie"),
    ("TV Show", "TV Show")
]:

    plt.plot(
        yearly_trends.index,
        yearly_trends[column],
        marker="o",
        label=f"Historical {content_type}"
    )

    plt.plot(
        future_years,
        predictions[content_type],
        marker="s",
        linestyle="--",
        label=f"Predicted {content_type}"
    )

plt.title("Netflix Future Content Growth Prediction")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.legend()
plt.grid(True)
plt.tight_layout()

# OUTPUT 4: Future forecast chart
plt.savefig(
    os.path.join(output_folder, "future_content_forecast.png"),
    dpi=300
)
plt.close()

print("4. Future forecast chart saved.")

# 10. Movie and TV Show comparison chart
plt.figure(figsize=(12, 6))

plt.bar(
    yearly_trends.index - 0.2,
    yearly_trends["Movie"],
    width=0.4,
    label="Movies"
)

plt.bar(
    yearly_trends.index + 0.2,
    yearly_trends["TV Show"],
    width=0.4,
    label="TV Shows"
)

plt.title("Year-wise Movies and TV Shows Comparison")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

# OUTPUT 5: Comparison chart
plt.savefig(
    os.path.join(output_folder, "content_type_comparison.png"),
    dpi=300
)
plt.close()

print("5. Content comparison chart saved.")

# 11. Generate report
report_path = os.path.join(
    output_folder,
    "task4_trend_prediction_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("TASK 4: TREND PREDICTION ANALYSIS\n")
    report.write("=" * 55 + "\n\n")

    report.write("Project: Netflix Content Trend Analysis\n")
    report.write("Algorithm: Linear Regression\n")
    report.write(f"Records Analyzed: {len(data)}\n")

    report.write(
        f"Historical Period: {yearly_trends.index.min()} "
        f"to {yearly_trends.index.max()}\n\n"
    )

    report.write("YEARLY CONTENT TRENDS\n")
    report.write(yearly_trends.to_string())

    report.write("\n\nFUTURE CONTENT PREDICTIONS\n")
    report.write(forecast_data.to_string(index=False))

    report.write("\n\nMODEL EVALUATION\n")

    for content_type, results in evaluation_results.items():
        report.write(f"\n{content_type}\n")
        report.write(f"MAE: {results['MAE']}\n")
        report.write(f"R2 Score: {results['R2 Score']}\n")

    report.write(
        "\n\nLIMITATION:\n"
        "Predictions are based on historical linear trends. "
        "They are estimates and do not guarantee actual "
        "future Netflix releases.\n"
    )

print("\n6. Evaluation report saved.")

# 12. Final output verification
print("\n" + "=" * 60)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGENERATED OUTPUT FILES:")

files = sorted(os.listdir(output_folder))

for index, filename in enumerate(files, start=1):
    print(f"{index}. {filename}")

print("\nTotal Output Files:", len(files))
print("Output Folder:", os.path.abspath(output_folder))

if len(files) == 6:
    print("\nSUCCESS: All 6 output files generated!")
else:
    print("\nWARNING: Expected 6 files, but found", len(files))