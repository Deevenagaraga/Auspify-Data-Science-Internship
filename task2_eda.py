# ============================================================
# AUS PIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Netflix Dataset
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------

print("=" * 60)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Input and output paths
INPUT_FILE = "netflix_cleaned.csv"
CHART_FOLDER = "task2_eda_charts"

# Create charts folder
os.makedirs(CHART_FOLDER, exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")

# Check dataset availability
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. "
        "Keep this file in the same folder as task2_eda.py."
    )

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Convert columns into suitable data types
if "date_added" in df.columns:
    df["date_added"] = pd.to_datetime(
        df["date_added"], errors="coerce"
    )

if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

print("\nDataset loaded successfully!")

# ------------------------------------------------------------
# 2. ANALYZE DATASET STRUCTURE AND STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: DATASET STRUCTURE AND STATISTICS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all").to_string())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ------------------------------------------------------------
# 3. STUDY CONTENT DISTRIBUTION BY TYPE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: CONTENT DISTRIBUTION BY TYPE")
print("=" * 60)

type_counts = df["type"].value_counts()

type_percentages = (
    df["type"].value_counts(normalize=True) * 100
)

print("\nContent Type Counts:")
print(type_counts)

print("\nContent Type Percentages:")
print(type_percentages.round(2))

# Save content type chart
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type",
    hue="type",
    legend=False
)

plt.title("Netflix Content Distribution by Type")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "01_content_type_distribution.png")
)
plt.close()

# ------------------------------------------------------------
# 4. IDENTIFY TOP COUNTRIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3A: TOP COUNTRIES")
print("=" * 60)

country_data = (
    df["country"]
    .dropna()
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)

country_data = country_data[
    (country_data != "") &
    (country_data.str.lower() != "unknown")
]

top_countries = country_data.value_counts().head(10)

print("\nTop 10 Countries:")
print(top_countries)

# Save country results
top_countries.to_csv(
    os.path.join(CHART_FOLDER, "top_10_countries.csv"),
    header=["Number of Titles"]
)

# Top countries chart
plt.figure(figsize=(11, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    hue=top_countries.index,
    legend=False
)

plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "02_top_10_countries.png")
)
plt.close()

# ------------------------------------------------------------
# 5. IDENTIFY TOP CATEGORIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3B: TOP CONTENT CATEGORIES")
print("=" * 60)

category_data = (
    df["listed_in"]
    .dropna()
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)

category_data = category_data[
    (category_data != "") &
    (category_data.str.lower() != "unknown")
]

top_categories = category_data.value_counts().head(10)

print("\nTop 10 Categories:")
print(top_categories)

# Save category results
top_categories.to_csv(
    os.path.join(CHART_FOLDER, "top_10_categories.csv"),
    header=["Number of Titles"]
)

# Top categories chart
plt.figure(figsize=(11, 6))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index,
    hue=top_categories.index,
    legend=False
)

plt.title("Top 10 Netflix Content Categories")
plt.xlabel("Number of Titles")
plt.ylabel("Category")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "03_top_10_categories.png")
)
plt.close()

# ------------------------------------------------------------
# 6. ANALYZE CONTENT RATINGS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONTENT RATING ANALYSIS")
print("=" * 60)

rating_counts = df["rating"].value_counts().head(15)

print("\nContent Rating Distribution:")
print(rating_counts)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values,
    hue=rating_counts.index,
    legend=False
)

plt.title("Netflix Content Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "04_content_rating_distribution.png")
)
plt.close()

# ------------------------------------------------------------
# 7. ANALYZE RELEASE YEAR TRENDS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RELEASE YEAR ANALYSIS")
print("=" * 60)

year_counts = (
    df["release_year"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
)

print("\nRelease Year Statistics:")
print(df["release_year"].describe())

plt.figure(figsize=(12, 6))

plt.plot(
    year_counts.index,
    year_counts.values,
    marker="o"
)

plt.title("Netflix Content Release Year Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "05_release_year_trend.png")
)
plt.close()

# ------------------------------------------------------------
# 8. ANALYZE MOVIE DURATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MOVIE DURATION ANALYSIS")
print("=" * 60)

movies = df[df["type"] == "Movie"].copy()

movies["duration_value"] = pd.to_numeric(
    movies["duration"]
    .astype(str)
    .str.extract(r"(\d+)")[0],
    errors="coerce"
)

movie_duration_stats = movies["duration_value"].describe()

print("\nMovie Duration Statistics (Minutes):")
print(movie_duration_stats)

plt.figure(figsize=(10, 5))

sns.histplot(
    movies["duration_value"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Netflix Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "06_movie_duration_distribution.png")
)
plt.close()

# ------------------------------------------------------------
# 9. ANALYZE TV SHOW SEASONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TV SHOW SEASON ANALYSIS")
print("=" * 60)

tv_shows = df[df["type"] == "TV Show"].copy()

tv_shows["season_count"] = pd.to_numeric(
    tv_shows["duration"]
    .astype(str)
    .str.extract(r"(\d+)")[0],
    errors="coerce"
)

season_counts = tv_shows["season_count"].value_counts().sort_index()

print("\nTV Show Season Counts:")
print(season_counts)

plt.figure(figsize=(10, 5))

sns.countplot(
    data=tv_shows.dropna(subset=["season_count"]),
    x="season_count",
    hue="season_count",
    legend=False
)

plt.title("Distribution of TV Show Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")

plt.tight_layout()
plt.savefig(
    os.path.join(CHART_FOLDER, "07_tv_show_seasons.png")
)
plt.close()

# ------------------------------------------------------------
# 10. ANALYZE CONTENT ADDED OVER TIME
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONTENT ADDED OVER TIME")
print("=" * 60)

if "date_added" in df.columns:

    added_data = df.dropna(subset=["date_added"]).copy()

    if not added_data.empty:

        added_data["year_added"] = (
            added_data["date_added"].dt.year
        )

        added_year_counts = (
            added_data["year_added"].value_counts().sort_index()
        )

        print("\nTitles Added by Year:")
        print(added_year_counts)

        plt.figure(figsize=(10, 5))

        sns.lineplot(
            x=added_year_counts.index,
            y=added_year_counts.values,
            marker="o"
        )

        plt.title("Netflix Content Added by Year")
        plt.xlabel("Year Added")
        plt.ylabel("Number of Titles")

        plt.tight_layout()
        plt.savefig(
            os.path.join(CHART_FOLDER, "08_content_added_by_year.png")
        )
        plt.close()

# ------------------------------------------------------------
# 11. GENERATE FINDINGS SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: EDA FINDINGS SUMMARY")
print("=" * 60)

total_titles = len(df)

movie_count = (df["type"] == "Movie").sum()
tv_count = (df["type"] == "TV Show").sum()

movie_percentage = (
    movie_count / total_titles * 100
    if total_titles else 0
)

tv_percentage = (
    tv_count / total_titles * 100
    if total_titles else 0
)

top_country = (
    top_countries.index[0]
    if not top_countries.empty else "Not Available"
)

top_category = (
    top_categories.index[0]
    if not top_categories.empty else "Not Available"
)

top_rating = (
    rating_counts.index[0]
    if not rating_counts.empty else "Not Available"
)

if not movies["duration_value"].dropna().empty:
    average_movie_duration = round(
        movies["duration_value"].mean(), 2
    )
else:
    average_movie_duration = "Not Available"

summary = f"""
NETFLIX DATASET - EXPLORATORY DATA ANALYSIS REPORT
==================================================

1. DATASET OVERVIEW

Total Number of Titles: {total_titles}
Total Number of Columns: {df.shape[1]}
Total Movies: {movie_count}
Total TV Shows: {tv_count}

2. CONTENT DISTRIBUTION

Movies: {movie_count} ({movie_percentage:.2f}%)
TV Shows: {tv_count} ({tv_percentage:.2f}%)

3. TOP COUNTRY

Top Country: {top_country}

4. TOP CATEGORY

Top Category: {top_category}

5. CONTENT RATINGS

Most Common Rating: {top_rating}

6. MOVIE DURATION

Average Movie Duration: {average_movie_duration} minutes

7. DATA QUALITY

Missing Values: {df.isnull().sum().sum()}
Duplicate Records: {df.duplicated().sum()}

8. CONCLUSION

The Exploratory Data Analysis helped identify
content distribution, popular countries, common
categories, content ratings, release year trends,
movie durations, and TV show seasons.

Visualizations were created to understand the
patterns and trends in the Netflix dataset.

==================================================
END OF REPORT
==================================================
"""

print(summary)

# Save report
REPORT_FILE = "task2_eda_report.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write(summary)

# ------------------------------------------------------------
# 12. SAVE ANALYSIS RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated Files:")

print("\nCharts Folder:", CHART_FOLDER)
print("EDA Report:", REPORT_FILE)
print("Top Countries CSV: task2_eda_charts/top_10_countries.csv")
print("Top Categories CSV: task2_eda_charts/top_10_categories.csv")

print("\nAll EDA visualizations and reports have been saved.")

print("\nThank you!")