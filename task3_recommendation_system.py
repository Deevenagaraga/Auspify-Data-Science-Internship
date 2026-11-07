# ============================================================
# TASK 3: NETFLIX CONTENT RECOMMENDATION SYSTEM
# Auspify Technologies - Data Science Internship
# ============================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("=" * 60)
print(" NETFLIX CONTENT RECOMMENDATION SYSTEM")
print("=" * 60)

# STEP 1: Load Dataset

data = pd.read_csv("netflix_cleaned.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# Check required columns
required_columns = [
    "show_id", "type", "title", "director",
    "country", "rating", "listed_in"
]

for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Required column missing: {column}")

# STEP 2: Feature Extraction

features = ["type", "director", "country", "rating", "listed_in"]

for column in features:
    data[column] = data[column].fillna("").astype(str)

data["title"] = data["title"].fillna("").astype(str)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Combine relevant features
data["combined_features"] = data[features].agg(" ".join, axis=1)

data["combined_features"] = data["combined_features"].apply(clean_text)

print("\nRelevant Features Extracted Successfully!")

# STEP 3: Text Vectorization using TF-IDF

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(data["combined_features"])

print("\nTF-IDF Text Preprocessing Completed!")
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# STEP 4: Calculate Content Similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nCosine Similarity Calculated Successfully!")

# Create output folder
output_folder = "task3_recommendation_outputs"
os.makedirs(output_folder, exist_ok=True)


# Recommendation Function

def recommend_titles(title, number=10):

    matches = data[
        data["title"].str.lower() == title.lower()
    ]

    if matches.empty:
        print(f"\nTitle '{title}' not found in dataset.")
        return None

    index = matches.index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected title itself
    similarity_scores = [
        item for item in similarity_scores
        if item[0] != index
    ]

    top_results = similarity_scores[:number]

    recommendations = []

    for movie_index, score in top_results:

        recommendations.append({
            "Selected Title": title,
            "Recommended Title": data.iloc[movie_index]["title"],
            "Type": data.iloc[movie_index]["type"],
            "Category": data.iloc[movie_index]["listed_in"],
            "Similarity Score": round(float(score), 4)
        })

    return pd.DataFrame(recommendations)


# STEP 5: Generate Recommendations

# Select a sample Netflix title
sample_title = "Stranger Things"

result = recommend_titles(sample_title, 10)

if result is None:
    # Fallback to the first available title
    sample_title = data.iloc[0]["title"]
    print("\nUsing available sample title:", sample_title)
    result = recommend_titles(sample_title, 10)

print("\n" + "=" * 60)
print(" TOP 10 RECOMMENDATIONS")
print("=" * 60)

print(result.to_string(index=False))

# Save recommendations
result.to_csv(
    os.path.join(output_folder, "recommended_titles.csv"),
    index=False
)

print("\nRecommendations saved successfully!")

# STEP 6: Evaluate Recommendation Quality
# Proxy evaluation based on category overlap.
# This is not a substitute for real user feedback.

def category_set(value):
    return {
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    }


selected_row = data[
    data["title"].str.lower() == sample_title.lower()
].iloc[0]

selected_categories = category_set(selected_row["listed_in"])

category_matches = []

for _, row in result.iterrows():

    recommended_categories = category_set(row["Category"])

    if selected_categories.intersection(recommended_categories):
        category_matches.append(1)
    else:
        category_matches.append(0)

precision_at_10 = (
    sum(category_matches) / len(category_matches)
    if category_matches else 0
)

print("\n" + "=" * 60)
print(" RECOMMENDATION EVALUATION")
print("=" * 60)

print("Selected Title:", sample_title)
print("Recommendations Generated:", len(result))
print("Category Match Count:", sum(category_matches))
print("Category-based Precision:", round(precision_at_10, 2))

# Save evaluation report
report_path = os.path.join(
    output_folder,
    "task3_evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("NETFLIX CONTENT RECOMMENDATION SYSTEM\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dataset Records: {len(data)}\n")
    file.write(f"Selected Title: {sample_title}\n")
    file.write(f"Recommendations Generated: {len(result)}\n")
    file.write(f"Category Matches: {sum(category_matches)}\n")
    file.write(
        f"Category-based Precision: {precision_at_10:.2f}\n\n"
    )

    file.write("Evaluation Note:\n")
    file.write(
        "Precision is estimated using shared categories "
        "between the selected title and recommended titles. "
        "It is a proxy metric, not a measure of actual user satisfaction.\n"
    )

print("\nEvaluation Report Saved Successfully!")

# STEP 7: Visualization

plt.figure(figsize=(12, 7))

plt.barh(
    result["Recommended Title"],
    result["Similarity Score"]
)

plt.xlabel("Cosine Similarity Score")
plt.ylabel("Recommended Netflix Titles")
plt.title(f"Top 10 Recommendations for {sample_title}")

plt.gca().invert_yaxis()
plt.tight_layout()

chart_path = os.path.join(
    output_folder,
    "recommendation_similarity_chart.png"
)

plt.savefig(chart_path, dpi=300)
plt.show()

print("\nRecommendation Chart Saved Successfully!")

print("\n" + "=" * 60)
print(" TASK 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)