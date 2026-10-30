import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("Dataset.csv")

print("Original Dataset Shape:", df.shape)

# Step 2: Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Step 3: Check duplicate rows
print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())

# Step 4: Remove duplicate rows
df = df.drop_duplicates()

# Step 5: Clean text columns
text_columns = [
    "type", "title", "director", "country",
    "rating", "duration", "listed_in"
]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Step 6: Convert empty strings to missing values
df[text_columns] = df[text_columns].replace("", pd.NA)

# Step 7: Handle missing values
df["director"] = df["director"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")
df["duration"] = df["duration"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")

# Step 8: Convert date_added to datetime
df["date_added"] = pd.to_datetime(
    df["date_added"].str.strip(),
    format="mixed",
    errors="coerce"
)

# Step 9: Ensure release_year is numeric
df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
).astype("Int64")

# Step 10: Check cleaned data
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

print("\nCleaned Dataset Shape:", df.shape)

print("\nUpdated Data Types:")
print(df.dtypes)

# Step 11: Save cleaned dataset
df.to_csv("netflix_cleaned.csv", index=False)

print("\nTask 1 completed successfully!")
print("Cleaned dataset saved as netflix_cleaned.csv")

