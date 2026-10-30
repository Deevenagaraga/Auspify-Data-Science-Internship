# Netflix Data Cleaning and Exploratory Data Analysis

## 1. Project Overview

This project focuses on cleaning, preprocessing, and performing Exploratory Data Analysis (EDA) on the Netflix Movies and TV Shows dataset using Python.

The main objective is to prepare the dataset for analysis, identify meaningful patterns, discover trends, and understand the distribution of Netflix content.

This project was completed as part of the Data Science Internship at Auspify Technologies.

## 2. Dataset Information

* **Dataset Name:** Netflix Movies and TV Shows
* **Total Records:** 8,790
* **Total Columns:** 10
* **File Format:** CSV

### Dataset Columns

1. show_id
2. type
3. title
4. director
5. country
6. date_added
7. release_year
8. rating
9. duration
10. listed_in

## 3. Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Visual Studio Code

## 4. Task 1 – Data Cleaning and Preprocessing

### Objective

To clean and preprocess the Netflix dataset and make it suitable for further analysis.

### Data Cleaning Steps

1. Imported the dataset using Pandas.
2. Checked missing values.
3. Checked duplicate records.
4. Removed duplicate rows.
5. Removed unnecessary spaces from text columns.
6. Handled empty text values.
7. Converted the `date_added` column into datetime format.
8. Converted `release_year` into numeric format.
9. Verified the cleaned dataset.
10. Exported the cleaned dataset as a CSV file.

### Task 1 Results

* Original Dataset Shape: 8,790 rows and 10 columns.
* Missing Values Before Cleaning: 0
* Duplicate Rows Before Cleaning: 0
* Missing Values After Cleaning: 0
* Duplicate Rows After Cleaning: 0
* Cleaned Dataset Shape: 8,790 rows and 10 columns.

### Task 1 Output

`netflix_cleaned.csv`

## 5. Task 2 – Exploratory Data Analysis (EDA)

### Objective

To explore the Netflix dataset and uncover trends, patterns, and meaningful insights.

### EDA Steps Performed

1. Analyzed dataset structure and statistical information.
2. Studied content distribution by type (Movies and TV Shows).
3. Identified the top 10 countries producing Netflix content.
4. Identified the top 10 content categories.
5. Analyzed content rating distribution.
6. Explored release year trends.
7. Analyzed movie duration.
8. Studied the number of seasons in TV Shows.
9. Analyzed Netflix content added over time.
10. Generated a summary report of the findings.

### Visualizations Created

The following visualizations were generated using Matplotlib and Seaborn:

1. Content Type Distribution
2. Top 10 Countries
3. Top 10 Content Categories
4. Content Rating Distribution
5. Release Year Trend
6. Movie Duration Distribution
7. TV Show Seasons Distribution
8. Content Added by Year

### Task 2 Output Files

* `task2_eda.py` – Python source code.
* `task2_eda_report.txt` – EDA findings summary.
* `task2_eda_charts/` – Visualization charts.
* `top_10_countries.csv` – Top country analysis.
* `top_10_categories.csv` – Top category analysis.

## 6. Project Folder Structure

```text
Auspify_Data_Science_Internship/
│
├── Dataset.csv
├── netflix_cleaned.csv
├── task1_data_cleaning.py
├── task2_eda.py
├── task2_eda_report.txt
├── README.md
│
├── task2_eda_charts/
│   ├── 01_content_type_distribution.png
│   ├── 02_top_10_countries.png
│   ├── 03_top_10_categories.png
│   ├── 04_content_rating_distribution.png
│   ├── 05_release_year_trend.png
│   ├── 06_movie_duration_distribution.png
│   ├── 07_tv_show_seasons.png
│   ├── 08_content_added_by_year.png
│   ├── top_10_countries.csv
│   └── top_10_categories.csv
│
└── Task2_Screenshots/
    └── Task 2 screenshots
```

## 7. Final Results

Both data cleaning and exploratory data analysis were completed successfully.

The cleaned Netflix dataset was used to explore content distribution, popular countries, content categories, ratings, release year trends, movie durations, and TV show seasons.

Visualizations and a summary report were generated to present the analysis results.

## 8. Conclusion

This project provided practical experience in data cleaning, preprocessing, exploratory data analysis, statistical analysis, and data visualization using Python.

The analysis helped identify important patterns and trends in the Netflix Movies and TV Shows dataset.

The project outputs are organized and prepared for further analysis.

---

**Internship:** Data Science Intern
**Organization:** Auspify Technologies
**Tasks Completed:**

* Task 1 – Data Cleaning and Preprocessing
* Task 2 – Exploratory Data Analysis (EDA)

**Intern:** Garaga Deevena
