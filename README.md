# Exploratory Gait Pattern Analysis in Parkinson's Disease

This project explores walking-force patterns in Parkinson's disease using the **PhysioNet Gait in Parkinson's Disease** dataset.

The main goal is not only to classify Parkinson's disease from walking data, but to understand **which gait patterns differ between Parkinson's patients and healthy controls**. The project focuses on interpretable gait features such as contact duration, step timing, stride timing, swing time, force loading, variability, and left-right asymmetry.

## Project Motivation

Parkinson's disease often affects walking. These changes may appear as slower rhythm, longer foot contact, higher variability, or less balanced left-right movement.

This project asks:

> Can vertical ground reaction force signals reveal interpretable differences in gait rhythm, force loading, and left-right balance between Parkinson's patients and healthy controls?

The project is designed as a thesis-facing portfolio project. It starts with exploratory signal analysis and feature extraction before moving to statistical comparison and simple classification.

## Dataset

Dataset: **PhysioNet Gait in Parkinson's Disease**

The dataset contains vertical ground reaction force signals recorded during walking. Each recording includes force measurements from sensors placed under the left and right feet.

The signal files are grouped into different study/source groups:

* `GaCo`, `GaPt`
* `JuCo`, `JuPt`
* `SiCo`, `SiPt`

where:

* `Co` = healthy control
* `Pt` = Parkinson's patient

The project preserves this structure so that results can be checked both across the full dataset and within each study group.

## Research Questions

The project is guided by the following questions:

1. Do Parkinson's patients show different foot-contact patterns compared with healthy controls?
2. Are contact duration, stride timing, swing time, and force loading different between groups?
3. Are Parkinson's gait patterns more variable or less symmetric?
4. Which interpretable gait features show the strongest group differences?
5. Can simple machine learning models separate Parkinson and control recordings using these extracted features?

## Project Workflow

The project is organized into several milestones.

### 1. Dataset Understanding

The first step was to inspect the dataset files and separate metadata files from signal files.

Outputs:

* `file_table.csv`
* `column_check.csv`

Key findings:

* All signal files have a consistent 19-column format.
* The dataset contains 306 gait signal files.
* Recording durations vary across files, so later features are normalized by time or event count where needed.

### 2. Foot-Contact Detection

A simple threshold-based method was used to detect whether each foot was touching the ground.

Basic idea:

* force near zero → foot is in the air
* force above threshold → foot is on the ground

This allowed detection of left and right foot-contact periods.

Extracted early features included:

* contact count
* contacts per second
* mean contact duration
* contact duration variability
* left-right contact differences

### 3. Contact Feature Extraction

The foot-contact detection pipeline was applied to all 306 signal files.

Output:

* `contact_feature_table.csv`

This table contains one row per recording and basic contact-based gait features.

### 4. Exploratory Group Analysis

Control and Parkinson recordings were compared using boxplots and group summary tables.

Early observations suggested:

* Parkinson recordings may show longer foot-contact durations.
* Parkinson recordings may show more variability in some contact-duration features.
* Contact rate alone may not clearly separate groups.
* Some Parkinson recordings show low contact-rate outliers.

These observations were treated as exploratory, not final conclusions.

### 5. Statistical Analysis

Mann-Whitney U tests and rank-biserial effect sizes were used to compare features between groups.

Why this was done:

* The feature distributions contain outliers.
* The data was not assumed to be normally distributed.
* P-values alone are not enough, so effect sizes were also included.

Outputs:

* `statistical_feature_comparison.csv`
* `study_wise_statistical_feature_comparison.csv`
* `feature_direction_consistency.csv`

The study-wise analysis is important because the dataset includes multiple study/source groups.

### 6. Improved Gait Features

The feature set was extended beyond basic contact duration.

New feature groups include:

* step timing
* stride timing
* swing time
* peak force
* force area
* left-right asymmetry

Output:

* `improved_gait_feature_table.csv`

These features make the project more gait-focused and thesis-relevant.

### 7. Improved Feature Analysis

The improved feature table was analyzed using descriptive statistics, boxplots, statistical testing, effect sizes, and study-wise consistency checks.

Outputs:

* `improved_feature_statistical_comparison.csv`
* `study_wise_improved_feature_statistical_comparison.csv`
* `improved_feature_direction_consistency.csv`
* `top_improved_gait_features.csv`

The goal was to identify which feature groups are most promising for explaining Parkinsonian gait differences.

### 8. Classification Baseline

Simple machine learning models were trained using the improved gait features.

Models:

* Logistic Regression
* Random Forest

Validation strategy:

* Subject-wise GroupKFold cross-validation

This is important because recordings from the same subject should not appear in both training and testing folds.

The classification step is treated as a supporting experiment, not the main goal of the project.

Outputs:

* `classification_baseline_results.csv`
* `random_forest_feature_importance.csv`
* optional: `study_holdout_classification_results.csv`

## Repository Structure

```text
parkinsons-gait-pattern-analysis/
│
├── data/
│   ├── raw/                 # Raw dataset files, not pushed to GitHub
│   └── processed/           # Processed feature tables and summaries
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_foot_contact_detection.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_exploratory_group_analysis.ipynb
│   ├── 05_statistical_analysis.ipynb
│   ├── 06_improved_gait_features.ipynb
│   ├── 07_improved_feature_analysis.ipynb
│   └── 08_classification_baseline.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── contact_detection.py
│   ├── feature_extraction.py
│   └── visualization.py
│
├── results/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Methods

### Signal Loading

Each gait signal file contains:

* time
* 8 left-foot sensor signals
* 8 right-foot sensor signals
* left total force
* right total force

The total force signals were used for contact detection and gait feature extraction.

### Foot-Contact Detection

A fixed force threshold was used to detect when the foot was in contact with the ground.

Contact events shorter than 0.2 seconds or longer than 2.0 seconds were filtered out to reduce unrealistic detections.

### Feature Extraction

The project extracts interpretable gait features from each recording.

Feature groups:

| Feature Group        | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| Contact features     | How often and how long each foot touches the ground |
| Step timing          | Time between alternating foot contacts              |
| Stride timing        | Time from one contact of the same foot to the next  |
| Swing timing         | Time when the foot is in the air                    |
| Force features       | Peak force and force area during contact            |
| Variability features | How consistent or unstable the walking pattern is   |
| Asymmetry features   | Difference between left and right foot behavior     |

### Statistical Analysis

The project uses:

* median group summaries
* boxplots
* Mann-Whitney U tests
* corrected p-values
* rank-biserial effect sizes
* study-wise consistency checks

### Classification

Classification was performed only after extracting interpretable features.

The models were evaluated using subject-wise validation to reduce leakage.

Metrics include:

* balanced accuracy
* F1-score
* sensitivity
* specificity
* precision

## Preliminary Findings

The exploratory analysis suggests that some Parkinson recordings show:

* longer foot-contact duration
* higher contact-duration variability
* lower contact-rate outliers
* possible differences in step/stride timing
* possible differences in force and left-right symmetry features

These results are preliminary and should be interpreted carefully. The project does not claim clinical diagnostic validity.

The project could be extended into a further research such as:

* interpretable gait biomarkers for Parkinson's disease
* gait variability and disease severity
* study-wise robustness of gait-based Parkinson detection
* comparison of threshold-based and adaptive gait-event detection
* subject-wise and study-wise validation of gait features

## Limitations

This project is exploratory.

Current limitations:

* Foot-contact detection uses a simple fixed threshold.
* Body weight, walking speed, age, and disease severity are not yet fully controlled.
* Some recordings may contain unusual signal patterns or outliers.
* Classification results should not be interpreted as a final diagnostic model.
* Study/source group effects may influence the results.

## Future Work

Possible next steps:

1. Add metadata such as age, gender, height, weight, UPDRS, and Hoehn & Yahr stage.
2. Normalize force features by body weight.
3. Compare fixed-threshold contact detection with adaptive thresholding.
4. Analyze disease severity instead of only Parkinson vs control.
5. Add study-holdout validation.
6. Build a small Streamlit dashboard for interactive signal and feature exploration.
7. Prepare a short thesis proposal based on the most promising feature groups.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```text
01_dataset_understanding.ipynb
02_foot_contact_detection.ipynb
03_feature_extraction.ipynb
04_exploratory_group_analysis.ipynb
05_statistical_analysis.ipynb
06_improved_gait_features.ipynb
07_improved_feature_analysis.ipynb
08_classification_baseline.ipynb
```

The raw dataset should be placed in:

```text
data/raw/
```

Processed outputs will be saved in:

```text
data/processed/
```

Figures will be saved in:

```text
results/figures/
```

## Project Status

Current status:

* Dataset loading completed
* Foot-contact detection completed
* Basic and improved feature extraction completed
* Exploratory analysis completed
* Statistical analysis completed
* Classification baseline completed
* README/reporting in progress

## Author

Masum Abid
Master's student in Medical Engineering
FAU Erlangen-Nürnberg
