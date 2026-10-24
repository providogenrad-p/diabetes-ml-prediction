# Diabetes Prediction – Pima Indians Diabetes Dataset

Predicting whether a patient has diabetes from routine medical measurements, and comparing five classification models against a simple baseline. Implemented in R.

## Dataset

The [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 female patients of Pima Indian heritage, aged 21 or older. Each patient has 8 numeric predictors and a binary target.

| Variable | Description |
| --- | --- |
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Score of family history of diabetes |
| Age | Age in years |
| Outcome | Target: 1 = diabetes, 0 = healthy |

The classes are imbalanced: 500 healthy (65%) and 268 diabetic (35%).

## Project structure

```
.
├── diabetes.csv            # dataset
├── diabetes_analysis.R     # full analysis script
└── README.md
```

## Requirements

R (version 4.0 or later) and the following packages:

```r
install.packages(c("corrplot", "caret", "rpart", "rpart.plot",
                   "randomForest", "class", "e1071", "pROC"))
```

## How to run

1. Place `diabetes.csv` in the same folder as `diabetes_analysis.R`.
2. Open the script in RStudio and set the working directory to that folder (Session → Set Working Directory → To Source File Location).
3. Run the whole script (Ctrl + Shift + Enter), or from a terminal:

   ```
   Rscript diabetes_analysis.R
   ```

The script prints all results to the console and draws the plots in the Plots pane. Random seeds are fixed (`set.seed(123)`), so the results are reproducible.

## Pipeline

1. **Data cleaning.** Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible, so they are treated as missing values (NA).
2. **Exploratory data analysis.** Outcome distribution, histograms, boxplots by outcome and a correlation matrix.
3. **Stratified split.** 70% training (538 patients) and 30% test (230 patients), with the same class ratio in both sets.
4. **Imputation.** Missing values are filled with medians computed on the training set only, to avoid data leakage.
5. **Modelling.** Five classifiers are trained and compared with a baseline:
   - Logistic Regression
   - Decision Tree (`rpart`)
   - Random Forest (500 trees)
   - k-NN (predictors standardized; k tuned with leave-one-out cross-validation, k = 21 selected)
   - Naive Bayes
   - ZeroR baseline (always predicts the majority class)
6. **Evaluation.** Confusion matrices, Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1 and AUC on the test set, plus ROC curves.

## Results

Test set (230 patients), sorted by AUC:

| Model | Accuracy | Balanced Acc. | Precision | Recall | Specificity | F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.770 | 0.727 | 0.701 | 0.588 | 0.867 | 0.639 | **0.851** |
| Random Forest | 0.765 | 0.727 | 0.686 | **0.600** | 0.853 | **0.640** | 0.834 |
| k-NN (k = 21) | 0.765 | 0.715 | 0.710 | 0.550 | 0.880 | 0.620 | 0.826 |
| Decision Tree | **0.778** | **0.728** | **0.738** | 0.562 | **0.893** | 0.638 | 0.824 |
| Naive Bayes | 0.726 | 0.691 | 0.613 | 0.575 | 0.807 | 0.594 | 0.818 |
| ZeroR (baseline) | 0.652 | 0.500 | – | 0.000 | 1.000 | – | 0.500 |

## Key findings

- **All five models clearly beat the baseline,** with AUC between 0.82 and 0.85 versus 0.50.
- **Logistic Regression is the recommended model.** It has the highest AUC (0.851) and is the easiest to interpret through its odds ratios.
- **The top four models are very close.** Their accuracy differs by at most 1.3 percentage points (3 test patients).
- **Glucose is the strongest predictor** in every analysis, followed by BMI and Age. Each extra 10 mg/dL of glucose raises the odds of diabetes by about 41%.
- **Recall is low for all models (0.55–0.60).** Roughly 40% of diabetic patients are missed at the default 0.5 threshold.

## Limitations

- Results come from a single train/test split; repeated cross-validation would give more reliable estimates.
- About 49% of Insulin and 30% of SkinThickness values are missing, and median imputation is a simple approach.
- All models use a fixed 0.5 probability threshold; a lower threshold would increase Recall for screening.
- The data covers only adult women of Pima Indian heritage, so results may not generalize to other populations.
