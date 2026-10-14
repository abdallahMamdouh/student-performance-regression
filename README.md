# Project 9: Student Academic Performance Multi-Model Regression Benchmark

## Overview & Problem Statement
Understanding educational outcomes requires identifying how demographic factors, parental education, nutritional support, and test preparation interact to influence student performance. 

The objective of this project is to model and predict student quantitative achievement (standardized `math score`) based on sociodemographic indicators. We perform an exhaustive empirical benchmark comparing 7 distinct regression families—ranging from Ordinary Least Squares to Kernel Support Vector Machines and modern gradient boosted ensembles—evaluating their coefficient of determination ($R^2$).

---

## Dataset Description
- **Source**: Students Performance in Exams Dataset (`StudentsPerformance.csv`).
- **File**: `StudentsPerformance.csv`
- **Volume**: 1,000 observations, 8 attributes.
- **Target Variable**: `math score` — Continuous score ranging from 0 to 100.
- **Predictor Attributes**:
  - `gender`: Student gender (`female`, `male`).
  - `race/ethnicity`: Anonymized demographic groups (Group A through E).
  - `parental level of education`: Highest education attained (`some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, `master's degree`).
  - `lunch`: Lunch subsidy type (`standard`, `free/reduced`) as an economic indicator.
  - `test preparation course`: Status of preparatory course completion (`none`, `completed`).
  - `reading score`: Continuous reading comprehension score (0 to 100).
  - `writing score`: Continuous writing evaluation score (0 to 100).

---

## Methodology & Machine Learning Workflow

### 1. Feature Engineering & Preparation
- Verified zero missing values and data consistency.
- Encoded categorical predictors (`gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`) using dummy/one-hot encoding.
- Separated features ($X$) and continuous target ($y = \text{math score}$).
- Partitioned dataset into 80% train and 20% holdout test partitions (`test_size=0.2`).

### 2. Multi-Model Regression Benchmark
Benchmarked 7 regression algorithms under uniform evaluation conditions:
1. **Linear Regression (LR)**: Ordinary Least Squares (OLS) estimating direct linear feature coefficients.
2. **Support Vector Regressor (SVM)**: SVR with epsilon-insensitive loss and kernel mapping.
3. **K-Nearest Neighbors Regressor (KNN)**: Distance-weighted local neighborhood interpolation.
4. **Decision Tree Regressor (DT)**: Single CART regression tree minimizing variance reduction.
5. **Random Forest Regressor (RF)**: Ensemble of bootstrap aggregated randomized regression trees.
6. **AdaBoost Regressor**: Sequential adaptive boosting with weighted regression residuals.
7. **XGBoost Regressor**: Scalable tree boosting minimizing squared error loss.

---

## Benchmark Results & Model Comparison

Evaluated on the holdout test set using the Coefficient of Determination ($R^2$):

| Rank | Model Family | Algorithm | $R^2$ Score | Variance Explained |
| :---: | :--- | :--- | :---: | :---: |
| 🥇 **1** | **Linear Models** | **Linear Regression (LR)** | **0.8854** | **88.54%** |
| 🥈 **2** | **Boosting Ensembles** | **AdaBoost Regressor** | **0.8546** | **85.46%** |
| 🥉 **3** | **Bagging Ensembles** | **Random Forest Regressor** | **0.8497** | **84.97%** |
| 4 | **Boosting Ensembles** | **XGBoost Regressor** | **0.8313** | **83.13%** |
| 5 | **Decision Trees** | **Decision Tree Regressor** | **0.7433** | **74.33%** |
| 6 | **Instance-based** | **K-Nearest Neighbors (KNN)** | **0.6734** | **67.34%** |
| 7 | **Kernel Methods** | **Support Vector Regressor (SVM)** | **0.6201** | **62.01%** |

### Key Engineering Insights:
- **Linear Regression achieves the highest performance ($R^2 = 0.8854$)**, outperforming complex ensembles and non-linear kernels.
- **Why?** Strong, near-linear collinear relationships exist between `reading score`, `writing score`, and `math score`. Complex non-linear models introduce slight overfitting and boundary variance, whereas OLS generalizes with minimal variance error.

---

## Quickstart Guide

### 1. Environment Setup
Navigate to the project folder:
```bash
cd 09-student-performance-regression
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Running the Benchmark Notebook
Launch Jupyter:
```bash
jupyter notebook Regression-Project.ipynb
```
Run all cells to execute feature preparation, train all 7 regressors, and generate the $R^2$ comparison dictionary and rankings.
