# Electricity Demand Forecasting Using Statistical, Machine Learning and Deep Learning Models

## Project Overview

This project develops a complete electricity demand forecasting pipeline using statistical forecasting, machine learning and deep learning techniques. The objective is to forecast future electricity demand using historical electricity consumption data while comparing the performance of multiple forecasting models.

The project automatically performs data loading, preprocessing, exploratory data analysis, feature engineering, model training, forecasting, evaluation and visualization. All generated outputs including figures, forecasts, trained models and evaluation metrics are automatically saved into their respective output folders.

---

# Project Objectives

The objectives of this project are:

- Load and clean raw electricity demand data.
- Perform complete exploratory data analysis.
- Create time-based and lag-based forecasting features.
- Implement benchmark forecasting models.
- Implement SARIMA forecasting.
- Implement Random Forest Regression.
- Implement Gradient Boosting Regression.
- Implement Bayesian Ridge Regression.
- Implement Long Short-Term Memory (LSTM) neural network.
- Compare forecasting performance using multiple evaluation metrics.

---

# Dataset

Dataset Used

time_series_60min_singleindex.csv

Location

```
data/raw/time_series_60min_singleindex.csv
```

Dataset Frequency

Hourly Electricity Demand

Dataset Size

- Original Rows : 50,401
- Cleaned Rows : 50,400

Target Variable

```
electricity_demand
```

---

# Data Processing

The preprocessing pipeline performs the following tasks automatically.

- Loads the raw dataset
- Converts timestamps to datetime format
- Sets datetime index
- Removes invalid observations
- Handles missing values
- Resamples data into

    - Hourly
    - Daily
    - Weekly
    - Monthly
    - Yearly

- Saves processed datasets

Generated processed datasets are stored in

```
data/processed/
```

---

# Exploratory Data Analysis

The project automatically performs complete exploratory data analysis.

Generated figures include

- Missing Values
- Time Series Plot
- Distribution Plot
- Box Plot
- Monthly Box Plot
- Yearly Box Plot
- Hourly Demand Pattern
- Weekday Pattern
- Monthly Trend
- Yearly Trend
- Correlation Heatmap
- Rolling Mean
- Rolling Standard Deviation
- Seasonal Decomposition
- ACF Plot
- PACF Plot
- QQ Plot

Statistical Tests

- Augmented Dickey Fuller Test
- KPSS Test

EDA outputs are saved in

```
outputs/figures/
```

Statistical reports are saved in

```
outputs/metrics/
```

---

# Feature Engineering

The following forecasting features are automatically generated.

Calendar Features

- Year
- Month
- Day
- Hour
- Week
- Quarter
- Day of Week
- Day of Year

Lag Features

- Lag 1
- Lag 24
- Lag 48
- Lag 168

Rolling Features

- Rolling Mean
- Rolling Standard Deviation
- Rolling Minimum
- Rolling Maximum

Cyclic Features

- Hour Sine
- Hour Cosine
- Month Sine
- Month Cosine
- Day Sine
- Day Cosine

The final feature engineered dataset contains

```
50,232 rows
33 columns
```

---

# Forecasting Models

The following forecasting models are implemented.

## Benchmark Models

- Mean Forecast
- Naive Forecast
- Seasonal Naive Forecast
- Drift Forecast

## Statistical Model

- SARIMA

## Machine Learning Models

- Random Forest Regressor
- Gradient Boosting Regressor
- Bayesian Ridge Regression

## Deep Learning Model

- Long Short-Term Memory (LSTM)

---

# Evaluation Metrics

Each model is evaluated using

- RMSE
- MAE
- MAPE
- MSE
- R² Score
- Bias

The comparison table is automatically generated.

---

# Project Structure

```
electricity-demand-forecasting/

│

├── data/

│   ├── raw/

│   ├── interim/

│   └── processed/

│

├── outputs/

│   ├── figures/

│   ├── forecasts/

│   ├── metrics/

│   └── model_objects/

│

├── reports/

│

├── scripts/

│

├── src/

│

└── tests/
```

---

# Installation

Create a virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Install required packages

```
pip install -r requirements.txt
```

---

# Running the Project

Step 1

Verify dataset

```
python scripts/download_data.py
```

Step 2

Create forecasting features

```
python scripts/make_features.py
```

Step 3

Run complete forecasting pipeline

```
python scripts/run_pipeline.py
```

Step 4

Evaluate all forecasting models

```
python scripts/evaluate_models.py
```

---

# Generated Outputs

The project automatically saves

Processed Data

```
data/processed/
```

EDA Figures

```
outputs/figures/
```

Forecast CSV Files

```
outputs/forecasts/
```

Evaluation Metrics

```
outputs/metrics/
```

Trained Models

```
outputs/model_objects/
```

---

# Final Model Performance

The final forecasting models are ranked according to prediction accuracy.

| Model | RMSE | MAE | MAPE | R² |
|-------|------|------|------|------|
| Random Forest | 469.23 | 378.55 | 0.70 | 0.9974 |
| Gradient Boosting | 512.51 | 392.81 | 0.73 | 0.9969 |
| Bayesian Ridge | 876.15 | 695.84 | 1.32 | 0.9910 |
| LSTM | 2153.48 | 1694.86 | 2.87 | 0.9535 |
| Seasonal Naive | 3006.76 | 2318.52 | 4.41 | 0.5268 |
| Mean | 4397.30 | 3788.83 | 6.97 | -0.0121 |
| Naive | 4459.11 | 3783.20 | 6.79 | -0.0408 |
| Drift | 5117.96 | 4339.89 | 8.05 | -0.3710 |
| SARIMA | 5372.87 | 4540.88 | 8.57 | -0.5110 |

The Random Forest model achieved the highest forecasting accuracy, followed by Gradient Boosting and Bayesian Ridge Regression. The LSTM model achieved competitive performance but required significantly higher computational resources. Traditional benchmark forecasting models and the SARIMA model showed comparatively lower predictive accuracy for the selected dataset.

---

# Anil Kumar

MSc Data Science Project

Electricity Demand Forecasting Using Statistical, Machine Learning and Deep Learning Models

2026