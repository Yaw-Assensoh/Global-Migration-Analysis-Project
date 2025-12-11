# Global Migration Analysis Project

A complete data pipeline for cleaning, analyzing, and visualizing global migration and population statistics. This project transforms raw demographic data into actionable insights through a structured processing workflow and interactive web application.

##  Project Overview

This project addresses the challenge of working with heterogeneous global demographic data by providing:
1. **Data Cleaning Pipeline**: Processes raw population/migration datasets into consistent, analysis-ready formats
2. **Interactive Dashboard**: A Streamlit web app for exploring cleaned data with visualizations and filters
3. **Analytical Insights**: Key metrics and visualizations for understanding global migration patterns

##  Live Application

**Access the live dashboard here:**  
👉 [https://yaw-assensoh-global-migration-analysis-project-app-tbxmqx.streamlit.app](https://yaw-assensoh-global-migration-analysis-project-app-tbxmqx.streamlit.app)


### Key Components
- **`cleaned_migration_data.csv`**: Standardized migration statistics for 233 countries
- **`cleaned_population_data.csv`**: Comprehensive population/demographic indicators
- **`app.py`**: Streamlit application serving the interactive dashboard
- **Data Processing Scripts**: Notebooks/scripts that transform raw source data into clean formats

##  Key Features

### 1. **Data Standardization**
- Unified schema across multiple data sources
- Consistent column naming and data types
- Derived metrics (Migration Rate per 1000, Migration Status categories)
- Geographic classification (Continent assignment)

### 2. **Interactive Dashboard**
- **Country Comparison**: Side-by-side metrics for any two countries
- **Migration Analysis**: Filter and sort by migration rates, net migrants, population
- **Demographic Insights**: Fertility rates, median age, urbanization correlations
- **Geographic Views**: Continent-based filtering and aggregation
- **Top Rankings**: Identify leaders in immigration/emigration, population growth

### 3. **Analytical Metrics**
- **Migration Rate per 1000**: Standardized measure of migration intensity
- **Migration Status Classification**: High/Moderate immigration/emigration categories
- **Growth Rate Analysis**: Annual population change percentages
- **Density Calculations**: Population per square kilometer
- **World Share Percentages**: Each country's proportion of global population

## 🔧 Data Schema



### Derived Metrics
- **`Migration_Rate_per_1000`**: `(Net_Migrants / Population) × 1000`
- **`Migration_Status`**: Classification based on migration intensity
- **`Growth_Rate`**: Calculated population growth metrics

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Pandas, Plotly, NumPy


