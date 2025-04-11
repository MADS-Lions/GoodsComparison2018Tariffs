ComparisonUSCanadianGoods2018Tariffs
==============================
# Introduction
The purpose of this library is to:

Compare the impact of 2018 Tariffs between US and Canadian goods and services using: <br><br>
    1. Visualizations <br>
    2. Differences in differences <br>
    3. Regression discontinuity <br>
    4. ARIMA model <br>

This package is meant to provide analysis and insights into tariffs for categories in the US and Canada so you can compare and contrast the impact of tariffs in 2018 on inflation CPI index for those categories. The analysis is provided in the notebooks and is short but is to provide evidence for the overall report. Visualizations bring support to our ideas and can be explored with streamlit. 


# Table of Contents

[Quickstart](#Quickstart) <br>
[ProjectOrganization](#ProjectOrganization) <br>
[Examples](#Examples) <br>
[ModelStructure](#ModelStructure) <br>


## Quickstart
From the main GOODSCOMPARISON2018TARIFFS folder run:
python3 -m streamlit run ./notebooks/Streamlit\ -\ visualization_exploration.py

use the Makefile to run the following:
1. make requirements
2. make pull_dvc
3. make run_notebook

## ProjectOrganization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make requirements` or `make pull_dvc`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   ├── Analysis_Visuals_and_Models: Groceries_and_Clothing&Footwear.ipynb - For Analysis of Groceries and Clothing&Footwear Categories for comparing American and Canadian felt impact from 2018 tariffs. Making a narrative about clothing and groceries and the impact of tariffs on them providing evidence with visuals regression discontinuity and differences in differences
    │   ├── Analysis_Visuals_and_Models: Shelter.ipynb - For Analysis of Shelter category for comparing American and Candian felt impact from 2018 tariffs
    │   ├── Data_Cleaning_and_Manipulation: Main_Datasets.ipynb - For data cleaning and manipulating the main datasets
    │   ├── Data_Exploration_and_Feature_Selection: CACPINormalDistributionTest.ipynb - For Canadian data exploration, normality testing, and feature selection on categories
    │   ├── Data_Exploration_and_Feature_Selection: USCPINormalDistributionTest.ipynb - For US data exploration, normality testing, and feature selection on categories
    │   ├── Streamlit: visualization_streamlit.py - for visualization of streamlit of data to garner insights for the user 
    │   ├── Visualization: Regression_Discontinuity_For_Items_Pre&Post_Tariffs.ipynb - visuals for potential RC diagrams
    │   ├── Visualization: US_tariff_tradewar_timeline.ipynb - visuals for showing the dates and CPI of the US tradewar in 2018
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── rc_difference.py - Visualization plots, Regression Discontinuity and Differences in Differences
    │       └── arima_model_function.py - Functions for PACF, ARIMA model
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Examples

Example of how to use regression_discontinuity and differences_in_differences

#Used case shows regression discontinuity model from May 2018 to December 2018 with a fuzzy regression with treatment period from 2018-07-01 to 2018-10-01 and differences in differences for two categories Vehicles and Education & Reading - if there is a second date it will average between the two dates in this case '2017-07-01' to '2017-10-01'

regression_discontinuity_model(df, 'Groceries, '2018-05-01', '2018-12-01', '2018-07-01', '2018-10-01', feature = 'Category', heteroskedasticity = 'HC3', fuzzy_sharp_omit = False)

differences_differences(df, 'Vehicles', 'Education & Reading', '2018-05-01', '2018-12-01', '2018-07-01', date4='2018-10-01', feature='Category', heteroskedasticity='HC3')

Example of how to use ARIMA model
arima_model(can_categories_df, category='Shelter', order = (10,1,2), tariff_date='2018-06-01', forecast_steps=8, in_sample_len=8)

Some examples from the research narrative for modelling categories with visuals, regression discontinuity, and ARIMA models

[NarrativeGroceriesClothing&Footwear]('./notebooks/Analysis_Visuals_and_Modes: Groceries_and_Clothing&Footwear.ipynb') <br>
[ModellingCanadaARIMAForAllCategories]('./notebooks/Modelling: Canada_Arima_Forecasts_9_main_categories.ipynb') <br>
[ModellingUSARIMAForAllCategories]('./notebooks/Modelling: US_arima_forecasts.ipynb') <br>











--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

