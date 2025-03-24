ComparisonUSCanadianGoods2018Tariffs
==============================
The purpose of this library is to:

Compare the impact of 2018 Tariffs between US and Canadian goods and services using:
    1. visualizations
    2. differences in differences
    3. regression discontinuity

This package is meant to provide analysis and insights into tariffs for categories in the US and America so you can compare and contrast the impact of tariffs in 2018 on inflation CPI index for those categories. The analysis is provided in the notebooks and is short but is to provide evidence for the overall report. Visualizations bring support to our ideas and can be explored with streamlit. 


# Table of Contents

[Quickstart](#Quickstart) <br>
[ProjectOrganization](#ProjectOrganization) <br>
[Examples](#Examples) <br>
[ModelStructure](#ModelStructure) <br>


## Quickstart
python3 -m streamlit run ./notebooks/visualization_streamlit.py


## ProjectOrganization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
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
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

# Examples

Example of how to use functions 

#Used case shows regression discontinuity model from May 2018 to December 2018 with a fuzzy regression with treatment period from 2018-07-01 to 2018-10-01 and differences in differences for two categories Vehicles and Education & Reading - if there is a second date it will average between the two dates in this case '2017-07-01' to '2017-10-01'

regression_discontinuity_model(df, 'Groceries, '2018-05-01', '2018-12-01', '2018-07-01', '2018-10-01', feature = 'Category', heteroskedasticity = 'HC3', fuzzy_sharp_omit = False)

differences_differences(df, 'Vehicles', 'Education & Reading', '2018-05-01', '2018-12-01', '2018-07-01', date4='2018-10-01', feature='Category', heteroskedasticity='HC3')

Some research narrative with used case examples on Groceries and Clothing & Footwear

[NarrativeGroceriesClothing&Footwear]('./notebooks/Official_Differences_and_RC.ipynb') <br>


# ModelStructure







--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

