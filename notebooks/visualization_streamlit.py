import altair as alt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import numpy as np
import altair as alt
from scipy.signal import detrend
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import seasonal_decompose, MSTL

import streamlit as st

 # Enable Streamlit compatibility in Jupyter


def plot_structure(df, category, date1, date2, feature = 'Category', goodtype = None, color = None, x_label = 'Date', y_label = '', title = '', lines_to_plot = []):
    print("Product: ", category)
    df_in_question = df.copy()
    if len(lines_to_plot) == 0:
        pass
    else:
        all_lines = []
        for line in lines_to_plot:
        
            vertical_line = alt.Chart(pd.DataFrame({'x': [line]})).mark_rule(
                color='black',
                strokeDash = [10, 5],
                strokeWidth = 2).encode(
                x='x'
            )
            all_lines.append(vertical_line)
    if goodtype == None:
        df_in_question = df_in_question[df_in_question[feature] == category]
    else:
        df_in_question = df_in_question[(df_in_question[feature] == category) & (df_in_question['GoodType'] == goodtype)]
    df_in_question = df_in_question[(df_in_question['REF_DATE']>=date1) & (df_in_question['REF_DATE']<=date2)]
    df_in_question = df_in_question.sort_values('REF_DATE')
    if color == None:
        chart = alt.Chart(df_in_question).mark_line().encode(
            x=alt.X('REF_DATE', title = x_label),
            y=alt.Y('VALUE', title = y_label),
            color=feature
        )
    else:
        chart = alt.Chart(df_in_question).mark_line().encode(
            x=alt.X('REF_DATE', axis = alt.Axis(title = x_label)),
            y=alt.Y('VALUE', axis = alt.Axis(title = y_label)),
            color=color
        )

    
    if len(lines_to_plot) == 0:
        chart_final = chart
    else:
        all_lines.extend([chart])
        chart_final = alt.layer(*all_lines)
    return chart_final

def plot_supply_and_demand_canada(sales_df, product, date1 = '2018-05-01', date2 = '2018-09-01', principlestats_cat = 'Total inventory, estimated values of total inventory at end of the month', principlestats_cat2 = 'Unfilled orders, estimated values of orders at end of month'):
    sales_df = sales_df[(sales_df['GoodType'] == product) & ((sales_df['PrincipleStats'] == principlestats_cat) | (sales_df['PrincipleStats'] == principlestats_cat2))].copy()
    
    scaler = StandardScaler()
    scaler2 = StandardScaler()
    mask1 = (sales_df['PrincipleStats']==principlestats_cat)
    mask2 = (sales_df['PrincipleStats']==principlestats_cat2)
    print(sales_df.loc[mask1, 'VALUE'].head())
    
    scaler.fit(sales_df[mask1]['VALUE'].to_numpy().reshape(-1,1))
    scaler2.fit(sales_df[mask2]['VALUE'].to_numpy().reshape(-1,1))
    sales_df.loc[mask1, "VALUE"] = scaler.transform(sales_df.loc[mask1, 'VALUE'].values.reshape(-1, 1))
    sales_df.loc[mask2, "VALUE"] = scaler2.transform(sales_df.loc[mask2, 'VALUE'].values.reshape(-1, 1))
    sales_df = sales_df[(sales_df['REF_DATE']>=date1) & (sales_df['REF_DATE']<=date2)]
    
    chart1 = alt.Chart(sales_df).mark_point().encode(
    x='REF_DATE',
    y='VALUE',
    color = alt.Color('PrincipleStats', legend = alt.Legend(title = 'Supply and Demand'))
    )
    return chart1


def regression_discontinuity_model(df, category, date1, date2, date3, date4 = None, feature = 'Category', goodtype = None, seasonality = None, period = [5, 7], heteroskedasticity = 'HC3', fuzzy_sharp_omit = False):
    print("Product: ", category)
    df_in_question = df.copy()
    if goodtype ==None: 
        df_in_question = df_in_question[df_in_question[feature] == category]
    else:
        df_in_question = df_in_question[(df_in_question[feature] == category) & (df_in_question['GoodType'] == goodtype)]
    df_in_question = df_in_question.sort_values('REF_DATE')
    if seasonality == None:
        df_in_question['VALUE_DIFF'] = df_in_question['VALUE'].diff()
    
        df_in_question.dropna(subset=['VALUE_DIFF'], inplace=True)
        df_in_question['VALUE_DETREND'] = detrend(df_in_question['VALUE_DIFF'])
        df_in_question = df_in_question[(df_in_question['REF_DATE']>=date1) & (df_in_question['REF_DATE']<=date2)]
        chart = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE_DIFF',
            color=feature
        )
        chart2 = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE',
            color=feature
        )
        chart3 = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE_DETREND',
            color=feature
        )
        display(chart2)
        display(chart)
        display(chart3)
    else:
        df_value = df_in_question['VALUE']
        df_value.index = pd.to_datetime(df_in_question['REF_DATE'])
        
        season_model = MSTL(df_value, periods=period).fit()
        
        df_in_question['VALUE_SEASON'] = season_model.seasonal[season_model.seasonal.columns[0]].tolist()
        df_in_question['RESID'] = season_model.resid.tolist()

        df_in_question['VALUE_DIF'] = df_in_question['VALUE'].values - df_in_question['VALUE_SEASON'].values
        
        df_in_question['TREND'] = season_model.trend.tolist()
        df_in_question['VALUE_TREND'] = df_in_question['TREND'].diff()
        df_in_question['VALUE_DIFF'] = df_in_question['VALUE_DIF'].diff()
        df_in_question.dropna(subset=['VALUE_DIFF'], inplace=True)
        df_in_question['VALUE_DETREND'] = detrend(df_in_question['VALUE_DIFF'])
        df_in_question = df_in_question[(df_in_question['REF_DATE']>=date1) & (df_in_question['REF_DATE']<=date2)]
        chart = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE_DIFF',
            color=feature
        )
        chart2 = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE',
            color=feature
        )
        chart3 = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='VALUE_DETREND',
            color=feature
        )
        chart4 = alt.Chart(df_in_question).mark_line().encode(
            x='REF_DATE',
            y='TREND',
            color=feature
        )
        display(chart2)
        display(chart)
        display(chart3)
        display(chart4)

from datetime import date
American_supply_demand = pd.read_csv('../data/processed/USA_Sales_Processed_Final.csv')
Canadian_supply_demand = pd.read_csv('../data/processed/Canada_Sales_Processed_Final.csv')
American_df = pd.read_csv('../data/processed/USA_CPI_Processed_2018_2019.csv')


Canadian_df = pd.read_csv('../data/processed/Canada_CPI_Processed_2018_2019.csv')
df_model_data_CAN = pd.read_csv('../data/processed/CAN_Categorized_Products_and_Services.csv')
df_model_data_USA = pd.read_csv('../data/processed/US_Categorized_Products_and_Services.csv')
dict_CAN = df_model_data_CAN.drop_duplicates(subset=['Product_Service']).set_index('Product_Service')['Category'].to_dict()
dict_USA = df_model_data_USA.drop_duplicates(subset=['Product_Service']).set_index('Product_Service')['Category'].to_dict()

American_df = pd.melt(American_df, var_name = 'Products and product groups',value_name = 'VALUE', id_vars = 'REF_DATE')

options = ['Vehicles', 'Groceries', 'Energy', 'Clothing & Footwear']
options2 = ['Motor vehicle parts manufacturing [3363]', 'Petroleum and coal product manufacturing [324]', 'Food manufacturing [311]', 'Apparel manufacturing [315]']
options3 = ['Motor Vehicle Bodies, Trailers, and Parts', 'Petroleum and Coal Products', 'Food Products', 'Apparel']
start_date, end_date = st.date_input("Select a date range", value = (date(2017,1,1), date(2020,2,1)))
selected_option = st.selectbox('Select a category', options)
Canadian_df['Category'] = Canadian_df['Products and product groups'].map(dict_CAN)
American_df['Category'] = American_df['Products and product groups'].map(dict_USA)
scale = StandardScaler()
scale2 = StandardScaler()
for product in Canadian_df['Products and product groups'].unique():
    mask = Canadian_df['Products and product groups'] == product
    scale.fit(Canadian_df[mask]['VALUE'].to_numpy().reshape(-1,1))
    Canadian_df.loc[mask, 'VALUE'] = scale.transform(Canadian_df[mask]['VALUE'].to_numpy().reshape(-1,1))
for product in American_df['Products and product groups'].unique():
    mask = American_df['Products and product groups'] == product
    scale2.fit(American_df[mask]['VALUE'].to_numpy().reshape(-1,1))
    American_df.loc[mask, 'VALUE'] = scale2.transform(American_df[mask]['VALUE'].to_numpy().reshape(-1,1))
American_df_2 = American_df.copy()
Canadian_df_2 = Canadian_df.copy()
American_df.drop(columns = 'Products and product groups', inplace = True)
Canadian_df.drop(columns = 'Products and product groups', inplace = True)
American_df = American_df.groupby(['REF_DATE', 'Category']).median().reset_index()
Canadian_df = Canadian_df.groupby(['REF_DATE', 'Category']).median().reset_index()

def plot_individual_product(df, category, date1, date2):
    df_final = df[(df['Category'] == category)&(df['REF_DATE']>=date1)&(df['REF_DATE']<=date2)]
    chart = alt.Chart(df_final).mark_line().encode(
        x=alt.X('REF_DATE', title = 'Date'),
        y='VALUE',
        color='Products and product groups'
    )
    return chart
chart = plot_structure(American_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart2 = plot_structure(Canadian_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart = chart.configure_axis(grid=False)
chart2 = chart2.configure_axis(grid=False)
st.write("American Inflation for ", selected_option)
st.altair_chart(chart)
st.write("Canadian Inflation for ", selected_option)
st.altair_chart(chart2)

st.write("All American products for category ", selected_option)
chart3 = plot_individual_product(American_df_2, selected_option, str(start_date), str(end_date))
chart4 = plot_individual_product(Canadian_df_2, selected_option, str(start_date), str(end_date))

chart3 = chart3.configure_axis(grid=False)
chart4 = chart4.configure_axis(grid=False)
st.altair_chart(chart3)
st.write("All Canadian products for category ", selected_option)

st.altair_chart(chart4)
for option, option2, option3 in zip(options, options2, options3):
    if selected_option == option:
        selected_option2 = option2
        selected_option3 = option3
if selected_option == 'Vehicles':
    chart5 = plot_supply_and_demand_canada(American_supply_demand, selected_option3, str(start_date), str(end_date), principlestats_cat='New Orders Percent Change Monthly', principlestats_cat2='Total Inventories')
    
else:
    chart5 = plot_supply_and_demand_canada(American_supply_demand, selected_option3, str(start_date), str(end_date), principlestats_cat='Finished Goods Inventories Percent Change Monthly', principlestats_cat2='Value of Shipments Percent Change Monthly')
chart6 = plot_supply_and_demand_canada(Canadian_supply_demand, selected_option2, str(start_date), str(end_date))


st.write("Supply and Demand for ", selected_option3, " in the USA")
st.altair_chart(chart5)
st.write("Supply and Demand for ", selected_option2, " in Canada")
st.altair_chart(chart6)