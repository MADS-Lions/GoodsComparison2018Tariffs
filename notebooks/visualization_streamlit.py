"This module is to present a visual with streamlit on ideas related to tariffs - comparing American and Canadian goods including Clothing&Footwear, Vehicles, Energy, and Groceries to see the impact tariffs had on their inflation"
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
from datetime import date

import streamlit as st

 # Enable Streamlit compatibility in Jupyter
st.set_page_config(layout="wide")

def plot_structure(df, category, date1, date2, feature = 'Category', goodtype = None, color = None, x_label = 'Date', y_label = '', title = '', lines_to_plot = []):
    """Plot structure of data for line plot for each category of vehicle, groceries, energy, and clothing and footwear
    Accepts Arguments:
     param::str::df: DataFrame which is the data to be plotted
     param::str::category: str which is the category to be plotted
     param::str::date1: str which is the start date for the plot
     param::str::date2: str which is the end date for the plot
     param::str::feature: str which is the feature to be plotted
     param::str::goodtype: str which is the type of good to be plotted
     param::str::color: str which is the color of the plot
     param::str::x_label: str which is the x-axis label
     param::str::y_label: str which is the y-axis label
     param::str::title: str which is the title of the plot
     param::list::lines_to_plot: list which is the list of lines to plot on the graph
    Returns:
     Returns::altairchart:: a line plot of the data for the category
    """
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

    y_values = [
        df_in_question.loc[df['REF_DATE'] == '2017-08-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2017-10-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2018-02-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2018-04-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2018-07-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2018-11-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2019-05-01', "VALUE"].iloc[0],
        df_in_question.loc[df['REF_DATE'] == '2019-09-01', "VALUE"].iloc[0]
    ]
    y_values_2 = [
        df_in_question.loc[df['REF_DATE'] == '2017-08-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2017-10-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2018-02-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2018-04-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2018-07-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2018-11-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2019-05-01', "VALUE"].iloc[0]-0.08,
        df_in_question.loc[df['REF_DATE'] == '2019-09-01', "VALUE"].iloc[0]-0.08

        
    ]
    
    df_text_arrow_df = pd.DataFrame({
        'x': ['2017-08-01', '2017-10-01', '2018-02-01', '2018-04-01', '2018-07-01', '2018-11-01', '2019-05-01', '2019-09-01'],
        'y': y_values,
        'y2': y_values_2,
        'text': ['Start of talks about intellectual property and end of trade talks with China', 'Results of Intellectual Inquiry', 'Tariffs on China by America', 'Tariffs on imported Canadian goods by America', 'China/Canada Tariffs on imported American goods', 'China has some tariff hikes', 'End of Canada/US Tariffs', 'Chinese exemptions for American products']
    })
    text_annotation = alt.Chart(df_text_arrow_df).mark_text(
        align='left',
        baseline='middle',
        dx=7
    ).encode(
        x='x',
        y='y',
        text='text'
    )
    arrow_annotation = alt.Chart(df_text_arrow_df).mark_text().encode(
        x='x',
        y='y2',
        text = alt.value('\u2191')
    )
    
    if len(lines_to_plot) == 0:
        chart_final = chart
    else:
        all_lines.extend([chart])
        chart_final = alt.layer(*all_lines)
    
    return chart_final + text_annotation + arrow_annotation

def plot_supply_and_demand(sales_df, product, date1 = '2018-05-01', date2 = '2018-09-01', principlestats_cat = 'Total inventory, estimated values of total inventory at end of the month', principlestats_cat2 = 'Unfilled orders, estimated values of orders at end of month'):
    """Plot supply and demand for Canada and USA
    Accepts arguments:
     param::str::sales_df: DataFrame which is the data to be plotted
     param::str::product: str which is the product to be plotted
     param::str::date1: str which is the start date for the plot
     param::str::date2: str which is the end date for the plot
     param::str::principlestats_cat: str which is the principlestats category to be plotted
     param::str::principlestats_cat2: str which is the principlestats category to be plotted
    Returns:
     Returns altairchart::a point plot of the supply and demand for the product
    """
    
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
    x=alt.X('REF_DATE', title='Date'),
    y='VALUE',
    color = alt.Color('PrincipleStats', legend = alt.Legend(title = 'Supply and Demand'))
    )
    return chart1


def regression_discontinuity_model(df, category, date1, date2, date3, date4 = None, feature = 'Category', goodtype = None, seasonality = None, period = [5, 7], heteroskedasticity = 'HC3', fuzzy_sharp_omit = False):
    """Regression Discontinuity model for the data
    Accepts:
     param::df::pandas dataframe: DataFrame which is the data to be plotted
     param::category::str: str which is the category to be plotted
     param::date1::str: str which is the start date for the plot
     param::date2::str: str which is the end date for the plot
     param::date3::str: str which is the date for the regression discontinuity
     param::date4::str: str which is the end date for the regression discontinuity
     param::feature::str: str which is the feature to be plotted
     param::goodtype::str: str which is the type of good to be plotted
     param::seasonality::boolean: bool which is the seasonality of the data
     param::period::list: list which is the period of the seasonality parameter
     param::heteroskedasticity::str: str which is the heteroskedasticity of the data
     param::fuzzy_sharp_omit::boolean: bool which plot trend and omits seasonality
    
    Returns:
        
     model: statsmodels.regression.linear_model.RegressionResultsWrapper, chart: altair chart, chart2: altair chart, chart3: altair chart
    """
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
        
    return chart4

def plot_individual_product(df, category, date1, date2):
    """Plot individual product for the data
    Accepts:
     param::df::pandas dataframe: DataFrame which is the data to be plotted
     param::category::str: str which is the category to be plotted
     param::date1::str: str which is the start date for the plot
     param::date2::str: str which is the end date for the plot
     
     Returns:
     chart: altair chart
    """
    df_final = df[(df['Category'] == category)&(df['REF_DATE']>=date1)&(df['REF_DATE']<=date2)]
    chart = alt.Chart(df_final).mark_line().encode(
        x=alt.X('REF_DATE', title = 'Date'),
        y='VALUE',
        color='Products and product groups'
    )
    return chart


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
options2 = ['Motor vehicle parts manufacturing [3363]', 'Food manufacturing [311]', 'Petroleum and coal product manufacturing [324]', 'Apparel manufacturing [315]']
options3 = ['Motor Vehicle Bodies, Trailers, and Parts', 'Food Products', 'Petroleum and Coal Products', 'Apparel']
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



if selected_option == 'Clothing & Footwear':
    chart_clothing_usa = regression_discontinuity_model(American_df, 'Clothing & Footwear', '2017-01-01', '2019-08-01', '2017-10-01', '2019-02-01', seasonality=True, fuzzy_sharp_omit = True)
    chart_clothing_can = regression_discontinuity_model(Canadian_df, selected_option, '2017-08-01', '2019-10-01', '2019-03-01', '2019-08-01', seasonality=True)

chart = plot_structure(American_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart2 = plot_structure(Canadian_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart = chart.configure_axis(grid=False).properties(width=4500, height = 650).interactive()
chart2 = chart2.configure_axis(grid=False).properties(width=4500, height = 650).interactive()
st.write("American Inflation for ", selected_option)
st.altair_chart(chart)
st.write("Canadian Inflation for ", selected_option)
st.altair_chart(chart2)
if selected_option == 'Clothing & Footwear':
    st.write("American Trend Inflation for ", selected_option)
    st.altair_chart(chart_clothing_usa)
    st.write("Canadian Trend Inflation for ", selected_option)
    st.altair_chart(chart_clothing_can)
st.write("All American products for category ", selected_option)
chart3 = plot_individual_product(American_df_2, selected_option, str(start_date), str(end_date))
chart4 = plot_individual_product(Canadian_df_2, selected_option, str(start_date), str(end_date))

chart3 = chart3.configure_axis(grid=False).properties(width=4500, height=650)
chart4 = chart4.configure_axis(grid=False).properties(width=4500, height=650)
st.altair_chart(chart3, use_container_width=True)
st.write("All Canadian products for category ", selected_option)

st.altair_chart(chart4, use_container_width=True)
for option, option2, option3 in zip(options, options2, options3):
    if selected_option == option:
        selected_option2 = option2
        selected_option3 = option3
if selected_option == 'Vehicles':
    chart5 = plot_supply_and_demand(American_supply_demand, selected_option3, str(start_date), str(end_date), principlestats_cat='New Orders Percent Change Monthly', principlestats_cat2='Total Inventories')
    
else:
    chart5 = plot_supply_and_demand(American_supply_demand, selected_option3, str(start_date), str(end_date), principlestats_cat='Finished Goods Inventories Percent Change Monthly', principlestats_cat2='Value of Shipments Percent Change Monthly')
chart6 = plot_supply_and_demand(Canadian_supply_demand, selected_option2, str(start_date), str(end_date))

chart5 = chart5.configure_axis(grid=False).properties(width=4500, height=650).interactive()
chart6 = chart6.configure_axis(grid=False).properties(width=4500, height=650).interactive()


st.write("Supply and Demand for ", selected_option3, " in the USA")
st.altair_chart(chart5, use_container_width=True)
st.write("Supply and Demand for ", selected_option2, " in Canada")
st.altair_chart(chart6, use_container_width=True)

