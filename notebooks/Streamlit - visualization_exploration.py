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
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
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

def plot_supply_and_demand(sales_df, product, date1 = '2018-05-01', date2 = '2018-09-01', principlestats_cat = 'Total inventory, estimated values of total inventory at end of the month', principlestats_cat2 = 'Unfilled orders, estimated values of orders at end of month', point_line='point', x_label = '', y_label = '', title = ''):
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
    
    if point_line == 'point':
        chart1 = alt.Chart(sales_df).mark_point().encode(
        x=alt.X('REF_DATE', title = x_label),
        y=alt.Y('VALUE', title = y_label),
        color = alt.Color('PrincipleStats', legend = alt.Legend(title = 'Supply and Demand'))
        )
    else: 
        chart1 = alt.Chart(sales_df).mark_line().encode(
        x=alt.X('REF_DATE', title = x_label),
        y=alt.Y('VALUE', title = y_label),
        color = alt.Color('PrincipleStats', legend = alt.Legend(title = 'Supply and Demand'))
        )
    return chart1.properties(title=title)


def regression_discontinuity_model(df, category, date1, date2, date3, date4 = None, feature = 'Category', goodtype = None, seasonality = None, period = [5, 7], heteroskedasticity = 'HC3', fuzzy_sharp_omit = False, point_line = 'line', x_label = '', y_label = ''):
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
    print("Product: ", category)
    df_in_question = df.copy()
    if goodtype ==None: 
        df_in_question = df_in_question[df_in_question[feature] == category]
    else:
        df_in_question = df_in_question[(df_in_question[feature] == category) & (df_in_question['GoodType'] == goodtype)]
    df_in_question = df_in_question.sort_values('REF_DATE')
    df_in_question['REF_DATE'] = pd.to_datetime(df_in_question['REF_DATE'])
    df_in_question['REF_DATE'] = df_in_question['REF_DATE'].dt.strftime('%Y-%m')
    df_in_question['y'] = [1.0]*len(df_in_question)
    df_in_question['text_1'] = ['First Tariff Period']*len(df_in_question)
    df_in_question['text_2'] = ['Second Tariff Period']*len(df_in_question)
    
    if seasonality == None:
        df_in_question['VALUE_DIFF'] = df_in_question['VALUE'].diff()
    
        df_in_question.dropna(subset=['VALUE_DIFF'], inplace=True)
        df_in_question['VALUE_DETREND'] = detrend(df_in_question['VALUE_DIFF'])
        df_in_question = df_in_question[(df_in_question['REF_DATE']>=date1) & (df_in_question['REF_DATE']<=date2)]
        if point_line =='line':
            chart = alt.Chart(df_in_question).mark_line().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE_DIFF', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
            chart2 = alt.Chart(df_in_question).mark_line().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title= y_label)),
                color=feature
            )
            chart3 = alt.Chart(df_in_question).mark_line().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE_DETREND', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
        else:
            chart = alt.Chart(df_in_question).mark_point().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE_DIFF', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
            chart2 = alt.Chart(df_in_question).mark_point().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
            chart3 = alt.Chart(df_in_question).mark_point().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('VALUE_DETREND', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
        
    else:
        df_value = df_in_question['VALUE']
        df_value.index = pd.to_datetime(df_in_question['REF_DATE'])
        
        season_model = STL(df_value, period = 7)
        season = season_model.fit()
        
        df_in_question['RESID'] = season.resid.tolist()

    
        
        df_in_question['TREND'] = season.trend.tolist()
        df_in_question['VALUE_TREND'] = df_in_question['TREND'].diff()
        
        df_in_question = df_in_question[(df_in_question['REF_DATE']>=date1) & (df_in_question['REF_DATE']<=date2)]
        if point_line =='line':
            
            chart4 = alt.Chart(df_in_question).mark_line().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('TREND', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
                color=feature
            )
        else:
            
            chart4 = alt.Chart(df_in_question).mark_point().encode(
                x=alt.X('REF_DATE', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = x_label)),
                y=alt.Y('TREND', axis = alt.Axis(tickCount = 5,labelFontSize=15, title = y_label)),
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


American_supply_demand = pd.read_csv('./data/processed/USA_Sales_Processed_Final.csv')
Canadian_supply_demand = pd.read_csv('./data/processed/Canada_Sales_Processed_Final.csv')
American_df = pd.read_csv('./data/processed/USA_CPI_Processed_2018_2019.csv')


Canadian_df = pd.read_csv('./data/processed/Canada_CPI_Processed_2018_2019.csv')
df_model_data_CAN = pd.read_csv('./data/processed/CAN_Categorized_Products_and_Services_NEW.csv')
df_model_data_USA = pd.read_csv('./data/processed/US_Categorized_Products_and_Services_NEW.csv')
dict_CAN = df_model_data_CAN.drop_duplicates(subset=['Product_Service']).set_index('Product_Service')['Category'].to_dict()
dict_USA = df_model_data_USA.drop_duplicates(subset=['Product_Service']).set_index('Product_Service')['Category'].to_dict()

American_df = pd.melt(American_df, var_name = 'Products and product groups',value_name = 'VALUE', id_vars = 'REF_DATE')

options = ['Groceries', 'Energy', 'Clothing and footwear']
options2 = ['Food manufacturing [311]', 'Petroleum and coal product manufacturing [324]', 'Apparel manufacturing [315]']
options3 = ['Food Products', 'Petroleum and Coal Products', 'Apparel']
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



if selected_option == 'Clothing and footwear':
    chart_clothing_usa = regression_discontinuity_model(American_df, 'Clothing and Footwear', '2017-01-01', '2019-08-01', '2017-10-01', '2019-02-01', seasonality=True, fuzzy_sharp_omit = True)
    chart_clothing_can = regression_discontinuity_model(Canadian_df, selected_option, '2017-08-01', '2019-10-01', '2019-03-01', '2019-08-01', seasonality=True)

chart = plot_structure(American_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart2 = plot_structure(Canadian_df, selected_option, str(start_date), str(end_date), x_label='Date', y_label="CPI Index for Inflation")
chart = chart.configure_axis(grid=False).properties(width=4500, height = 650).interactive()
chart2 = chart2.configure_axis(grid=False).properties(width=4500, height = 650).interactive()
st.write("American Inflation for ", selected_option)
st.altair_chart(chart)
st.write("Canadian Inflation for ", selected_option)
st.altair_chart(chart2)
if selected_option == 'Clothing and Footwear':
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

chart5 = plot_supply_and_demand(American_supply_demand, selected_option3, str(start_date), str(end_date), "Finished Goods Inventories", "Inventories to Shipments Ratios", x_label='Date', y_label = 'US Supply and Demand of Manufacturing for ' + selected_option3, point_line='line')
chart6 = plot_supply_and_demand(Canadian_supply_demand, selected_option2, str(start_date), str(end_date), x_label='Date', y_label = 'Canadian Supply and Demand of Manufacturing for ' + selected_option2, point_line='line')

chart5 = chart5.configure_axis(grid=False).properties(width=4500, height=650).interactive()
chart6 = chart6.configure_axis(grid=False).properties(width=4500, height=650).interactive()


st.write("Supply and Demand for ", selected_option3, " in the USA")
st.altair_chart(chart5, use_container_width=True)
st.write("Supply and Demand for ", selected_option2, " in Canada")
st.altair_chart(chart6, use_container_width=True)



US_sales_groceries_clothing = pd.read_csv("./data/raw/ClothingGroceriesUSSalesData.csv")
US_sales_groceries_clothing['Clothing Sales'] = US_sales_groceries_clothing['Clothing Sales'].str.replace(',', '')  
US_sales_groceries_clothing['Grocery Sales'] = US_sales_groceries_clothing['Grocery Sales'].str.replace(',', '')
US_sales_groceries_clothing['Gas Sales'] = US_sales_groceries_clothing['Gas Sales'].str.replace(',', '')
US_sales_groceries_clothing['Clothing Sales'] = US_sales_groceries_clothing['Clothing Sales'].astype(float)
US_sales_groceries_clothing['Grocery Sales'] = US_sales_groceries_clothing['Grocery Sales'].astype(float)
US_sales_groceries_clothing['Gas Sales'] = US_sales_groceries_clothing['Gas Sales'].astype(float)
US_sales_groceries_clothing['REF_DATE'] = pd.to_datetime(US_sales_groceries_clothing['Date'])
US_sales_groceries_clothing['Clothing Sales'] = scale.fit_transform(US_sales_groceries_clothing['Clothing Sales'].values.reshape(-1, 1))
US_sales_groceries_clothing['Grocery Sales'] = scale.fit_transform(US_sales_groceries_clothing['Grocery Sales'].values.reshape(-1, 1))
US_sales_groceries_clothing['Gas Sales'] = scale.fit_transform(US_sales_groceries_clothing['Gas Sales'].values.reshape(-1, 1))
US_sales_groceries = US_sales_groceries_clothing[['REF_DATE', 'Grocery Sales']]
US_sales_groceries['Category'] = ['Groceries']*len(US_sales_groceries)
US_sales_groceries['VALUE'] = US_sales_groceries['Grocery Sales']
Sales_Canada_Groceries = pd.read_csv('./data/raw/20100082.csv')
dict_month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
Sales_Canada_Groceries['Month'] = Sales_Canada_Groceries['REF_DATE'].str.split('-').str[0]
Sales_Canada_Groceries['Month'] = Sales_Canada_Groceries['Month'].map(dict_month)

Sales_Canada_Groceries['Year'] = ['20']*len(Sales_Canada_Groceries) + Sales_Canada_Groceries['REF_DATE'].str.split('-').str[1]
Sales_Canada_Groceries['Day'] = ['01']*len(Sales_Canada_Groceries)
Sales_Canada_Groceries['REF_DATE'] = pd.to_datetime(Sales_Canada_Groceries[['Year', 'Month', 'Day']])
Sales_Canada_Groceries['Category'] = Sales_Canada_Groceries['North American Industry Classification System (NAICS)']
if selected_option == 'Clothing and footwear':
    
    chart7 = alt.Chart(US_sales_groceries_clothing).mark_line().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('Clothing InventorySales', title = 'Inventory to Sales Ratio for Clothing in US')
    )
    Sales_Canada_Clothing = Sales_Canada_Groceries[(Sales_Canada_Groceries['REF_DATE'] >= pd.Timestamp(start_date)) & (Sales_Canada_Groceries['REF_DATE']<= pd.Timestamp(end_date)) & (Sales_Canada_Groceries['Category'] == 'Clothing and clothing accessories retailers [4581]')]
    scale = StandardScaler()
    Sales_Canada_Clothing['VALUE'] = scale.fit_transform(Sales_Canada_Clothing['VALUE'].to_numpy().reshape(-1,1))
    chart8 = alt.Chart(Sales_Canada_Clothing).mark_line().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('VALUE', title = 'Retail Sales of Clothing and Footwear in Canada')
    )
elif selected_option == 'Groceries':
    chart7 = alt.Chart(US_sales_groceries).mark_line().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('Grocery Sales', title = 'Sales for US Groceries')
    )
    Sales_Canada_Groc = Sales_Canada_Groceries[(Sales_Canada_Groceries['REF_DATE'] >= pd.Timestamp(start_date)) & (Sales_Canada_Groceries['REF_DATE']<= pd.Timestamp(end_date)) & (Sales_Canada_Groceries['Category'] == 'Supermarkets and other grocery retailers (except convenience retailers) [44511]')]
    scale = StandardScaler()
    Sales_Canada_Groc['VALUE'] = scale.fit_transform(Sales_Canada_Groc['VALUE'].to_numpy().reshape(-1,1))
    chart8 = alt.Chart(Sales_Canada_Groc).mark_point().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('VALUE', title = 'Retail Sales of Groceries in Canada')
    )

else: 
    chart7 = alt.Chart(US_sales_groceries_clothing).mark_line().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('Gas Sales', title = 'Sales for US Gas')
    )

    Sales_Canada_Gas = Sales_Canada_Groceries[(Sales_Canada_Groceries['REF_DATE'] >= pd.Timestamp(start_date)) & (Sales_Canada_Groceries['REF_DATE']<= pd.Timestamp(end_date)) & (Sales_Canada_Groceries['Category'] == 'Gasoline stations and fuel vendors [457]')]
    scale = StandardScaler()
    Sales_Canada_Gas['VALUE'] = scale.fit_transform(Sales_Canada_Gas['VALUE'].to_numpy().reshape(-1,1))
    
    chart8 = alt.Chart(Sales_Canada_Gas).mark_point().encode(
        x = alt.X('REF_DATE', title = 'Date'),
        y = alt.Y('VALUE', title = 'Retail Sales of Fuel and Gas in Canada')
    )

st.write("Sales Data for ", selected_option, " in the USA")
st.altair_chart(chart7, use_container_width=True)
st.write("Sales Data for ", selected_option, " in Canada")
st.altair_chart(chart8, use_container_width=True)
