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


def plot_structure(df, category, date1, date2, feature = 'Category', goodtype = None, color = None, x_label = 'Date', y_label = '', title = '', lines_to_plot = []):
    """plot the line graph structure of the data for a given cateogry between two dates
     Parameters Arguments:
      Accepts: df: DataFrame, category: str, date1: str, date2: str, feature: str, goodtype: str, color: str, x_label: str, y_label: str, title: str, lines_to_plot: list
      Returns: chart_final: altair chart
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

    
    if len(lines_to_plot) == 0:
        chart_final = chart
    else:
        all_lines.extend([chart])
        chart_final = alt.layer(*all_lines)
    return chart_final

def plot_supply_and_demand_canada(sales_df, product, date1 = '2018-05-01', date2 = '2018-09-01', principlestats_cat = 'Total inventory, estimated values of total inventory at end of the month', principlestats_cat2 = 'Unfilled orders, estimated values of orders at end of month'):
    """Plot the supply and demand of a product in Canada
     Parameters Arguments:
      Accepts: sales_df: DataFrame, product: str, date1: str, date2: str, principlestats_cat: str, principlestats_cat2: str
      Returns: chart1: altair chart
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
    x='REF_DATE',
    y='VALUE',
    color = alt.Color('PrincipleStats', legend = alt.Legend(title = 'Supply and Demand'))
    )
    return chart1


def regression_discontinuity_model(df, category, date1, date2, date3, date4 = None, feature = 'Category', goodtype = None, seasonality = None, period = [5, 7], heteroskedasticity = 'HC3', fuzzy_sharp_omit = False):
    """Regression Discontinuity Model
     Parameters Arguments:
      Accepts: df: DataFrame, category: str, date1: str, date2: str, date3: str, date4: str, feature: str, goodtype: str, seasonality: str, period: list, heteroskedasticity: str, fuzzy_sharp_omit: bool
      Returns: model: statsmodels.regression.linear_model.RegressionResultsWrapper, chart: altair chart, chart2: altair chart, chart3: altair chart
    """
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
    
    
    df_in_question['treat'] = np.where((df_in_question['REF_DATE']>=date3) & (df_in_question['REF_DATE']<=date4), 1, 0)
    if date4==None:
        df_in_question['above_or_below'] = np.where(df_in_question['REF_DATE']>date3, 1, 0)
    else:
        df_in_question['above_or_below'] = np.where(df_in_question['REF_DATE']>date4, 1, 0)
    
    df_in_question = df_in_question.sort_values(by=['Category', 'REF_DATE'])
    
    
    dict_replace = {unique_date: count for count, unique_date in enumerate(df_in_question['REF_DATE'].unique())}
    
    df_in_question['Date_Replaced'] = df_in_question['REF_DATE'].map(dict_replace)
    final_date = None 
    for date, num_date in zip(df_in_question['REF_DATE'].iloc[:], df_in_question['Date_Replaced'].iloc[:]):
        
        if date == date3:
            final_date = num_date
    
    
                
    df_in_question['Num_Date'] = df_in_question['Date_Replaced'] - final_date
    
    df_in_question['Num_Date'] = df_in_question['Num_Date'].astype(float)
    if seasonality == None:
        if date4==None:
            model = smf.ols(formula = 'VALUE_DETREND ~ above_or_below + Num_Date + above_or_below:Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
        else:
        #first_stage_model
            
            first_stage_model = smf.ols(formula = 'treat ~ above_or_below + Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
            print(first_stage_model.summary())
            
            df_in_question['PredTreatment'] = first_stage_model.predict(df_in_question)
            
            
            model = smf.ols(formula = 'VALUE_DETREND ~ PredTreatment + Num_Date + PredTreatment:Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
    else:
        if date4==None:
            model = smf.ols(formula = 'VALUE_TREND ~ above_or_below + Num_Date + above_or_below:Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
        else:
        #first_stage_model
            
            first_stage_model = smf.ols(formula = 'treat ~ above_or_below + Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
            print(first_stage_model.summary())
            
            df_in_question['PredTreatment'] = first_stage_model.predict(df_in_question)
            
            
            model = smf.ols(formula = 'VALUE_TREND ~ PredTreatment + Num_Date + PredTreatment:Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)

    if fuzzy_sharp_omit == True:
        df_in_question = df_in_question[(df_in_question['REF_DATE']>=date4) | (df_in_question['REF_DATE']<=date3)]
        
        model = smf.ols(formula = 'VALUE_TREND ~ above_or_below + Num_Date + above_or_below:Num_Date', data=df_in_question).fit(cov_type=heteroskedasticity)
    else:
        pass

    if seasonality == None:
        return (model, chart, chart2, chart3)
    else:
        return (model, chart, chart2, chart3, chart4)
    

def differences_differences(df, category1, category2, date1, date2, date3, date4=None, feature='Category', heteroskedasticity='HC3'):
    """Differences in Differences Model
     Parameters Arguments:
      Accepts: df: DataFrame, category1: str, category2: str, date1: str, date2: str, date3: str, date4: str, feature: str, heteroskedasticity: str
      Returns: model: statsmodels.regression.linear_model.RegressionResultsWrapper
    """
    df_in_question = df.copy()
    
    
    df_differences_differences = df_in_question[(df_in_question[feature]==category1) | (df_in_question[feature]==category2)]
    df_differences_differences = df_differences_differences[(df_differences_differences['REF_DATE']<=date2)&(df_differences_differences['REF_DATE']>=date1)]
    scale = StandardScaler()
        
    df_differences_differences = df_differences_differences[(df_differences_differences[feature]==category1)|(df_differences_differences[feature]==category2)]
    
    if date4==None:
        pass
    else:
        mask = (df_differences_differences['REF_DATE']<=date4)&(df_differences_differences['REF_DATE']>=date3)
        mask2 = df_differences_differences['REF_DATE']==date3
        new_value = df_differences_differences[mask]['VALUE'].mean()
        df_differences_differences.loc[mask2, "VALUE"] = new_value
        df_differences_differences = df_differences_differences[(df_differences_differences['REF_DATE']<=date3) | (df_differences_differences['REF_DATE']>date4)]

    df_differences_differences['VALUE'] = scale.fit_transform(df_differences_differences['VALUE'].values.reshape(-1, 1))
    
    
    df_differences_differences['post'] = np.where(df_differences_differences['REF_DATE']>date3, 1, 0)
    df_differences_differences['tariff_non_tariffed'] = np.where(df_differences_differences['Category']==category1, 1, 0)
    
    model = smf.ols(formula = 'VALUE ~  tariff_non_tariffed + post + tariff_non_tariffed:post', data=df_differences_differences).fit(cov_type=heteroskedasticity).summary()
    
    
    return model

#test parallel trends assumption:
def plot_for_parallel_trends(df, date1, date2, category_tariff, category_non_tariff, category_3 = None):
    """Plot for Parallel Trends
     Parameters Arguments:
      Accepts: df: DataFrame, date1: str, date2: str, category_tariff: str, category_non_tariff: str, category_3: str
      Returns: chart1: altair chart
    """
    df_Canada_CPI_Scaled_US_on_Canada_Tariffs = df[(df['REF_DATE']<=date2)&(df['REF_DATE']>=date1)]
    scale = StandardScaler()
    if category_3 ==None:
        df_Canada_CPI_Scaled_US_on_Canada_Tariffs_1 = df_Canada_CPI_Scaled_US_on_Canada_Tariffs[(df_Canada_CPI_Scaled_US_on_Canada_Tariffs['Category']==category_tariff)|(df_Canada_CPI_Scaled_US_on_Canada_Tariffs['Category']==category_non_tariff)]
    else:
        df_Canada_CPI_Scaled_US_on_Canada_Tariffs_1 = df_Canada_CPI_Scaled_US_on_Canada_Tariffs[(df_Canada_CPI_Scaled_US_on_Canada_Tariffs['Category']==category_tariff)|(df_Canada_CPI_Scaled_US_on_Canada_Tariffs['Category']==category_non_tariff)|(df_Canada_CPI_Scaled_US_on_Canada_Tariffs['Category']==category_3)]
    df_Canada_CPI_Scaled_US_on_Canada_Tariffs_1['VALUE'] = scale.fit_transform(df_Canada_CPI_Scaled_US_on_Canada_Tariffs_1['VALUE'].values.reshape(-1, 1))
    chart1 = alt.Chart(df_Canada_CPI_Scaled_US_on_Canada_Tariffs_1).mark_line().encode(
        x='REF_DATE',
        y='VALUE',
        color = alt.Color('Category', legend = alt.Legend(title = 'Tariff and Non-Tariffed Goods')
    ))

    return chart1