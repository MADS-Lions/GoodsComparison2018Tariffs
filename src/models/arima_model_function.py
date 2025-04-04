import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.api import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def arima_model(df, category, order, tariff_date, forecast_steps, in_sample_len=0):
    
    historical_data = df[df.index <= tariff_date][category].asfreq('MS', method='pad')

    arima_model = ARIMA(historical_data, order=order).fit()
    
    if in_sample_len != 0:
        pred = arima_model.predict(start=historical_data.index[-in_sample_len], end=historical_data.index[-1])
        test_data = historical_data.iloc[-in_sample_len:]

        # Augmented Dickey-Fuller (ADF) Test
        differenced_data = historical_data.diff(order[1]).dropna()

# Perform ADF test on the differenced data
        adf_test = adfuller(differenced_data)
        adf_stat, adf_p_value = adf_test[0], adf_test[1]

        print(f"\n📊 **Augmented Dickey-Fuller (ADF) Test on Differenced Data:**")
        print(f"  p-value: {adf_p_value:.4f} {'(Stationary ✅)' if adf_p_value < 0.05 else '(Non-Stationary ❌)'}")

        # Evaluation Metrics
        mae = mean_absolute_error(test_data, pred)
        mae_ratio = mae / test_data.mean()
        rmse = np.sqrt(np.mean((test_data - pred) ** 2))

        print("\n📊 **In-Sample Evaluation Metrics**")
        print(f"    MAE = {mae:.4f}, MAEP = {mae_ratio:.2%}")
        print(f"    RMSE = {rmse:.4f}")
        
        # Residuals & Autocorrelation
        residuals = test_data - pred
        acf_residuals = acf(residuals, nlags=5)  
        ljung_box_pval = q_stat(acf_residuals[1:], len(residuals))[1][-1]

        print(f"\n**Ljung-Box Test (Residuals Autocorrelation):**")
        print(f"  p-value: {ljung_box_pval:.4f} {'(No autocorrelation ✅)' if ljung_box_pval > 0.05 else '(Residuals correlated ❌)'}")

         # Breusch-Pagan Test for Heteroskedasticity
        exog = add_constant(test_data)  # Add intercept term
        bp_test = het_breuschpagan(residuals, exog)
        bp_p_value = bp_test[1]  # p-value

        print(f"\n📊 **Breusch-Pagan Test (Heteroskedasticity):**")
        print(f"  p-value: {bp_p_value:.4f} {'(Homoskedastic ✅)' if bp_p_value > 0.05 else '(Heteroskedastic ❌)'}")

        print(f"\n📊 **Model Selection Criteria:**")
        print(f"  AIC: {arima_model.aic:.4f}")

        pred = arima_model.predict(start=historical_data.index[-in_sample_len], end=historical_data.index[-1])
        test_data = historical_data.iloc[-in_sample_len:]


    forecast_info = arima_model.get_forecast(steps=forecast_steps)
    future_forecast = forecast_info.predicted_mean
    conf_int = forecast_info.conf_int()

    # Create forecast index with monthly frequency (MS = Month Start)
    forecast_dates = pd.date_range(start=historical_data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

    # Create DataFrame for forecast
    forecast_df = pd.DataFrame({
        'Forecast': future_forecast,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    }, index=forecast_dates)

    post_tariff_data = df[(df.index > tariff_date)][category].asfreq('MS', method='pad')
    post_tariff_data = pd.DataFrame({'Actuals': post_tariff_data[:forecast_steps]}, index=forecast_dates)
    
    return (pred if in_sample_len > 0 else None), (test_data if in_sample_len > 0 else None), forecast_df, post_tariff_data, tariff_date, arima_model.polynomial_ar


def plot_arima_results(full_data, category, arima_output):
    """
    Plots the ARIMA model results based on the outputs of the `arima_model` function.
    
    Parameters:
    - category: str, the column name for CPI values to forecast
    - full_data: pd.DataFrame, full dataset with a datetime index
    - arima_output: tuple, output from `arima_model` (contains historical predictions, forecast, and post-tariff data)
    - tariff_date: str, the date of the tariff (defaults to '2018-06-01')
    """
    
    # Ensure the column exists
    if category not in full_data.columns:
        raise ValueError(f"Category '{category}' not found in DataFrame")

    historical_pred, test_data, forecast_df, post_tariff_data, tariff_date, _ = arima_output
    # Ensure tariff_date is parsed correctly as a Timestamp
    
    tariff_date = pd.Timestamp(tariff_date)

    # Ensure full_data index is a DatetimeIndex
    if not isinstance(full_data.index, pd.DatetimeIndex):
        raise TypeError("The index of full_data is not a DatetimeIndex.")

    # Filter train_data for dates before the tariff date
    train_data = full_data[full_data.index < tariff_date]
    # Plotting the data
    plt.figure(figsize=(12, 6))

    # Plot the actual train data
    plt.plot(train_data.index, train_data[category], label="Actual Train Data", color='green', linewidth=2)

    # Plot the predicted data from the model
    if historical_pred is not None:
        plt.plot(historical_pred.index, historical_pred, label="Predicted Test Data", color='#1f77b4', linestyle='--', linewidth=1.5)

    # Plot the actual test data
    if test_data is not None:
        plt.plot(test_data.index, test_data, label="Actual Test Data", color='#ff8c00', linewidth=2)

    # Plot forecasted values
    if forecast_df is not None:
        plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecasted Values", color='red', linestyle='dashed', linewidth=1.5)

        # Add Confidence Interval (if needed)
        #plt.fill_between(forecast_df.index, forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray', alpha=0.3, label="95% Confidence Interval")

    # Plot post-tariff actual data
    if post_tariff_data is not None:
        plt.plot(post_tariff_data.index, post_tariff_data['Actuals'], label="Actual Post-Tariff Data", color='purple', linewidth=2)

    # Mark the tariff date
    plt.axvline(tariff_date, color="black", linestyle="--", label="Tariff Date")

    # Formatting the plot
    plt.xlabel("Date")
    plt.ylabel("CPI Value")
    plt.title(f"ARIMA Forecast vs Actual CPI Values Post Tariff: {category}")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_acf_pcf(full_data, category, tariff_date, lags=20, alpha=.05):
    """
    Plots the ACF & PACF

    Parameters:
    - train_data: pd.DataFrame, the dataset used for training (must have a datetime index).
    - category: str, the column name for CPI values being modeled.
    - lags: int, number of lags to display in PACF plot (default is 20).
    - alpha: float, confidence level to use for testing (default is 0.05)
    """

    # Ensure the category exists in the dataset
    if category not in full_data.columns:
        raise ValueError(f"Category '{category}' not found in training data.")

    train_data = full_data[full_data.index < pd.Timestamp(tariff_date)]


    # Extract the target variable
    data_series = train_data[category].diff().dropna()
    non_sta_data = train_data[category].dropna()


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(data_series, lags=lags, alpha=alpha, ax=axes[0])
    axes[0].set_title(f"ACF on {category}")

    plot_pacf(data_series, lags=lags, alpha=alpha, method='ywm', ax=axes[1])
    axes[1].set_title(f"PACF {category}")

    plt.tight_layout()
    plt.show()



