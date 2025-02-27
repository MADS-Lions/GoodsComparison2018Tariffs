import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from scipy.signal import correlate

def line_plot(data, x, y, y2, color=None, title=""):
    chart1 = alt.Chart(data).mark_line().encode(
        x=x,
        y=y,
        color=color
    )
    chart2 = alt.Chart(data).mark_line().encode(
        x=x,
        y=y2,
        color=color
    )
    return (chart1 + chart2).properties(
        title=title
    )


def histogram(data, x, color=None, title=""):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(x, bin=True),
        y='count()',
        color=color
    ).properties(
        title=title
    )
    return chart

def cross_correlation_plot(data1, data2, name_data1, name_data2):
    correlated_data = correlate(data1, data2)
    lags = np.arange(-len(data1) + 1, len(data1))
    plt.figure(figsize=(10, 5))
    plt.stem(lags, correlated_data, basefmt=" ")
    plt.title('Cross-Correlation Between ' + name_data1 + " and " + name_data2)
    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.grid()
    plt.legend()
    plt.show()
    return None


