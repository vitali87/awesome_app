import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch import arch_model

st.set_page_config(page_title="Liuda's awesome app",
                   layout='wide')

st.title("Liuda's awesome app")


@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


data_init = pd.read_excel('Book1.xlsx',
                          sheet_name='Sheet3',
                          names=['date', 'rate', 'volatility'])
df = pd.read_excel('Book1.xlsx',
                   sheet_name='Sheet2',
                   names=['date', 'rate', 'volatility'])


ts = df.loc[:, 'rate']
ts.index = df['date']
diff_ts = np.diff(ts)
train = ts[ts.index <= datetime(2019, 12, 12)]

if st.checkbox('Show dataframe'):
    x = st.slider('How many rows to show?', value=5,
                  max_value=data_init.shape[0])
    st.write(data_init.head(x))

if st.checkbox('Plot All Data'):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Line(x=data_init['date'], y=data_init['rate'], name='rate'),
        secondary_y=False
    )

    fig.add_trace(
        go.Line(x=data_init['date'],
                y=data_init['volatility'], name='volatility'),
        secondary_y=True
    )

    st.plotly_chart(fig, use_container_width=True)

if st.checkbox('Plot Data After Financial Crisis'):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Line(x=df['date'], y=df['rate'], name='rate'),
        secondary_y=False
    )

    fig.add_trace(
        go.Line(x=df['date'], y=df['volatility'], name='volatility'),
        secondary_y=True
    )

    with st.expander("See explanation"):
        st.write("""
         The behaviour of time series has changed significantly since the Financial Crisis,
         therefore data has been trancated from the left. Compare this plot with All data.
     """)
    st.plotly_chart(fig, use_container_width=True)


if st.checkbox('Plot Training Data'):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Line(x=df['date'], y=train, name='rate'),
        secondary_y=False
    )
    with st.expander("See explanation"):
        st.write("""
         The data is anomaluous during Covid-19, therefore data has been trancated from the right.
         Overall, training spans from 2009 to 2020.
     """)
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox('Forecast and Plot'):
    years = st.radio(
        "How many years ahead to forecast",
        (5, 10, 15, 20, 25, 30))

    with st.spinner('Wait for it...'):
        PERIODS_AHEAD = years*251
        mod = sm.tsa.statespace.SARIMAX(train, trend='c', order=(0, 1, 1))
        res = mod.fit(disp=False)
        res.summary()

        start = len(ts)
        end = start + PERIODS_AHEAD

        predict = res.get_prediction(start, end)

        predict_ci = predict.conf_int()

        x = [*range(start, end)]
        fig = go.Figure([
            go.Scatter(
                x=x,
                y=predict.predicted_mean,
                line=dict(color='rgb(0,100,80)'),
                mode='lines',
                name='predicted_mean',
            ),
            go.Scatter(
                x=x,
                y=predict_ci['upper rate'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                x=x,
                y=predict_ci['lower rate'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ),
            go.Scatter(
                name='var_pred_mean',
                x=x,
                y=predict.var_pred_mean,
                mode='lines',
                line=dict(color='rgb(100,0,0)'),
            )
        ])
        st.plotly_chart(fig, use_container_width=True)


csv_all = convert_df(data_init)
csv_after_crisis = convert_df(df)
csv_training = convert_df(train)


st.download_button(
    label="Download All data as CSV",
    data=csv_all,
    file_name='all_data.csv',
    mime='text/csv',
)
st.download_button(
    label="Download After-Crisis data as CSV",
    data=csv_after_crisis,
    file_name='after_crisis_data.csv',
    mime='text/csv',
)
st.download_button(
    label="Download Training data as CSV",
    data=csv_training,
    file_name='training_data.csv',
    mime='text/csv',
)
