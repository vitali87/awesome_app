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

data_init = pd.read_excel('Book1.xlsx',
                          sheet_name='Sheet3',
                          names=['date', 'rate', 'volatility'])

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Line(x=data_init['date'], y=data_init['rate'], name='rate'),
    secondary_y=False
)

fig.add_trace(
    go.Line(x=data_init['date'], y=data_init['volatility'], name='volatility'),
    secondary_y=True
)

fig.show()

df = pd.read_excel('Book1.xlsx',
                   sheet_name='Sheet2',
                   names=['date', 'rate', 'volatility'])


PERIODS_AHEAD = 10*251
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Line(x=df['date'], y=df['rate'], name='rate'),
    secondary_y=False
)

fig.add_trace(
    go.Line(x=df['date'], y=df['volatility'], name='volatility'),
    secondary_y=True
)

fig.show()

ts = df.loc[:, 'rate']
ts.index = df['date']
diff_ts = np.diff(ts)
train = ts[ts.index <= datetime(2019, 12, 12)]
test_size = len(ts)-len(train)

# Automatic ARIMA model
model = auto_arima(train, start_p=0, start_q=0)
model.summary()
forecast = model.predict(steps=PERIODS_AHEAD)

# fig3 = model.plot_diagnostics()
# fig3.show()

# Showing that time series is not stationary
# The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root
# If the pvalue is above a critical size, then we cannot reject that there is a unit root.
dftest = adfuller(ts)

# First difference is stationary which can be seen from auto_arima as well
adfuller(diff_ts)


mod = sm.tsa.statespace.SARIMAX(train, trend='c', order=model.order)
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
fig.show()

# ACF and PACF plotss
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

fig = sm.graphics.tsa.plot_acf(ts, lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(ts, lags=40, ax=axes[1], method='ywm')

fig.show()

roll_mean = ts.rolling(window=8, center=False).mean()
roll_std = ts.rolling(window=8, center=False).std()

fig.add_trace(
    go.Line(x=df['date'], y=roll_mean, name='roll_mean'),
    secondary_y=False
)

fig.add_trace(
    go.Line(x=df['date'], y=roll_std, name='roll_std'),
    secondary_y=False
)
fig.show()

# Apply seasonal_decomposition
decomposition = seasonal_decompose(np.log(df.loc[:, 'rate']), period=5)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(
    go.Line(x=df['date'], y=trend, name='trend'),
    secondary_y=False
)
fig2.add_trace(
    go.Line(x=df['date'], y=residual, name='residual'),
    secondary_y=True
)
fig2.show()

# ---------------------------------------------------- #
# define ARCH family models
model1 = arch_model(res.resid, mean='Zero', vol='GARCH', p=5, q=5)
model2 = arch_model(res.resid, mean='Zero', vol='ARCH', q=5)

model_fit1 = model1.fit()
model_fit2 = model2.fit()

model_fit1.summary()
model_fit2.summary()

yhat1 = model_fit1.forecast(horizon=PERIODS_AHEAD)
yhat2 = model_fit2.forecast(horizon=PERIODS_AHEAD)

# Variability will go up no matter which method
fig = go.Figure([
    go.Scatter(
        x=[*range(PERIODS_AHEAD)],
        y=yhat1.variance.values[-1, :],
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='GARCH'
    ),
    go.Scatter(
        x=[*range(PERIODS_AHEAD)],
        y=yhat2.variance.values[-1, :],
        line=dict(color='rgba(68, 68, 68, 0.3)'),
        mode='lines',
        name='ARCH'
    )
])
fig.show()
