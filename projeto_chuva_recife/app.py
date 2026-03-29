import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

st.title("🌧️ Previsão de Chuvas - Recife")


dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")

rain = (
    5 +
    10 * np.sin(2 * np.pi * dates.dayofyear / 365) +
    np.random.normal(0, 3, len(dates))
)

rain = np.clip(rain, 0, None)

df = pd.DataFrame({
    "date": dates,
    "rain": rain
})

# Prophet

df_prophet = df.rename(columns={
    'date': 'ds',
    'rain': 'y'
})

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Grafico

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_prophet['ds'],
    y=df_prophet['y'],
    mode='lines',
    name='Histórico'
))

fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Previsão'
))

fig.update_layout(
    title='Previsão de Chuvas',
    xaxis_title='Data',
    yaxis_title='Precipitação (mm)'
)

st.plotly_chart(fig)
