import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(
    page_title="Previsão de Chuvas Recife",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Previsão de Chuvas - Recife")

dates = pd.date_range(start="2016-01-01", end="2025-12-31", freq="D")

rain = (
    8 +
    12 * np.sin(2 * np.pi * dates.dayofyear / 365) +
    np.random.normal(0, 4, len(dates))
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

col1, col2, col3 = st.columns(3)

col1.metric("📅 Período Histórico", "2016 - 2025")
col2.metric("🌧️ Média de Chuva", f"{df['rain'].mean():.2f} mm")
col3.metric("🔮 Horizonte de Previsão", "12 meses")



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
    name='Histórico',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Previsão',
    line=dict(width=3, dash='dot')
))

fig.update_layout(
    title={
        'text': '🌧️ Previsão de Chuvas - Recife',
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title='Data',
    yaxis_title='Precipitação (mm)',
    hovermode='x unified',
    template='plotly_dark',
    height=650,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

fig.update_xaxes(
    rangeslider_visible=True
)

st.plotly_chart(fig)
