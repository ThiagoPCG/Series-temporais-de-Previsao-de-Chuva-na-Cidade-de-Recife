from prophet import Prophet

def train_model(df, events=None):
    df_prophet = df.rename(columns={'date': 'ds', 'rain': 'y'})
    
    if events is not None:
        model = Prophet(holidays=events, yearly_seasonality=True, weekly_seasonality=True)
    else:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)

    model.fit(df_prophet)
    return model, df_prophet
