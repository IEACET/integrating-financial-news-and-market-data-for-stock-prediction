
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import pytz
import holidays

from sklearn.linear_model import Ridge
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE_DIR, "news_data")

class StreamlitCallback(keras.callbacks.Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append(logs)
        self.placeholder.write(self.logs)

@st.cache_data
def load_stocks_df():
    return pd.read_csv("indian_stocks.csv")

stocks_df = load_stocks_df()

def normalize_indian_ticker(ticker):
    ticker = ticker.upper()
    if "." not in ticker:
        return ticker + ".NS"
    return ticker

def is_nse_market_open():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Weekend
        return False
    in_holidays = holidays.IN(years=now.year)
    if now.date() in in_holidays:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

@st.cache_data(ttl=3600)
def validate_ticker(ticker):
    ticker = normalize_indian_ticker(ticker)
    try:
        data = yf.download(ticker, period="1d")
        return not data.empty
    except:
        return False

@st.cache_data(ttl=86400)
def fetch_historical_data(
    ticker,
    start_date="2000-01-01",
    end_date=None,
    adjust=True
):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=adjust
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def intro(ticker):
    st.title("Stock Prediction Web App")
    if not validate_ticker(ticker):
        st.error("Invalid ticker symbol.")
        return

    try:
        data = fetch_historical_data(ticker, adjust=False)

        if data.empty:
            st.error("No data found.")
            return

        stock_info = yf.Ticker(ticker).info
        st.subheader(f"Stock Details for {ticker}")
        if 'longName' in stock_info:
            st.write(f"Company: {stock_info['longName']}")
        if 'sector' in stock_info:
            st.write(f"Sector: {stock_info['sector']}")
        if 'industry' in stock_info:
            st.write(f"Industry: {stock_info['industry']}")

        st.subheader("Historical Stock Data Visualization")
        columns_to_plot = ['Open', 'High', 'Low', 'Close', 'Volume']
        tabs = st.tabs(columns_to_plot)

        for i, col in enumerate(columns_to_plot):
            with tabs[i]:
                fig = go.Figure(data=go.Scatter(x=data.index, y=data[col], mode='lines'))
                fig.update_layout(title=f"{col} for {ticker}", xaxis_title="Date", yaxis_title=f"{col}")
                st.plotly_chart(fig, width='stretch')
        st.dataframe(data)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def plot_actual_vs_predicted(
    df,
    y_test_rescaled,
    predictions,
    price_columns,
    with_news=False,
    baseline=False
):

    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    # Flatten columns if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_test = pd.DataFrame(y_test_rescaled, columns=price_columns, index=df.index[-len(y_test_rescaled):]).copy()
    df_pred = pd.DataFrame(predictions, columns=price_columns, index=df_test.index).copy()

    # BASELINE REGRESSION PLOT
    # ============================
    if baseline:
        st.subheader("Actual vs Predicted Prices (Daily – Test Period Only)")

        tabs = st.tabs(price_columns)

        for i, col in enumerate(price_columns):
            with tabs[i]:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df_test.index,
                        y=df_test[col],
                        mode="lines",
                        name=f"Actual {col}"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_pred.index,
                        y=df_pred[col],
                        mode="lines",
                        name=f"Predicted {col}"
                    )
                )

                fig.update_layout(
                    title=f"{col} Price Prediction (Regression Baseline)",
                    xaxis_title="Date",
                    yaxis_title=col
                )

                st.plotly_chart(fig, width="stretch")

        return

    # Check history length to decide aggregation
    use_monthly = with_news or (len(df.index.year.unique()) < 5)  # Use monthly if short history or with_news

    if use_monthly:
        df_train['Month'] = df_train.index.to_period('M')
        df_test['Month'] = df_test.index.to_period('M')
        df_pred['Month'] = df_pred.index.to_period('M')


        df_train_agg = df_train.groupby('Month').mean(numeric_only=True)
        df_test_agg = df_test.groupby('Month').mean(numeric_only=True)
        df_pred_agg = df_pred.groupby('Month').mean(numeric_only=True)
        df_actual = pd.concat([df_train_agg, df_test_agg])
        prediction_start = df_pred_agg.index.min()
        agg_label = "Monthly"

        x_actual = df_actual.index.strftime('%b %Y')
        x_pred = df_pred_agg.index.strftime('%b %Y')
        pred_start_str = prediction_start.strftime('%b %Y')
    else:
        df_train['Year'] = df_train.index.year
        df_test['Year'] = df_test.index.year
        df_pred['Year'] = df_pred.index.year

        df_train_agg = df_train.groupby('Year').mean(numeric_only=True)
        df_test_agg = df_test.groupby('Year').mean(numeric_only=True)
        df_pred_agg = df_pred.groupby('Year').mean(numeric_only=True)
        df_actual = pd.concat([df_train_agg, df_test_agg])
        prediction_start = df_pred_agg.index.min()
        agg_label = "Yearly"

        x_actual = df_actual.index.astype(str)
        x_pred = df_pred_agg.index.astype(str)
        pred_start_str = str(prediction_start)

    st.subheader(f"Actual vs. Predicted {agg_label} Averages")
    tabs = st.tabs(price_columns)

    for i, col in enumerate(price_columns):
        with tabs[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_actual, y=df_actual[col], mode='lines', name=f'Actual {col}'))
            fig.add_trace(go.Scatter(x=x_pred, y=df_pred_agg[col], mode='lines', name=f'Predicted {col}'))
            
            fig.add_shape(
                type="line",
                x0=pred_start_str,
                y0=0,
                x1=pred_start_str,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="white", width=1.5, dash="dash")
            )
            
            fig.add_annotation(
                text="Prediction Start",
                x=pred_start_str,
                y=1,
                yref="paper",
                showarrow=False,
                textangle=-90,
                xanchor="right",
                yanchor="bottom"
            )
            
            fig.update_layout(
                title=f'{col} Price Prediction ({agg_label})',
                xaxis_title='Period',
                yaxis_title=f'{col} Price',
                xaxis=dict(type='category')
            )
            st.plotly_chart(fig, width='stretch')

def load_and_preprocess_data(ticker, with_news=False, symbol=None):
    if with_news:
        return load_price_with_news_data(ticker, symbol)
    else:
        return load_price_only_data_for_lr(ticker)

def load_price_only_data_for_lr(ticker, prediction_start_date=None):
    df = fetch_historical_data(ticker)
    df.index = df.index.tz_localize(None)
    if prediction_start_date is not None:
     df = df[df.index >= prediction_start_date - timedelta(days=60)]


    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]

    df['Log_Volume'] = np.log(df['Volume'] + 1e-9)
    df.drop(columns=['Volume'], inplace=True)

    price_cols = ['Open', 'High', 'Low', 'Close', 'Log_Volume']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[price_cols])
    scaled_df = pd.DataFrame(scaled, columns=price_cols, index=df.index)

    window = 60
    X, y = [], []

    for i in range(window, len(scaled_df)):
        X.append(scaled_df.iloc[i-window:i].values.flatten())
        y.append(scaled_df.iloc[i].values)

    X = np.array(X)
    y = np.array(y)

    if prediction_start_date is not None:
    # find first index where prediction starts
     split_idx = np.where(df.index[60:] >= prediction_start_date)[0][0]

     X_train = X[:split_idx]
     X_test  = X[split_idx:]
     y_train = y[:split_idx]
     y_test  = y[split_idx:]
    else:
    # fallback to old behavior
     X_train = X[:-100]
     X_test  = X[-100:]
     y_train = y[:-100]
     y_test  = y[-100:]

    return X_train, X_test, y_train, y_test, scaler, df, price_cols


def load_price_with_news_data(ticker, symbol):
    df = fetch_historical_data(ticker, start_date=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"))
    df.index = df.index.tz_localize(None)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]

    df['Log_Volume'] = np.log(df['Volume'] + 1e-9)
    df.drop(columns=['Volume'], inplace=True)

    price_cols = ['Open', 'High', 'Low', 'Close', 'Log_Volume']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[price_cols])
    scaled_df = pd.DataFrame(scaled, columns=price_cols, index=df.index)

    news_path = os.path.join(NEWS_DIR, f"{symbol.lower()}_news.csv")

    if not os.path.exists(news_path):
        st.error(f"News data not found at {news_path}. Run scraper first.")
        return (None,) * 7

    dfnews = pd.read_csv(news_path)
    dfnews['datetime'] = pd.to_datetime(dfnews['datetime'], errors='coerce')
    dfnews = dfnews.dropna(subset=['datetime'])
    dfnews['datetime'] = dfnews['datetime'].dt.tz_localize(None)
    dfnews.sort_values('datetime', inplace=True)

    scaled_df = scaled_df.reset_index()
    scaled_df.columns = ['datetime'] + price_cols
    scaled_df['datetime'] = pd.to_datetime(scaled_df['datetime'])

    daily_sentiment = dfnews.groupby(dfnews['datetime'].dt.date).agg({
        'compound_score': 'mean',
        'negative_score': 'mean',
        'neutral_score': 'mean',
        'positive_score': 'mean'
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'compound_score', 'negative_score', 'neutral_score', 'positive_score']
    daily_sentiment['datetime'] = pd.to_datetime(daily_sentiment['date'])

    merged = pd.merge_asof(
        scaled_df.sort_values('datetime'),
        daily_sentiment[['datetime', 'compound_score']].sort_values('datetime'),
        on='datetime',
        direction='backward'
    )

    merged['compound_score'].fillna(0, inplace=True)
    merged.set_index('datetime', inplace=True)

    feature_cols = price_cols + ['compound_score']
    window = 60
    X, y = [], []

    for i in range(window, len(merged)):
        X.append(merged[feature_cols].iloc[i-window:i].values)
        y.append(merged[price_cols].iloc[i].values)

    return (
        np.array(X[:-100]),
        np.array(X[-100:]),
        np.array(y[:-100]),
        np.array(y[-100:]),
        scaler,
        df,
        price_cols
    )

def build_model(input_shape, output_shape):
    model = keras.Sequential([
        # CNN layers for local pattern extraction (e.g., short-term price fluctuations)
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # LSTM layers for sequential modeling
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.3),
        
        # Dense layers for output
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_new_model(X_train, y_train, X_test, y_test, input_shape, output_shape, model_path, log_placeholder):
    model = build_model(input_shape, output_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    with st.spinner("Training in progress..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[StreamlitCallback(log_placeholder), early_stop],
            verbose=0
        )

    model.save(model_path)
    st.success(f"Model saved to {model_path}")

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss'))
    fig_loss.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig_loss, width='stretch')

    return model

def show_regression_results(y_true, y_pred, price_cols):
    st.subheader("📊 Regression Model Results")

    results = []
    for i, col in enumerate(price_cols):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        results.append({
            "Price": col,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R² Score": round(r2, 3)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, width="stretch")

def linear_regression_baseline(ticker):
    st.header("📈 Linear Regression Baseline Model")
    st.caption("This model uses historical stock prices only (2000–present) and does NOT use news sentiment.")
    prediction_start_date = st.session_state.get("prediction_start_date", None)

    X_train, X_test, y_train, y_test, scaler, df, price_cols = load_price_only_data_for_lr(
    ticker,
    prediction_start_date=prediction_start_date
)

    if X_train is None:
        st.error("Failed to load data.")
        return

    model_path = f"{ticker}_lr_model.pkl"

    if os.path.exists(model_path):
        st.info(f"Loading existing model from {model_path}")
        import joblib
        model = joblib.load(model_path)
    else:
        st.info("Building and training a new model...")
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        import joblib
        joblib.dump(model, model_path)
        st.success(f"Model saved to {model_path}")

    predictions = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(y_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    
    show_regression_results(
    y_test_rescaled,
    predictions_rescaled,
    price_cols
)

    plot_actual_vs_predicted(
    df,
    y_test_rescaled,
    predictions_rescaled,
    price_cols,
    baseline=True
)

def short_term_prediction_lstm(ticker):
    symbol = ticker.replace(".NS", "")
    st.write("Training a short-term LSTM model with news sentiment...")

    X_train, X_test, y_train, y_test, scaler, df, price_cols = load_and_preprocess_data(ticker, with_news=True, symbol=symbol)

    if X_train is None:
        st.error("Failed to load data.")
        return

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = len(price_cols)
    model_path = f"{ticker}_short_term_model.h5"

    log_placeholder = st.empty()

    if os.path.exists(model_path):
        st.info(f"Loading existing model from {model_path}")
        model = load_model(model_path, compile=False)
        try:
            model.predict(np.zeros((1, *input_shape)))
        except Exception as e:
            st.info(f"Model shape mismatch: {e}. Training new model...")
            model = train_new_model(X_train, y_train, X_test, y_test, input_shape, output_shape, model_path, log_placeholder)
    else:
        st.info("Building and training a new model...")
        model = train_new_model(X_train, y_train, X_test, y_test, input_shape, output_shape, model_path, log_placeholder)

    predictions = model.predict(X_test)
    y_test_rescaled = scaler.inverse_transform(y_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    prediction_start_date = df.index[-100]
    st.session_state["prediction_start_date"] = prediction_start_date

    plot_actual_vs_predicted(
    df,
    y_test_rescaled,
    predictions_rescaled,
    price_cols,
    with_news=True
)

def open_trained_model(ticker, with_news=False):
    symbol = ticker.replace(".NS", "")
    model_path = f"{ticker}_short_term_model.h5" if with_news else f"{ticker}_lr_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Train the model first.")
        return
    X_train, X_test, y_train, y_test, scaler, df, price_cols = load_and_preprocess_data(
        ticker, with_news=with_news, symbol=symbol
    )
    if X_train is None:
        st.error("Failed to load data.")
        return
    if with_news:
        model = load_model(model_path, compile=False)
        input_shape = (X_train.shape[1], X_train.shape[2])
        try:
            model.predict(np.zeros((1, *input_shape)))
        except Exception as e:
            st.error(f"Model shape mismatch: {e}. Please retrain the model.")
            return
    else:
        import joblib
        model = joblib.load(model_path)

    predictions = model.predict(X_test)

    y_test_rescaled = scaler.inverse_transform(y_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    plot_actual_vs_predicted(df, y_test_rescaled, predictions_rescaled, price_cols, with_news=with_news)

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="5m")
        
        if data.empty:
            return None
        
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_sma(data, short_window=5, long_window=10):
    data["SMA_5"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_10"] = data["Close"].rolling(window=long_window).mean()
    return data

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def calculate_zscore(data, window=20):
    data["ZScore"] = (data["Close"] - data["Close"].rolling(window=window).mean()) / data["Close"].rolling(window=window).std()
    return data

def generate_signals(data):
    data["Signal"] = "Hold"
    data.loc[(data["SMA_5"] > data["SMA_10"]) & (data["RSI"] < 70) & (data["ZScore"] < 1.5), "Signal"] = "Buy"
    data.loc[(data["SMA_5"] < data["SMA_10"]) & (data["RSI"] > 30) & (data["ZScore"] > -1.5), "Signal"] = "Sell"
    return data

def apply_sentiment_filter(data, sentiment_score, threshold=0.05):
    if sentiment_score < -threshold:
        data.loc[data["Signal"] == "Buy", "Signal"] = "Hold"
    elif sentiment_score > threshold:
        data.loc[data["Signal"] == "Sell", "Signal"] = "Hold"
    return data

def plot_signals_plotly(data):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_5"], mode='lines', name='SMA 5', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_10"], mode='lines', name='SMA 10', line=dict(color='green')))

    buy_signals = data[data["Signal"] == "Buy"]
    sell_signals = data[data["Signal"] == "Sell"]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode='markers',
                             marker=dict(color='lime', size=12, symbol='triangle-up'),
                             name='Buy Signal'))

    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode='markers',
                             marker=dict(color='red', size=12, symbol='triangle-down'),
                             name='Sell Signal'))

    fig.update_layout(
        title="📊 Buy/Sell Signals with SMA & RSI",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig, width='stretch')

    st.subheader("📈 Signal Summary")
    signal_counts = data["Signal"].value_counts()
    st.write(signal_counts)

    latest_signal = data.iloc[-1]["Signal"]
    st.metric("Current Signal", latest_signal)

def plot_live_dashboard(ticker):
    st.header("📊 Live OHLCV Dashboard")

    if is_nse_market_open():
        st.success("🟢 NSE Market is OPEN")
    else:
        st.warning("🔴 NSE Market is CLOSED (showing last available data)")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    with st.spinner("Fetching live data..."):
        live_data = get_stock_data(ticker)

    if live_data is None or live_data.empty:
        st.warning("No live data available.")
        return

    st.subheader("📈 Current Price")
    current_price = live_data["Close"].iloc[-1]
    st.metric("Close Price", f"₹{current_price:.2f}")

    st.subheader("📊 OHLCV Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=live_data.index,
        open=live_data['Open'],
        high=live_data['High'],
        low=live_data['Low'],
        close=live_data['Close']
    )])
    fig.update_layout(title=f"{ticker} Live Price Chart", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, width='stretch')

    st.subheader("📊 Volume")
    fig_vol = go.Figure(data=go.Bar(x=live_data.index, y=live_data["Volume"]))
    fig_vol.update_layout(title="Volume Over Time", xaxis_title="Time", yaxis_title="Volume")
    st.plotly_chart(fig_vol, width='stretch')

    st.dataframe(live_data.tail(20))

def plot_live_signals(ticker):
    with st.spinner("Fetching live data..."):
        live_data = get_stock_data(ticker)

    if live_data is None or live_data.empty:
        st.warning("No live data available.")
        return

    if len(live_data) < 20:
        st.warning(f"Not enough data ({len(live_data)} points). Need at least 20.")
        return

    live_data = calculate_sma(live_data)
    live_data = calculate_rsi(live_data)
    live_data = calculate_zscore(live_data)
    live_data = generate_signals(live_data)

    st.subheader("📊 Live Signals Chart")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data["Close"], name="Close", line=dict(color="blue")))
    fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data["SMA_5"], name="SMA 5", line=dict(color="orange")))
    fig_ma.add_trace(go.Scatter(x=live_data.index, y=live_data["SMA_10"], name="SMA 10", line=dict(color="green")))
    fig_ma.update_layout(title="Price & Moving Averages", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig_ma, width='stretch')

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=live_data.index, y=live_data["RSI"], name="RSI", line=dict(color="purple")))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="RSI Indicator", xaxis_title="Time", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, width='stretch')

    latest = live_data.iloc[-1]
    signal = latest["Signal"]
    st.metric("Current Signal", signal)
    st.metric("RSI", round(latest["RSI"], 2))
    st.metric("Z-Score", round(latest["ZScore"], 2))

    st.markdown("### 📄 Live Data Snapshot")
    st.dataframe(live_data.tail(20))

def short_term_live(ticker):
    st.header("📡 Live Short-Term Trading Signals")

    if is_nse_market_open():
        st.success("🟢 NSE Market is OPEN")
    else:
        st.warning("🔴 NSE Market is CLOSED (showing last available data)")

    if st.button("🔄 Refresh Now"):
        st.rerun()

    plot_live_signals(ticker)

def get_latest_sentiment(symbol):
    news_path = os.path.join(NEWS_DIR, f"{symbol.lower()}_news.csv")

    if not os.path.exists(news_path):
        return 0.0

    dfnews = pd.read_csv(news_path)
    dfnews["datetime"] = pd.to_datetime(dfnews["datetime"], errors="coerce")
    dfnews = dfnews.dropna(subset=["datetime"])

    recent = dfnews[dfnews["datetime"] >= (datetime.now() - timedelta(days=3))]
    if recent.empty:
        return 0.0

    return recent["compound_score"].mean()

def short_term_signals(ticker):
    st.title("Stock Buy/Sell Signal Dashboard")

    data = get_stock_data(ticker)

    if data is None or data.empty:
        st.warning("No intraday data available right now.")
        return

    if len(data) < 20:
        st.warning(f"Not enough data points ({len(data)} candles). Try later when market has more activity.")
        st.dataframe(data)
        return
    
    symbol = ticker.replace(".NS", "")
    sentiment_score = get_latest_sentiment(symbol)
    st.markdown(f"### 🧠 Recent News Sentiment Score: **{sentiment_score:.3f}**")

    data = calculate_sma(data)
    data = calculate_rsi(data, window=14)
    data = calculate_zscore(data, window=20)
    data = generate_signals(data)
    data = apply_sentiment_filter(data, sentiment_score)

    if sentiment_score > 0.05:
        st.success("📈 Overall sentiment is POSITIVE")
    elif sentiment_score < -0.05:
        st.error("📉 Overall sentiment is NEGATIVE")
    else:
        st.info("⚖️ Overall sentiment is NEUTRAL")

    plot_signals_plotly(data)

def short_term(ticker):
    mode = st.sidebar.radio(
        "Select Mode",
        ["Short-Term Signals", "Live OHLCV", "Live Signals"],
        key="short_term_mode"
    )

    if mode == "Short-Term Signals":
        short_term_signals(ticker)
    elif mode == "Live OHLCV":
        plot_live_dashboard(ticker)
    elif mode == "Live Signals":
        short_term_live(ticker)

def main():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Indian Stock Market Price Prediction & Trading Signals")
    st.subheader("NSE-focused analysis using Deep Learning and Market Sentiment")

    st.sidebar.title("Navigation")

    stock_options = stocks_df["symbol"].str.upper().tolist()
    selected_symbol = st.sidebar.selectbox("Select NSE Stock", stock_options)
    ticker = normalize_indian_ticker(selected_symbol)

    page = st.sidebar.radio(
        "Select Section",
        [
            "Home",
            "Sentiment",
            "Regression Baseline",
            "Long-term Model (With News)",
            "Long-term Model Results",
            "Live Demo"
        ]
    )

    if page == "Home":
        intro(ticker)

    elif page == "Sentiment":
        st.header("🧠 News Sentiment Analysis")

        symbol = ticker.replace(".NS", "")

        price_df = fetch_historical_data(
            ticker,
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )

        if price_df.empty:
            st.warning("No stock price data available.")
            return

        price_df = price_df.reset_index()
        price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.tz_localize(None).dt.floor("D")
        price_df = price_df[["Date", "Close"]]

        news_path = os.path.join(NEWS_DIR, f"{symbol.lower()}_news.csv")

        if not os.path.exists(news_path):
            st.warning(f"No sentiment data available at {news_path}.")
            return

        dfnews = pd.read_csv(news_path)
        dfnews["datetime"] = pd.to_datetime(dfnews["datetime"]).dt.tz_localize(None)
        dfnews["Date"] = dfnews["datetime"].dt.floor("D")

        daily_sentiment = (
            dfnews.groupby("Date")["compound_score"]
            .mean()
            .reset_index()
        )

        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)

        price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()
        daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"]).dt.normalize()
        
        merged_df = pd.merge(
            price_df,
            daily_sentiment,
            on="Date",
            how="left"
        )
        merged_df["compound_score"] = merged_df["compound_score"].fillna(0)

        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist).date()
        stock = yf.Ticker(ticker)
        current_info = stock.info
        current_price = current_info.get('regularMarketPrice', current_info.get('currentPrice', merged_df["Close"].iloc[-1]))

        if merged_df["Date"].iloc[-1].date() < today:
            new_row = {'Date': pd.to_datetime(today), 'Close': current_price, 'compound_score': 0}
            new_df = pd.DataFrame([new_row])
            merged_df = pd.concat([merged_df, new_df], ignore_index=True)
        elif is_nse_market_open():
            merged_df.loc[merged_df.index[-1], "Close"] = current_price

        merged_df = merged_df.sort_values("Date").reset_index(drop=True)

        st.subheader("📈 Stock Price Movement")
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=merged_df["Date"],
                y=merged_df["Close"],
                mode="lines+markers",
                name="Close Price"
            )
        )
        fig_price.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig_price, width='stretch')

        st.subheader("🧠 News Sentiment Trend")

        fig_sent = go.Figure()

        fig_sent.add_trace(
            go.Bar(
                x=merged_df["Date"],
                y=merged_df["compound_score"],
                name="Daily Sentiment",
                marker_color=np.where(
                    merged_df["compound_score"] >= 0, "#00c853", "#ff5252"
                )
            )
        )

        fig_sent.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray"
        )

        fig_sent.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment",
            height=400
        )
        st.plotly_chart(fig_sent, width='stretch')

        with st.expander("📰 View raw news & sentiment"):
            st.dataframe(dfnews[["datetime", "title", "compound_score"]])

    elif page == "Regression Baseline":
        linear_regression_baseline(ticker)

    elif page == "Long-term Model (With News)":
        st.header("📈 Long-Term Prediction Model with News Sentiment")
        st.caption("This model uses recent 2 years prices + news sentiment.")
        short_term_prediction_lstm(ticker)

    elif page == "Long-term Model Results":
        st.subheader("📊 Long-term Model Evaluation Results")
        open_trained_model(ticker, with_news=True)

    elif page == "Live Demo":
        st.header("📡 Live Market Demo")
        short_term(ticker)

if __name__ == "__main__":
    main()