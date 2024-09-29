import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema
import os
import logging
from tqdm import tqdm
from scipy.optimize import minimize
import traceback
from datetime import datetime
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker, period="1mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            logging.warning(f"No data available for {ticker}")
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def identify_candlestick_patterns(df):
    logging.info("Identifying Bullish Engulfing pattern...")
    
    # Bullish Engulfing pattern
    df['Bullish_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous day was bearish
        (df['Open'] < df['Close'].shift(1)) &           # Open below previous close
        (df['Close'] > df['Open'].shift(1)) &           # Close above previous open
        (df['Close'] > df['Open'])                      # Current day is bullish
    )
    
    return df

def identify_peaks_valleys(data):
    data['Peaks'] = data['High'].iloc[argrelextrema(data['High'].values, np.greater_equal, order=5)[0]]
    data['Valleys'] = data['Low'].iloc[argrelextrema(data['Low'].values, np.less_equal, order=5)[0]]
    return data

def identify_support_resistance(data, proximity=0.02):
    supports = []
    resistances = []

    valley_points = data.dropna(subset=['Valleys'])
    for i in range(len(valley_points)):
        val1 = valley_points['Valleys'].iloc[i]
        nearby_valleys = valley_points[(valley_points['Valleys'] >= val1 * (1 - proximity)) &
                                       (valley_points['Valleys'] <= val1 * (1 + proximity))]
        if len(nearby_valleys) >= 2:
            supports.append(val1)

    peak_points = data.dropna(subset=['Peaks'])
    for i in range(len(peak_points)):
        peak1 = peak_points['Peaks'].iloc[i]
        nearby_peaks = peak_points[(peak_points['Peaks'] >= peak1 * (1 - proximity)) &
                                   (peak_points['Peaks'] <= peak1 * (1 + proximity))]
        if len(nearby_peaks) >= 2:
            resistances.append(peak1)

    data['Support'] = min(supports) if supports else np.nan
    data['Resistance'] = max(resistances) if resistances else np.nan
    return data

def calculate_ema(data, periods=[8, 20]):
    for period in periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def fit_trendline(data, n_peaks=3, tolerance=0.01):
    # Find peaks
    peaks = data[data['High'] == data['High'].rolling(window=5, center=True).max()]
    peaks = peaks.sort_index(ascending=False).head(n_peaks)
    
    if len(peaks) < 2:
        return None
    
    x = (peaks.index - peaks.index[0]).days.values.astype(float)
    y = peaks['High'].values
    
    # Define the error function
    def error(params):
        m, b = params
        err = np.sum(np.abs(y - (m * x + b)))
        penalty = np.sum(np.maximum(0, y - (m * x + b + tolerance * y)))
        return err + penalty * 1000
    
    # Fit the line
    result = minimize(error, [0, y.mean()], method='Nelder-Mead')
    m, b = result.x
    
    # Generate trendline values
    trendline = pd.Series(index=data.index, dtype=float)
    for i, date in enumerate(data.index):
        days = (date - peaks.index[0]).days
        trendline[date] = m * days + b
    
    return trendline

def plot_chart(data, ticker):
    # Prepare the data
    plot_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Create additional plots
    apds = [
        mpf.make_addplot(data['EMA_8'], color='blue', width=0.7, label='EMA 8'),
        mpf.make_addplot(data['EMA_20'], color='red', width=0.7, label='EMA 20'),
    ]
    
    if 'Trendline' in data.columns and not data['Trendline'].isnull().all():
        apds.append(mpf.make_addplot(data['Trendline'], color='purple', width=1.5, linestyle=':', label='Trendline'))
    
    # Prepare marker data for Bullish Engulfing pattern
    bullish_engulfing_markers = data['Low'].where(data['Bullish_Engulfing'], np.nan)
    
    apds.append(mpf.make_addplot(bullish_engulfing_markers, type='scatter', marker='^', markersize=200, color='yellow', label='Bullish Engulfing'))
    
    # Plot the chart
    fig, axes = mpf.plot(plot_data, type='candle', style='yahoo', volume=True, addplot=apds,
                         figsize=(16, 10), title=f'\n{ticker} - Bullish Engulfing Pattern Analysis',
                         returnfig=True)
    
    # Add support and resistance lines
    ax = axes[0]
    if not np.isnan(data['Support'].iloc[-1]):
        ax.axhline(y=data['Support'].iloc[-1], color='g', linestyle='--', linewidth=1, label='Support')
    if not np.isnan(data['Resistance'].iloc[-1]):
        ax.axhline(y=data['Resistance'].iloc[-1], color='r', linestyle='--', linewidth=1, label='Resistance')
    
    # Adjust legend
    ax.legend(loc='upper left', fontsize=8)
    
    return fig

def analyze_tickers(tickers, period="1mo", interval="1d", last_n_candles=5, n_peaks=3, tolerance=0.01):
    results = []
    for ticker in tqdm(tickers, desc="Analyzing tickers"):
        logging.info(f"Analyzing {ticker}...")
        data = fetch_data(ticker, period, interval)
        
        if data is None or data.empty:
            logging.warning(f"No data available for {ticker}")
            continue
        
        data = identify_candlestick_patterns(data)
        data = identify_peaks_valleys(data)
        data = identify_support_resistance(data)
        data = calculate_ema(data)
        
        # Fit trendline
        trendline = fit_trendline(data, n_peaks=n_peaks, tolerance=tolerance)
        if trendline is not None:
            data['Trendline'] = trendline
        
        # Check if there's a Bullish Engulfing pattern in the last n candles
        if data['Bullish_Engulfing'].iloc[-last_n_candles:].any():
            logging.info(f"Bullish Engulfing pattern detected for {ticker}")
            results.append((ticker, data))
        else:
            logging.info(f"No Bullish Engulfing pattern detected for {ticker}")
    
    return results

# Streamlit app
st.title('Candle Pattern Analyzer')

# User input for tickers
ticker_input = st.text_input('Enter tickers separated by commas', 'AAPL,MSFT,GOOGL')
tickers = [ticker.strip() for ticker in ticker_input.split(',')]

# User input for parameters
period = st.selectbox('Select time period', ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=1)
interval = st.selectbox('Select interval', ['1d', '1wk', '1mo'], index=0)
last_n_candles = st.slider('Number of recent candles to check for pattern', 1, 10, 2)
n_peaks = st.slider('Number of peaks for trendline', 2, 10, 3)
tolerance = st.slider('Tolerance for trendline', 0.01, 0.10, 0.01, 0.01)

if st.button('Analyze Tickers'):
    results = analyze_tickers(tickers, period, interval, last_n_candles, n_peaks, tolerance)
    
    if results:
        for ticker, data in results:
            st.subheader(f'Analysis for {ticker}')
            fig = plot_chart(data, ticker)
            st.pyplot(fig)
    else:
        st.write('No Bullish Engulfing patterns detected for the given tickers and parameters.')