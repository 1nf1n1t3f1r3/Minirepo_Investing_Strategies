import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas_market_calendars as mcal
import os
import json

# Global variables for capital management
initial_capital = 100000  # Starting with $100,000
available_capital = initial_capital  # Track available capital
max_concurrent_trades = 20  # Example: max 2 open trades across all tickers
allocation_per_trade = 0.05  # 50% of available capital per trade

# Global position tracker
global_open_trades = []  # List to track open trades across all tickers

# Liquidity and turnover thresholds
min_avg_daily_volume = 1000000
min_turnover = 1000000

# Strategy Parameters
ma_length = 100
upper_multiplier = 3.0
lower_multiplier = 1.0

min_days_below_upper = 15
min_band_position_threshold = 100  # Minimum threshold (starting from 100)
max_band_position_threshold = 1000  # Maximum threshold

# # dates
# download_end_date = '2024-10-30'

# Set up trading calendar for NYSE
nyse = mcal.get_calendar("XNYS")

# Goes with def fetch_stock_data
# Define storage folder for cached data
storage_folder = "D:\\trading_data\\ticker_data_cache_unadjusted"
metadata_file = "D:\\trading_data\\ticker_data_cache\\ticker_metadata.json"
os.makedirs(storage_folder, exist_ok=True)

# Initialize metadata dictionary
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        ticker_metadata = json.load(f)
else:
    ticker_metadata = {}

def fetch_stock_data(ticker, start_date, end_date):
    # Remove any whitespace from ticker
    ticker = ticker.strip()

    file_path = os.path.join(storage_folder, f"{ticker}.csv")

    # Determine start_date_extended using metadata or fetching if necessary
    if start_date:
        start_date_extended = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=125)).strftime('%Y-%m-%d')
    else:
        # Retrieve earliest date from metadata if available
        if ticker in ticker_metadata:
            # Use only the 'earliest_date' part of the dictionary
            start_date_extended = ticker_metadata[ticker].get("earliest_date")
            print(f"Using cached earliest date for {ticker}: {start_date_extended}")
        else:
            # Call YF to get the earliest date for this ticker if not in metadata
            print(f"Fetching earliest date for {ticker} from Yahoo Finance")
            df_full_history = yf.download(ticker, start=None, end=None, auto_adjust=False, multi_level_index=False)
            if not df_full_history.empty:
                start_date_extended = df_full_history.index.min().strftime('%Y-%m-%d')
                # Store both earliest_date and average_performance
                ticker_metadata[ticker] = {"earliest_date": start_date_extended, "average_performance": None}

                print(f"Cached earliest date for {ticker}: {start_date_extended}")
            else:
                print(f"No data available for {ticker}")
                return pd.DataFrame()  # Return empty DataFrame if no data found

    # Set end_date to today if not provided
    if end_date:
        end_date_extended = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=0)).strftime('%Y-%m-%d')
    else:
        end_date_extended = (date.today() + timedelta(days=0)).strftime('%Y-%m-%d')

    # Load cached data if file exists
    if os.path.exists(file_path):
        df_cached = pd.read_csv(file_path, index_col="Date", parse_dates=True)

        # Ensure tz-naive and sort by date
        df_cached.index = df_cached.index.tz_localize(None)
        df_cached.sort_index(inplace=True)

        # Determine the full range of trading days needed
        actual_start = start_date_extended
        actual_end = end_date_extended
        trading_days = nyse.valid_days(start_date=actual_start, end_date=actual_end).tz_localize(None)

        # Identify missing trading days by comparing against cached data
        missing_dates = pd.DatetimeIndex(trading_days).difference(df_cached.index)

        if not missing_dates.empty:
            print(f"Detected gaps in cached data for {ticker}. Missing dates: {missing_dates}")

            # Get the earliest and latest missing dates to form a single download range
            missing_start = missing_dates.min()
            missing_end = missing_dates.max()

            # For recent missing dates, use a wider range to avoid single-day issues
            if (date.today() - missing_end.date()).days <= 7:
                missing_start = missing_start - timedelta(days=3)
                missing_end = date.today() + timedelta(days=1)

            missing_start_str = missing_start.strftime('%Y-%m-%d')
            missing_end_str = missing_end.strftime('%Y-%m-%d')
            print(f"Fetching data from {missing_start_str} to {missing_end_str} for {ticker}")

            # Download the full range of missing data in one call
            df_new = yf.download(ticker, start=missing_start_str, end=missing_end_str, auto_adjust=False, multi_level_index=False)

            # Only concatenate if df_new has data
            if not df_new.empty:
                df_cached = pd.concat([df_cached, df_new]).sort_index()
                df_cached = df_cached[~df_cached.index.duplicated(keep='first')]

                # Save updated data back to file
                df_cached.to_csv(file_path)
            else:
                print(f"No new data found for {ticker} from {missing_start_str} to {missing_end_str}")

            df_result = df_cached
        else:
            # Cached data is complete, use it directly
            df_result = df_cached.loc[start_date_extended:end_date_extended] if start_date_extended and end_date_extended else df_cached
            print(f"Full data loaded from cache for {ticker}")

    else:
        # If no cached file, download entire date range
        print(f"Downloading full data range for {ticker}")
        df_result = yf.download(ticker, start=start_date_extended, end=end_date_extended, auto_adjust=False, multi_level_index=False)

        # Remove duplicates if any
        if not df_result.empty:
            df_result = df_result[~df_result.index.duplicated(keep='first')]
            df_result.to_csv(file_path)

    # Final duplicate removal for robustness and ensure a clean return
    df_result = df_result[~df_result.index.duplicated(keep='first')]
    return df_result


# EMA
def moving_average(series, window):
    return series.ewm(span=window, adjust=False).mean()  # EMA
    # return series.rolling(window=window).mean()  # SMA

# Function to calculate Bollinger Bands
def bollinger_bands(df, length=100, upper_multiplier=3.0, lower_multiplier=1.0):
    basis = moving_average(df['Close'], window=length)
    std_dev = df['Close'].rolling(window=length).std()

    upper_band = basis + (upper_multiplier * std_dev)
    lower_band = basis - (lower_multiplier * std_dev)

    df['basis'] = basis
    df['upper'] = upper_band
    df['lower'] = lower_band

    # Calculate Band_Position for each row in the DataFrame
    df['Band_Position'] = ((df['Close'] - df['lower']) / (df['upper'] - df['lower'])) * 100  # Scale position between 0-100

    # Drop rows with NaN values resulting from the calculations
    df.dropna(subset=['basis', 'upper', 'lower', 'Band_Position'], inplace=True)

    return df



# Combine and clean data for multiple tickers
def get_combined_data(input_file):
    input_data = pd.read_csv(input_file)
    all_data = []

    for _, row in input_data.iterrows():
        ticker = row['Ticker']
        start_date = row['Date']
        end_date = row['Date2'] if not pd.isna(row['Date2']) else row['Date3']

        try:
            start_date_extended = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=50)).strftime('%Y-%m-%d')
            end_date_extended = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=100)).strftime('%Y-%m-%d')
            df = fetch_stock_data(ticker, start_date_extended, end_date_extended)

            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Filter data to match exactly the desired date range for this ticker
            df = df[(df.index >= start_date_extended) & (df.index <= end_date_extended)]

            df['Date'] = df.index
            df['Ticker'] = ticker

            # Calculate Keltner Channel bands
            df = bollinger_bands(df)

            # Verify Band_Position exists after keltner_channel calculation
            if 'Band_Position' not in df.columns:
                print(f"Error: Band_Position is missing for {ticker}")
                continue


            # Shift values for entry conditions
            df['prev_close'] = df['Close'].shift(1)
            df['prev_upper'] = df['upper'].shift(1)

            relevant_data = df[['Date', 'Close', 'upper', 'lower', 'Ticker', 'prev_close', 'prev_upper', 'Band_Position']]

            all_data.append(relevant_data)

        except Exception as e:

            print(f"Error processing {ticker}: {e}")

    combined_data = pd.concat(all_data, ignore_index=True)

    combined_data.sort_values(by='Date', inplace=True)

    # Drop rows with NaN in 'prev_close', 'prev_upper', or 'Band_Position' created by shifting
    combined_data.dropna(subset=['prev_close', 'prev_upper', 'Band_Position'], inplace=True)

    return combined_data


# Process trades day by day with streak-based entry logic
def process_trades_day_by_day(combined_data):
    global available_capital

    global min_days_below_upper
    global min_band_position_threshold
    global max_band_position_threshold

    global_open_trades = []
    trades = []
    days_below_upper = {}

    for current_date, daily_data in combined_data.groupby('Date'):
        open_trades_today = len(global_open_trades)

        for _, row in daily_data.iterrows():
            ticker = row['Ticker']
            close_price = row['Close']
            upper_band = row['upper']
            lower_band = row['lower']
            prev_close = row['prev_close']
            prev_upper = row['prev_upper']
            band_position = row['Band_Position']  # Access Band_Position value for the current bar

            # Initialize streak counter if not present
            if ticker not in days_below_upper:
                days_below_upper[ticker] = 0

            # Update the streak based on previous bar's close relative to the upper band
            if prev_close > prev_upper:
                days_below_upper[ticker] = 0
            else:
                days_below_upper[ticker] += 1

            # Check if we already have an open trade for this ticker
            ticker_open_trade = any(trade['Ticker'] == ticker for trade in global_open_trades)

            # Entry conditions, including the new Band_Position check
            if (
                close_price > upper_band  # Current bar close crosses above the upper band
                and open_trades_today < max_concurrent_trades
                and not ticker_open_trade
                and prev_close < prev_upper  # Previous bar below upper band
                and days_below_upper[ticker] >= min_days_below_upper  # Previous streak requirement
                and band_position >= min_band_position_threshold  # Band_Position threshold
                and band_position <= max_band_position_threshold  # Band_Position threshold
            ):

                # Allocate capital and create a new trade
                trade_allocation = allocation_per_trade * available_capital
                entry_price = close_price
                entry_date = current_date
                initial_stop_price = lower_band

                global_open_trades.append({
                    'Ticker': ticker,
                    'Entry Date': entry_date,
                    'Entry Price': entry_price,
                    'Position Size': trade_allocation,
                    'Initial Stop Price': initial_stop_price
                })

                open_trades_today += 1
                print(f"Opened trade for {ticker} on {entry_date} at price {entry_price}")

            # Close open trades if price goes below the lower band
            for trade in global_open_trades[:]:
                if close_price < lower_band and trade['Ticker'] == ticker:
                    exit_price = close_price
                    trade_result = exit_price - trade['Entry Price']
                    shares_bought = trade['Position Size'] / trade['Entry Price']
                    profit_loss = shares_bought * trade_result
                    available_capital += profit_loss
                    exit_date = current_date

                    risked_capital = shares_bought * (trade['Entry Price'] - trade['Initial Stop Price'])
                    r_multiple = profit_loss / risked_capital if risked_capital > 0 else np.nan

                    # Append completed trade details to trades list
                    trades.append({
                        'Ticker': trade['Ticker'],
                        'Entry Date': trade['Entry Date'],
                        'Exit Date': exit_date,
                        'Allocated Capital': trade['Position Size'],
                        'Shares Bought': shares_bought,
                        'Entry Price': trade['Entry Price'],
                        'Exit Price': exit_price,
                        'Trade Result': trade_result,
                        'Profit/Loss': profit_loss,
                        'Risked Capital': risked_capital,
                        'Initial Stop': trade['Initial Stop Price'],
                        'R_Multiple': r_multiple
                    })

                    # Remove closed trade from open trades list
                    global_open_trades.remove(trade)
                    open_trades_today -= 1

    # Return DataFrame with all recorded trades
    trades_df = pd.DataFrame(trades)
    return trades_df



# Function to calculate performance metrics
def calculate_performance_metrics(trades_df):
    # Calculate total return and time period
    total_profit = trades_df['Profit/Loss'].sum()
    final_capital = initial_capital + total_profit

    # Avoid division by zero and invalid operations
    total_years = (trades_df['Exit Date'].max() - trades_df['Entry Date'].min()).days / 365.25
    if total_years <= 0:
        cagr = np.nan
    else:
        cagr = (final_capital / initial_capital) ** (1 / total_years) - 1

    # Max Drawdown
    equity_curve = initial_capital + trades_df['Profit/Loss'].cumsum()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # MAR (Managed Account Ratio)
    mar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Winrate
    winrate = (trades_df['Profit/Loss'] > 0).mean()

    # Calculate trade returns as Profit/Loss divided by Allocated Capital
    trades_df['Return'] = trades_df['Profit/Loss'] / trades_df['Allocated Capital']

    # Sharpe Ratio (assuming risk-free rate = 0) based on trade returns
    avg_return = trades_df['Return'].mean()
    return_std = trades_df['Return'].std()
    sharpe_ratio = avg_return / return_std if return_std != 0 else np.nan

    # Sortino Ratio (assuming minimum acceptable return = 0)
    downside_returns = trades_df.loc[trades_df['Return'] < 0, 'Return']
    downside_std = downside_returns.std()
    sortino_ratio = avg_return / downside_std if downside_std != 0 else np.nan

    # CAGMAR Ratio (CAGR / Max Drawdown)
    cagmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Omega Ratio (threshold = 0)
    threshold_return = 0  # Typically 0
    excess_returns = trades_df['Return'] - threshold_return
    positive_sum = excess_returns[excess_returns > 0].sum()
    negative_sum = abs(excess_returns[excess_returns < 0].sum())
    omega_ratio = positive_sum / negative_sum if negative_sum != 0 else np.nan

    # Add performance metrics to every row
    trades_df['CAGR'] = cagr
    trades_df['MAR'] = mar
    trades_df['Max Drawdown'] = max_drawdown
    trades_df['Winrate'] = winrate
    trades_df['Sharpe Ratio'] = sharpe_ratio
    trades_df['Sortino Ratio'] = sortino_ratio
    trades_df['CAGMAR'] = cagmar
    trades_df['Omega Ratio'] = omega_ratio

    return trades_df



# Main function to process the input file
def process_input_file(input_file, ma_length, upper_multiplier, lower_multiplier):
    combined_data = get_combined_data(input_file)
    trades_df = process_trades_day_by_day(combined_data)

    # Add performance metrics to the final output
    trades_df = calculate_performance_metrics(trades_df)

    # Order the columns for output
    trades_df = trades_df[
        ['Ticker', 'Entry Date', 'Exit Date', 'Allocated Capital', 'Shares Bought', 'Entry Price', 'Exit Price',
         'Trade Result', 'Profit/Loss', 'Risked Capital', 'Initial Stop', 'R_Multiple', 'CAGR', 'MAR', 'Max Drawdown',
         'Winrate', 'Sharpe Ratio', 'Sortino Ratio', 'CAGMAR', 'Omega Ratio']]

    # Write output to CSV
    output_file = input_file.split(".")[0] + "_BBO_BB-res.csv"
    trades_df.to_csv(output_file, index=False)

    return trades_df


# Example usage:
input_file = 'small_SPY_500_201020.csv'
output_df = process_input_file(input_file, ma_length, upper_multiplier, lower_multiplier)

