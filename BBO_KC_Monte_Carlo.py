import pandas as pd
import numpy as np
import random
from BBO_KC import process_input_file  # Import the first script function

# Global variables. Same as strategy script
initial_capital = 100000  # Starting capital

allocation_per_trade = 0.05  # 50% of available capital per trade
max_concurrent_trades = 20  # Max number of concurrent trades

# Global variables. Monte Carlo
n_iterations = 88  # Number of Monte Carlo iterations
skip_trade_probability = 0.2  # Probability of skipping a trade


# Monte Carlo simulation: skip trades randomly but do not shuffle
def monte_carlo_simulation(trades_df):
    results = []

    for iteration in range(n_iterations):
        available_capital = initial_capital
        equity_curve = [available_capital]
        closed_trades = []
        total_r_multiples = 0
        r_multiples_count = 0

        for index, trade in trades_df.iterrows():
            if random.random() < skip_trade_probability:
                continue

            allocated_capital = available_capital * allocation_per_trade
            shares_bought = allocated_capital / trade['Entry Price']
            profit_loss = shares_bought * (trade['Exit Price'] - trade['Entry Price'])
            available_capital += profit_loss

            r_multiple = trade['R_Multiple']
            total_r_multiples += r_multiple
            r_multiples_count += 1

            trade_return = profit_loss / allocated_capital
            closed_trades.append({'Profit/Loss': profit_loss, 'Return': trade_return})

            equity_curve.append(available_capital)

        total_profit = available_capital - initial_capital
        total_years = (trades_df['Exit Date'].max() - trades_df['Entry Date'].min()).days / 365.25

        cagr = (available_capital / initial_capital) ** (1 / total_years) - 1 if total_years > 0 else np.nan
        max_drawdown = min([(min(equity_curve[i:]) - max(equity_curve[:i + 1])) / max(equity_curve[:i + 1])
                            for i in range(len(equity_curve) - 1)], default=0)
        mar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

        returns = [trade['Return'] for trade in closed_trades]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = avg_return / return_std if return_std != 0 else np.nan

        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns)
        sortino_ratio = avg_return / downside_std if downside_std != 0 else np.nan

        cagmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

        positive_sum = sum([r for r in returns if r > 0])
        negative_sum = abs(sum([r for r in returns if r < 0]))
        omega_ratio = positive_sum / negative_sum if negative_sum != 0 else np.nan

        total_r_multiples = total_r_multiples if r_multiples_count > 0 else np.nan
        average_r_multiple = total_r_multiples / r_multiples_count if r_multiples_count > 0 else np.nan

        results.append({
            'Iteration': iteration + 1,
            'Winrate': np.mean([1 if trade['Profit/Loss'] > 0 else 0 for trade in closed_trades]),
            'Total Profit': total_profit,

            'Total R Multiples': total_r_multiples,
            'Average R Multiple': average_r_multiple,

            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,

            'Omega Ratio': omega_ratio,
            'CAGR': cagr,

            'Max Drawdown': max_drawdown,
            'MAR': mar,
            'CAGMAR': cagmar,
        })

    return pd.DataFrame(results)


# Main function to process input and run Monte Carlo
def process_input_with_monte_carlo(input_file):
    # Process the input file to get original trades
    trades_df = process_input_file(input_file, ma_length=20, upper_multiplier=2, lower_multiplier=2)  # Call function from the strategy script

    # Create output file names based on the input file name
    output_file = input_file.split(".")[0] + "_BBO_KC-res.csv"
    monte_carlo_output_file = input_file.split(".")[0] + "_BBO_KC-res-MC.csv"

    # Save the original processed trades
    trades_df.to_csv(output_file, index=False)

    # Run Monte Carlo simulations on the processed trades
    monte_carlo_results = monte_carlo_simulation(trades_df)

    # Save the Monte Carlo results
    monte_carlo_results.to_csv(monte_carlo_output_file, index=False)

    print(f"Original processed trades saved to {output_file}")
    print(f"Monte Carlo results saved to {monte_carlo_output_file}")

# Example usage
input_file = 'SPY_500_2020.csv'
process_input_with_monte_carlo(input_file)
