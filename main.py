import os

import pandas as pd
import pytz

from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

from weight_generator import model
import configparser
import ast


class template:

    def __init__(self, AlphaModel_name):
    
        
        self.AlphaModel_name = AlphaModel_name

        config = configparser.ConfigParser()
        config.read('config.ini')
        
        self.start_str = config['dates']['start_str']
        self.burn_it_str = config['dates']['burn_in_str']
        self.end_str = config['dates']['end_str']

        self.strategy_symbols = ast.literal_eval(config['strategy_symbols']['symbols'])

        self.csv_dir_path = config['data_filename']['file']

        
    def run(self):
        
        
        # Duration of the backtest
        start_dt = pd.Timestamp(self.start_str, tz=pytz.UTC)
        burn_in_dt = pd.Timestamp(self.burn_it_str, tz=pytz.UTC)
        end_dt = pd.Timestamp(self.end_str, tz=pytz.UTC)


        # Construct the symbols and assets necessary for the backtest
        # This utilises the SPDR US sector ETFs, all beginning with XL
        assets = ['EQ:%s' % symbol for symbol in self.strategy_symbols]

        # As this is a dynamic universe of assets (XLC is added later)
        # we need to tell QSTrader when XLC can be included. This is
        # achieved using an asset dates dictionary
        asset_dates = {asset: start_dt for asset in assets}
        strategy_universe = DynamicUniverse(asset_dates)

        # To avoid loading all CSV files in the directory, set the
        # data source to load only those provided symbols
        csv_dir = os.environ.get(self.csv_dir_path[1:-2], self.csv_dir_path)
        print(self.csv_dir_path)
        print(csv_dir)
        strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols = self.strategy_symbols)
        strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])


        # Generate the alpha model instance for the top-N momentum alpha model
        strategy_alpha_model = model(assets, self.csv_dir_path, self.AlphaModel_name)


        # Construct the strategy backtest and run it
        strategy_backtest = BacktestTradingSession(
            start_dt,
            end_dt,
            strategy_universe,
            strategy_alpha_model,
            rebalance='end_of_month',
            long_only=True,
            cash_buffer_percentage=0.01,
            burn_in_dt=burn_in_dt,
            data_handler=strategy_data_handler
        )
        strategy_backtest.run()

        # Construct benchmark assets (buy & hold SPY)
        benchmark_symbols = ['SPY']
        benchmark_assets = ['EQ:SPY']
        benchmark_universe = StaticUniverse(benchmark_assets)
        benchmark_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=benchmark_symbols)
        benchmark_data_handler = BacktestDataHandler(benchmark_universe, data_sources=[benchmark_data_source])

        # Construct a benchmark Alpha Model that provides
        # 100% static allocation to the SPY ETF, with no rebalance
        benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
        benchmark_backtest = BacktestTradingSession(
            burn_in_dt,
            end_dt,
            benchmark_universe,
            benchmark_alpha_model,
            rebalance='buy_and_hold',
            long_only=True,
            cash_buffer_percentage=0.01,
            data_handler=benchmark_data_handler
        )
        benchmark_backtest.run()

        # Performance Output
        tearsheet = TearsheetStatistics(
            strategy_equity=strategy_backtest.get_equity_curve(),
            benchmark_equity=benchmark_backtest.get_equity_curve(),
            title='alpha'+self.AlphaModel_name
        )
        tearsheet.plot_results()

if __name__ == '__main__':
    obj = template('003')
    obj.run()