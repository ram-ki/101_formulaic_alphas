from datetime import datetime

import pandas as pd
import numpy as np
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel

from formulaic_alphas import FormulaicAlphas


class Model(AlphaModel):

    def __init__(self, assets, csv_dir_path, AlphaModel_name):

        self.assets = assets
        self.csv_dir_path = csv_dir_path

        self.weights_df = {}

        for asset in assets:

            path = csv_dir_path + str(asset[3:]) + '.csv'
            df = pd.read_csv(path)

            obj = FormulaicAlphas(df, AlphaModel_name)
            df = obj.df_data
            self.weights_df[asset] = {}
            for index, row in df.iterrows():
                time = datetime.fromisoformat(row['Date'])
                time = pd.Timestamp(row['Date'], tz=pytz.UTC)
                if (row['weights'] != np.nan):
                    self.weights_df[asset][str(time)] = row['weights']

    def _generate_signals(self, dt, weights):

        weights_sum = 0
        for asset in self.assets:
            weights[asset] = self.weights_df[asset][str(dt)]

            if (weights[asset] >= 0):
                weights_sum += weights[asset]

        for asset in self.assets:
            weights[asset] = max(weights[asset] / weights_sum, 0)
        return weights

    def __call__(self, dt):

        weights = {asset: 0.0 for asset in self.assets}

        dt = dt.replace(hour=0, minute=0, second=0)

        try:
            weights = self._generate_signals(dt, weights)
        except:
            weights = {asset: 0.0 for asset in self.assets}

        return weights
