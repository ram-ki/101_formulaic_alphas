Algorithmic trading strategies are driven by signals that indicate when to buy or
sell assets to generate superior returns relative to a benchmark such as an index.

The portion of an asset’s return that is not explained by exposure to this benchmark
is called alpha, and hence the signals that aim to produce such uncorrelated returns
are also called alpha factors.

Ultimately, the goal of active investment management is to generate alpha, defined
as portfolio returns in excess of the benchmark used for evaluation. The fundamental
law of active management postulates that the key to generating alpha is having
accurate return forecasts combined with the ability to act on these forecasts
In this project we have attempted to develop the code and results based on the
formulaic alphas mentioned in the paper 101 Formulaic Alphas (https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf)

Install necessary libraries
```
pip3 install -r requirements.txt
```

To generate the backtest results modify the appropriate alpha call in formulaic_alphas.py and run the command
```
python3 main.py
```

