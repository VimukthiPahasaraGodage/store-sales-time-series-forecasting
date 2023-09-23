import pandas as pd

# load the data
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}

past_sales = pd.read_csv(
    'train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
past_sales = past_sales.set_index('date').to_period('D')

forecast_sales = pd.read_csv(
    'test.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
forecast_sales = forecast_sales.set_index('date').to_period('D')
