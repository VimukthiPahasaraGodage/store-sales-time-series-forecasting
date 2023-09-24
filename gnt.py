import pandas as pd

holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
oil = pd.read_csv('oil.csv', parse_dates=['date'])
oil.columns = ['date', 'oil']
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])
stores = pd.read_csv('stores.csv')
transaction = pd.read_csv('transactions.csv', parse_dates=['date'])

train = train.merge(stores, on='store_nbr')
train = train.merge(oil, on='date', how='left')
holidays = holidays.rename(columns={'type': 'holiday_type'})
train = train.merge(holidays, on='date', how='left')

train['oil'] = train['oil'].interpolate(limit_direction='both')

train['holiday'] = train['holiday_type'].notnull().astype(int)

train.to_csv('new_train.csv')
