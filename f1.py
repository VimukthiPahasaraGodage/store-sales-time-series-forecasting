import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

if __name__ == '__main__':
    param_dfs_range = [range(1, 5), range(15, 20), range(30, 35), range(45, 50)]
    dfs = []

    for range_ in param_dfs_range:
        for i in range_:
            dfs.append(pd.read_csv(str(i) + '.csv'))

    param_df = pd.concat(dfs, axis=0)
    param_df.to_csv('a1_params.csv')

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

    result_dfs = []

    for index, row in param_df.iterrows():
        store_nbr = row['store_nbr']
        family = row['family']
        order = row['order']
        seasonal_order = row['seasonal_order']
        if type(order) is str:
            order = eval(order)
            seasonal_order = eval(seasonal_order)

        train = past_sales[(past_sales['store_nbr'] == str(store_nbr)) & (past_sales['family'] == family)]
        train = train['sales'].tolist()

        test = forecast_sales[(forecast_sales['store_nbr'] == str(store_nbr)) & (forecast_sales['family'] == family)]

        start_ = len(train)
        end_ = len(train) + len(test) - 1

        if type(order) is tuple and type(seasonal_order) is tuple:
            sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            sarima_model_fit = sarima_model.fit()

            pred = sarima_model_fit.predict(start=start_, end=end_, dynamic=False, typ="levels")
        else:
            pred = [train[0] for i in range(start_, end_ + 1)]

        test['sales'] = pred
        result_dfs.append(test)

    result = pd.concat(result_dfs, axis=0)
    result.to_csv('a1.csv')
    print('successfully saved the result!')
