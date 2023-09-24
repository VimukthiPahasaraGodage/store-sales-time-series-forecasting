import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

if __name__ == '__main__':
    param_df = pd.read_csv('a2_params.csv')

    dtype = {
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint64',
    }

    past_sales = pd.read_csv(
        'new_train.csv',
        dtype=dtype,
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    past_sales = past_sales.set_index('date').to_period('D')

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

        train_length = len(train)

        test = train.iloc[train_length - 16:, :]
        train = train.iloc[:train_length - 16, :]

        start_ = len(train)
        end_ = len(train) + len(test) - 1

        if type(order) is tuple and type(seasonal_order) is tuple:
            try:
                sarima_model = SARIMAX(train['sales'], order=order, seasonal_order=seasonal_order,
                                       exog=train[['holiday', 'onpromotion', 'oil']])
                sarima_model_fit = sarima_model.fit()
            except:
                print('###############################################################################################')
                print('############################# Error : Diverging from default ##################################')
                sarima_model = SARIMAX(train['sales'], order=order, seasonal_order=seasonal_order,
                                       enforce_stationarity=False, exog=train[['holiday', 'onpromotion', 'oil']])
                sarima_model_fit = sarima_model.fit()
                print('############################## Diverge Successful! ############################################')
                print('###############################################################################################')

            pred = sarima_model_fit.predict(start=start_, end=end_, dynamic=False, typ="levels", exog=test[['holiday', 'onpromotion', 'oil']])
        else:
            pred = [(train['sales'].tolist())[0] for i in range(start_, end_ + 1)]

        test['pred_sales'] = pred
        result_dfs.append(test)

    result = pd.concat(result_dfs, axis=0)
    result.to_csv('a3_v.csv')
    print('successfully saved the result!')
