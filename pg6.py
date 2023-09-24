import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller


def get_params_for_sarimax(store_number, train_):
    params_list = []
    for family in train_['family'].unique():
        print(f'store {store_number}/54    family {family}: Starting...')
        train = train_[(train_['store_nbr'] == str(store_number)) & (train_['family'] == family)]
        train = train['sales'].tolist()

        if train.count(train[0]) != len(train):
            adftest_ = adfuller(train, autolag='AIC')
            p_value_adftest = adftest_[1]

            if p_value_adftest > 0.05:
                smodel = auto_arima(train,
                                    start_p=2,
                                    d=None,
                                    start_q=2,
                                    max_p=4,
                                    max_d=2,
                                    max_q=4,
                                    m=30,
                                    start_P=1,
                                    D=None,
                                    start_Q=1,
                                    max_P=2,
                                    max_D=1,
                                    max_Q=2,
                                    seasonal=True,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
            else:
                smodel = auto_arima(train,
                                    start_p=2,
                                    d=0,
                                    start_q=2,
                                    max_p=4,
                                    max_d=0,
                                    max_q=4,
                                    m=1,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
            order = smodel.order
            seasonal_order = smodel.seasonal_order
            params = [store_number, family, order, seasonal_order]
            params_list.append(params)
        else:
            params = [store_number, family, -1, -1]
            params_list.append(params)

        print(f'store {store_number}/54    family{family} Ended')

    # save the model params in a csv file
    model_params = pd.DataFrame(params_list, columns=['store_nbr', 'family', 'order', 'seasonal_order'])
    model_params.to_csv(str(store_number) + '.csv')
    print('params saved successfully!')


if __name__ == '__main__':
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

    # create a process for each store number and start all the processes simultaneously
    index = 0
    for store_nbr in range(25, 30):
        get_params_for_sarimax(store_nbr, past_sales)

    print('End of fitting...')
