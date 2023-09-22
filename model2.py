import multiprocessing
from multiprocessing import Pool

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def forecast(store_number, family, index, train, test):
    params_list = []

    print(f'store {store_number}/54    family {family}: Starting...')
    train = train[(train['store_nbr'] == store_number) & (train['family'] == family)]
    train = train['sales'].tolist()

    test = test[(test['store_nbr'] == store_number) & (test['family'] == family)]

    _start_ = len(train)
    _end_ = len(train) + len(test) - 1

    if train.count(train[0]) != len(train):
        adftest_ = adfuller(train, autolag='AIC')
        p_value_adftest = adftest_[1]

        kpsstest = kpss(train)
        p_value_kpss = kpsstest[1]

        if p_value_adftest < 0.05 and p_value_kpss < 0.05:
            smodel = auto_arima(train,
                                start_p=0,
                                d=0,
                                start_q=0,
                                max_p=7,
                                max_d=0,
                                max_q=7,
                                m=1,
                                seasonal=False,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
        else:
            smodel = auto_arima(train,
                                start_p=0,
                                d=None,
                                start_q=0,
                                max_p=7,
                                max_d=3,
                                max_q=7,
                                m=7,
                                start_P=0,
                                D=None,
                                start_Q=0,
                                max_P=7,
                                max_D=3,
                                max_Q=7,
                                seasonal=True,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

        order = smodel.order
        seasonal_order = smodel.seasonal_order

        sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        sarima_model_fit = sarima_model.fit()

        # save order and seasonal order
        params = [store_number, family, order, seasonal_order]
        params_list.append(params)

        pred = sarima_model_fit.predict(start=_start_, end=_end_, dynamic=False, typ="levels")
    else:
        pred = [train[0] for i in range(_start_, _end_ + 1)]

    test['sales'] = pred

    print(f'store {store_number}/54    family {family}: Ended')

    # save the test dataframe as a csv file
    test.to_csv('result_' + str(index) + '.csv')
    print('###########################################################################################')
    print('###########################################################################################')
    print(f'store {store_number}/54    family {family} result saved successfully!')
    print('###########################################################################################')
    print('###########################################################################################')

    # save the model params in a csv file
    model_params = pd.DataFrame(params_list, columns=['store_nbr', 'family', 'order', 'seasonal_order'])
    model_params.to_csv('params_' + str(index) + '.csv')
    print(f'store {store_number}/54    family{family} params saved successfully!')


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

    forecast_sales = pd.read_csv(
        'test.csv',
        dtype=dtype,
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    forecast_sales = forecast_sales.set_index('date').to_period('D')

    num_process = past_sales['family'].nunique() * past_sales['store_nbr'].nunique()
    train_sets = [past_sales.copy() for i in range(0, num_process)]
    test_sets = [forecast_sales.copy() for i in range(0, num_process)]

    print('num train sets: ', len(train_sets), ' num test sets: ', len(test_sets))

    # create a process for each store number and start all the processes simultaneously
    index = 0
    pool = Pool(multiprocessing.cpu_count())
    for store_nbr in past_sales['store_nbr'].unique():
        for family in past_sales['family'].unique():
            pool.apply_async(forecast, args=(store_nbr,
                                             family,
                                             index,
                                             train_sets[index],
                                             test_sets[index]))
            index += 1

    pool.close()

    print('processes are running in background...')

    pool.join()
