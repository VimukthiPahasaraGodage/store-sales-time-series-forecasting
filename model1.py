from multiprocessing import Process

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def forecast(store_number, family_list, family_list_number, train, test):
    family_index = 0
    results = {}
    params_list = []
    for family in family_list:
        family_index += 1
        print(f'store {store_number}/54    family {family_index}/33: Starting...')
        train = train[(train['store_nbr'] == store_number) & (train['family'] == family)]
        _train_ = train['sales'].tolist()

        test = test[(test['store_nbr'] == store_number) & (test['family'] == family)]

        _start_ = len(train)
        _end_ = len(train) + len(test) - 1

        pred = None
        if _train_.count(_train_[0]) != len(_train_):
            adftest_ = adfuller(_train_, autolag='AIC')
            p_value_adftest = adftest_[1]

            kpsstest = kpss(_train_)
            p_value_kpss = kpsstest[1]

            if p_value_adftest < 0.05 and p_value_kpss < 0.05:
                smodel = auto_arima(_train_,
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
                smodel = auto_arima(_train_,
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

            sarima_model = SARIMAX(train['sales'].tolist(), order=order, seasonal_order=seasonal_order)
            sarima_model_fit = sarima_model.fit()

            # save order and seasonal order
            params = [store_number, family, order, seasonal_order]
            params_list.append(params)

            pred = sarima_model_fit.predict(start=_start_, end=_end_, dynamic=False, typ="levels")
        else:
            pred = [_train_[0] for i in range(_start_, _end_ + 1)]

        test['sales'] = pred

        # update the submission dictionary
        for i, j in zip(test['id'].tolist(), test['sales'].tolist()):
            results[i] = j

        print(f'store {store_number}/54    family {family_index}/33: Ended')

    # save the test dataframe as a csv file
    results_of_store = pd.DataFrame.from_dict(results, orient='index')
    results_of_store.to_csv(f'sales_{store_number}_{family_list_number}.csv')

    # save the model params in a csv file
    model_params = pd.DataFrame(params_list, columns=['store_nbr', 'family', 'order', 'seasonal_order'])
    model_params.to_csv(f'params_{store_number}_{family_list_number}.csv')


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

    families = past_sales['family'].unique()
    family_sub_lists = [families[x:x + 5] for x in range(0, len(families), 5)]

    for sub_family_list in family_sub_lists:
        print('family sub lists: ', sub_family_list)

    # create a process for each store number and start all the processes simultaneously
    process_list = []
    for store_nbr in past_sales['store_nbr'].unique():
        i = 1
        for sub_family_list in family_sub_lists:
            process_list.append(Process(target=forecast,
                                        args=(store_nbr,
                                              sub_family_list,
                                              i,
                                              past_sales,
                                              forecast_sales)))

    print('total number of processes: ', len(process_list))

    for process in process_list:
        process.start()

    print('processes are running in background...')
