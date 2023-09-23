import pandas as pd
from pmdarima import auto_arima


def get_params_for_sarimax(store_number, train_):
    params_list = []
    for family in train_['family'].unique()[0:1]:
        print(f'store {store_number}/54    family {family}: Starting...')
        train = train_[(train_['store_nbr'] == str(store_number)) & (train_['family'] == family)]
        train = train['sales'].tolist()

        if train.count(train[0]) != len(train):
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
    for store_nbr in range(1, 5):
        get_params_for_sarimax(store_nbr, past_sales)

    print('End of fitting...')
