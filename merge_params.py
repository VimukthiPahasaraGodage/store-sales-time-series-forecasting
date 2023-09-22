import pandas as pd

dataframes = []
for i in range(0, 1781):
    filename = 'params_' + str(i) + '.csv'
    dataframes.append(pd.read_csv(filename))

final = pd.concat(dataframes, axis=0)

final.sort_index(inplace=True)

final.columns = ['id', 'store_nbr', 'family', 'order', 'seasonal_order']

final.to_csv('parameters.csv')
