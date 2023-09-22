import pandas as pd

dataframes = []
for i in range(0, 1781):
    filename = 'result_' + str(i) + '.csv'
    dataframes.append(pd.read_csv(filename))

final = pd.concat(dataframes, axis=0)

final.sort_index(inplace=True)

final = final[['id', 'sales']]

final.to_csv('submission.csv')
