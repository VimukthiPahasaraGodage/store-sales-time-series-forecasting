import pandas as pd

dataframes = []
for df_file in ['a1.csv', 'a2.csv', 'a3.csv']:
    dataframes.append(pd.read_csv(df_file))

final = pd.concat(dataframes, axis=0)
final.to_csv('final.csv')

final.drop(['date', 'store_nbr', 'family', 'onpromotion'], inplace=True, axis=1)

final.sort_values('id', inplace=True)

final = final[['id', 'sales']]

final.to_csv('submission.csv', index=False)
