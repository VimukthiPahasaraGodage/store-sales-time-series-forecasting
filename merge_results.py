import pandas as pd

dataframes = []
for df_file in ['a1.csv', 'a2.csv', 'a3.csv']:
    dataframes.append(pd.read_csv(df_file))

final = pd.concat(dataframes, axis=0)

final.sort_index(inplace=True)

final = final[['id', 'sales']]

final.to_csv('submission.csv')
