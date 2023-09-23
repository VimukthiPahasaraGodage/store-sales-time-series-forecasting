import pandas as pd

dataframes = []
for df_file in ['a1_params.csv', 'a2_params.csv', 'a3_params.csv']:
    dataframes.append(pd.read_csv(df_file))

final = pd.concat(dataframes, axis=0)

final.reset_index(drop=True, inplace=True)

final.to_csv('parameters.csv')
