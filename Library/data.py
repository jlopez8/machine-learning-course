import pandas as pd

def get_data(filename, rename_columns={}, remap={}, combinator={}):
    df = pd.read_csv(filename)

    # re-format data
    df.rename(columns=rename_columns, inplace=True)

    # remap cat var designator
    for column, col_map in remap.items():
        df[column] = df[column].map(col_map)

    for combo, columns in combinator.items():
        df[combo] = df[columns].sum(axis=1)

    return df
