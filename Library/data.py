import pandas as pd

def get_data(filename, rename_columns={}, remap={}, combinator={}, **kwargs):
    df = pd.read_csv(filename, index_col=kwargs.get("index_col", None))

    # re-format data
    df.rename(columns=rename_columns, inplace=True)

    # remap cat var designator
    for column, col_map in remap.items():
        df[column] = df[column].map(col_map)

    for combo, columns in combinator.items():
        df[combo] = df[columns].sum(axis=1)

    return df

def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy(deep=True)

    return df_clean
