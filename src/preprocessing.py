import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    return df

def create_health_label(df):
    df['health_risk'] = 0  # Low

    # Moderate
    df.loc[(df['quality'] >= 5) & (df['quality'] <= 6), 'health_risk'] = 1

    # High
    df.loc[df['quality'] >= 7, 'health_risk'] = 2

    return df