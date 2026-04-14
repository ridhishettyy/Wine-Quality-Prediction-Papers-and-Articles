import pandas as pd

# Load dataset
def load_data(path):
    df = pd.read_csv(path, sep=';')
    print(df.columns)
    return df

# Create health risk labels
def create_health_label(df):
    # df['health_risk'] = 0  # Low risk
    df.loc[
        (df['alcohol'] > 5) & (df['sulphates'] > 0.3),
        'health_risk'
    ] = 0

    # Moderate risk (stricter than before)
    df.loc[
        (df['alcohol'] > 11) & (df['sulphates'] > 0.6),
        'health_risk'
    ] = 1

    # High risk (even stricter)
    df.loc[
        (df['alcohol'] > 13) & (df['sulphates'] > 0.8),
        'health_risk'
    ] = 2

    return df