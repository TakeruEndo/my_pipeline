import pandas as pd


def get_date_time(df, target):
    # datetime型に変換
    datetime = pd.to_datetime(df[target], format='%Y-%m-%d %H:%M')
    df['datetime'] = datetime
    df['year'] = datetime.map(lambda x: x.year)
    df['month'] = datetime.map(lambda x: x.month)
    df['day'] = datetime.map(lambda x: x.day)
    df['hour'] = datetime.map(lambda x: x.hour)
    df['minute'] = datetime.map(lambda x: x.minute)
    df['second'] = datetime.map(lambda x: x.second)
    df['dayofweek'] = datetime.map(lambda x: x.dayofweek)

    df = df.drop(target, axis=1)
    return df


