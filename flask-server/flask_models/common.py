import pandas as pd

def load_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except:
        try:
            df = pd.read_csv(path+'.zip')
            return df.to_csv(path, encoding='utf-8', index=False)
        except:
            return pd.DataFrame()