from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1200)


def create_and_train_model(x_learn, y_learn, model, n_cores):
    """General method to create and train model"""
    print(model.fit(x_learn, y_learn))
    start_time = datetime.now()
    c_val = cross_val_score(model, x_learn, y_learn, cv=10, n_jobs=n_cores)
    end_time = datetime.now()
    print(type(model).__name__, "with n_jobs =", n_cores, "took:", (end_time.second - start_time.second), "seconds")
    print(type(model).__name__, "cross_val_score:", c_val.mean())
    return model, c_val


def model_predict_for_training(x, y, model):
    model_result = model.predict_proba(x)[:]
    dt_result = pd.DataFrame(model_result[:, 1])
    dt_result['Predict'] = dt_result[0].map(lambda s: 1 if s >= 0.6 else 0)
    dt_result['testAnswer'] = pd.DataFrame(y)
    dt_result['Correct'] = np.where((dt_result['Predict'] == dt_result['testAnswer']), 1, 0)
    return dt_result


def model_predict(x, model):
    model_result = model.predict_proba(x)[:]
    dt_result = pd.DataFrame(model_result[:, 1])
    return dt_result


def model_save(path, model):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_trained_model(path):
    return pickle.load(open(path, 'rb'))


def get_df_template():
    data = pd.read_csv("../dataset/tmdb_5000_movies_result.csv")
    data = data.drop(labels=["Nice"], axis=1)
    template_row = data.iloc[0]
    template_row[template_row != 0] = 0
    return template_row


def get_trained_rfc():
    data = pd.read_csv("../dataset/tmdb_5000_movies_result.csv")
    print(data.info())

    # TODO: Взять из своего старого кернела иной способ разбиения
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
    train, test = data[data['is_train'] == True], data[data['is_train'] == False]

    train.drop(['is_train'], axis=1, inplace=True)
    test.drop(['is_train'], axis=1, inplace=True)

    train["Nice"] = train["Nice"].astype(int)

    Y_train = train["Nice"]
    X_train = train.drop(labels=["Nice"], axis=1)

    rfc = RandomForestClassifier()
    rfc, rfc_c_val = create_and_train_model(X_train, Y_train, rfc, -1)

    return rfc
