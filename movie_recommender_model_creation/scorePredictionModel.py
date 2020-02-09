"""Testing models performance and quality"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # used for plot interactive graph.
import pickle
from datetime import datetime
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1200)


# TODO: CatBoost
# TODO: c_val
# Переписать функции так, чтобы остались только 2: с фит и с проверкой тестовой выборки
# Кажется, он всё таки не так скроссвалидировал - попробовать с тестовой выборкой ещё
def create_and_learn_model(x_learn, y_learn, model, n_cores):
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
    # dt_result['Predict'] = dt_result[0].map(lambda s: 1 if s >= 0.6 else 0)
    return dt_result


def model_save(path, model):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_trained_rfc(path):
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

    # TODO: опять же, взять из своего кернела
    data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
    # data.drop('budget', axis=1, inplace=True)

    train, test = data[data['is_train'] == True], data[data['is_train'] == False]

    train.drop(['is_train'], axis=1, inplace=True)
    test.drop(['is_train'], axis=1, inplace=True)

    train["Nice"] = train["Nice"].astype(int)

    Y_train = train["Nice"]
    X_train = train.drop(labels=["Nice"], axis=1)
    Y_test = test["Nice"]
    X_test = test.drop(labels=["Nice"], axis=1)
    print(X_test.head(5))

    test_result = np.asarray(Y_test)

    rfc = RandomForestClassifier()
    rfc, rfc_c_val = create_and_learn_model(X_train, Y_train, rfc, 1)
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    # rfc_result = model_predict_for_training(X_test, test_result, rfc)
    # print(rfc_result.head())
    # print("RandomForestClassifier \"Correct\" mean:", rfc_result['Correct'].mean())
    # model_save("flask_models/rfc.pkl", rfc)

    return rfc


# data = pd.read_csv("../dataset/tmdb_5000_movies_result.csv")
data = common.load_csv("../dataset/tmdb_5000_movies_result.csv")
print(data.info())
#
# # TODO: опять же, взять из своего кернела
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75
# data.drop('budget', axis=1, inplace=True)

train, test = data[data['is_train'] == True], data[data['is_train'] == False]

train.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

train["Nice"] = train["Nice"].astype(int)

Y_train = train["Nice"]
X_train = train.drop(labels=["Nice"], axis=1)
Y_test = test["Nice"]
X_test = test.drop(labels=["Nice"], axis=1)
print(X_test.head(5))
print(X_test.columns)

test_result = np.asarray(Y_test)
# Decision Tree
dtc = DecisionTreeClassifier()
dtc, dtc_c_val = create_and_learn_model(X_train, Y_train, dtc, -1)
dec_result = model_predict(X_test, test_result,dtc)

print(dec_result.head())
print("DecisionTreeClassifier Tree \"Correct\" mean:", dec_result['Correct'].mean())
# +budget
# 0.7504230118443317
# 0.7533222591362126
# 0.7597109304426377
# 0.790268456375839

# -budget
# 0.7718855218855218
# 0.7592905405405406
# 0.7906382978723404

# K-Nearest Neighbors (60)
knc = KNeighborsClassifier(n_neighbors=60)
knc, knc_c_val = create_and_learn_model(X_train, Y_train, knc, -1)
knc_result = model_predict(X_test, test_result, knc)
print(knc_result.head())
print("KNeighborsClassifier \"Correct\" mean:", knc_result['Correct'].mean())
# +budget
# 0.7934782608695652
# 0.7867383512544803
# 0.7991836734693878

# -budget
# 0.7897977132805629
# 0.806930693069307
# 0.8010160880609652

# Random Forest
# TODO: параметры подобрать в RandomForestClassifier
rfc = RandomForestClassifier()
rfc, rfc_c_val = create_and_learn_model(X_train, Y_train, rfc, -1)
rfc_result = model_predict(X_test, test_result, rfc)
print(rfc_result.head())
print("RandomForestClassifier \"Correct\" mean:", rfc_result['Correct'].mean())
model_save("flask_models/rfc.pkl", rfc)

# Gradient Boosting
gbc = GradientBoostingClassifier()
gbc, gbc_c_val = create_and_learn_model(X_train, Y_train, gbc, -1)
gbc_result = model_predict(X_test, test_result, gbc)
print(gbc_result.head())
print("GradientBoostingClassifier \"Correct\" mean:", gbc_result['Correct'].mean())
# +budget
# 0.8110831234256927

# -budget
# 0.8105175292153589

cv_means = []
cv_means.append(dtc_c_val.mean())
cv_means.append(knc_c_val.mean())
cv_means.append(rfc_c_val.mean())
cv_means.append(gbc_c_val.mean())
cv_std = []
cv_std.append(dtc_c_val.std())
cv_std.append(knc_c_val.std())
cv_std.append(rfc_c_val.std())
cv_std.append(gbc_c_val.std())
res1 = pd.DataFrame({"ACC":cv_means,"Std":cv_std,"Algorithm":["DecisionTree","K-Nearest Neighbors","Random Forest","Gradient Boosting"]})
res1["Type"]= "CrossValid"
g = sns.barplot("ACC","Algorithm",data = res1, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
plt.figure(figsize=(10, 10))
plt.show(g)

tv_means = []
tv_means.append(dec_result['Correct'].mean())
tv_means.append(knc_result['Correct'].mean())
tv_means.append(rfc_result['Correct'].mean())
tv_means.append(gbc_result['Correct'].mean())
res2 = pd.DataFrame({"ACC":tv_means,"Algorithm":["DecisionTree","K-Nearest Neighbors","Random Forest","Gradient Boosting"]})
res2['Type'] = "Test";

g = sns.barplot("ACC","Algorithm",data = res2, palette="Set2",orient = "h")
g.set_xlabel("Mean Accuracy")
g = g.set_title("Test scores")
plt.figure(figsize=(10, 10))
plt.show(g)

res = pd.concat([res1,res2])
g = sns.factorplot(x='Algorithm', y='ACC', hue='Type',palette="coolwarm", data=res, kind='bar')
g.set_xticklabels(rotation=90)
plt.figure(figsize=(10, 10))
plt.show(g)

dec_fea = pd.DataFrame(dtc.feature_importances_)
dec_fea["name"] = list(X_train)
print(dec_fea.sort_values(by=0, ascending=False).head())
g = sns.barplot(0,"name",data = dec_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Decision Tree")
plt.show(g)

rf_fea = pd.DataFrame(rfc.feature_importances_)
rf_fea["name"] = list(X_train)
rf_fea.sort_values(by=0, ascending=False).head()
g = sns.barplot(0,"name",data = rf_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Random Forest")
plt.show(g)

gb_fea = pd.DataFrame(gbc.feature_importances_)
gb_fea["name"] = list(X_train)
gb_fea.sort_values(by=0, ascending=False).head()
g = sns.barplot(0,"name",data = gb_fea.sort_values(by=0, ascending=False)[0:10], palette="Set2",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Gradient Boosting")
plt.show(g)

voting = pd.DataFrame()
voting["knn"] =knc_result['Predict']
voting["GB"] = gbc_result['Predict']
voting["RF"] = rfc_result['Predict']
voting['sum'] = voting.sum(axis=1)

voting['Predict'] = voting['sum'].map(lambda s: 1 if s >= 2 else 0)

voting['testAnswer'] = pd.DataFrame(test_result)

voting['Correct'] = np.where((voting['Predict'] == voting['testAnswer'])
                     , 1, 0)

print(voting.head())
print(voting['Correct'].mean())
print(confusion_matrix(voting['testAnswer'], voting['Predict']))

