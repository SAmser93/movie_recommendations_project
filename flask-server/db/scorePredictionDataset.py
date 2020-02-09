import json
import pandas as pd
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1200)
import matplotlib.pyplot as plt
import seaborn as sns  # used for plot interactive graph.
import warnings
warnings.filterwarnings('ignore')


def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


# return a missing value rather than an error upon indexing/key failure
def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])


# TODO: либо переделать это здесь, либо заменить этим мёрдж данных в CBF
def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['companies_1'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['companies_2'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['companies_3'] = tmdb_movies['production_companies'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['keywords'] = tmdb_movies['keywords'].apply(pipe_flatten_names)
    return tmdb_movies


# For list datatypes
def Obtain_list_Occurences(column_name, data):
    listOcc = []
    for i in data[column_name]:
        split_genre = list(map(str, i.split('|')))
        for j in split_genre:
            if j not in listOcc:
                listOcc.append(j)
    return listOcc


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count


def TopTen(theList):
    TopTen = list()
    for i in range(0, 10):
        TopTen.append(theList[i][0])
    return TopTen


def show_heatmap(data):
    plt.figure(figsize=(10, 10))
    g = sns.heatmap(data[list(data)].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.01)
    plt.show(g)


def show_joint(data, x, y):
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(x=x, y=y, data=data);
    plt.show(g)


def show_regplot(data, x, y):
    plt.figure(figsize=(10, 10))
    g = sns.regplot(x=x, y=y, data=data)
    plt.show(g)


def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if key in frequencytable:
            frequencytable[key] += 1
        else:
            frequencytable[key] = 1
    return frequencytable


movies = load_tmdb_movies("../dataset/tmdb_5000_movies.csv")
credits = load_tmdb_credits("../dataset/tmdb_5000_credits.csv")
data = convert_to_original_format(movies, credits)
# print(data.head())

# missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(10))
# 3091 homepage and 844 taglines missing TODO: мб заменить на колонку "есть ли домашняя страничка" вместо тупого дропа?
# TODO: мб дропнуть "language" и оставить только "original lang"?
# TODO: тэглайн нужен ли? Думаю, тоже заменить на "есть ли"
data.drop(['homepage'], axis=1, inplace=True)
# Потому что эти столбцы не должны влиять на предсказывание
data.drop(['vote_count','revenue','popularity'], axis=1, inplace=True)

# Correlation matrix between numerical values
# show_heatmap(data)
# На vote_average влияют title_year и duration
# show_joint(data, "title_year", "vote_average") # В базе в основном фильмы с 97го по сейчас
# show_joint(data, "runtime", "vote_average") # Большая часть фильмов длится в районе полутора часов
data = data[data['vote_average'] != 0] # У многих нет оценки, а нам это важно для расчётов
# show_regplot(data, "title_year", "vote_average") # Старым фильмам ставят оценки повыше
# show_regplot(data, "runtime", "vote_average") # Более продолжительным фильмам - тоже

# В качестве границы "хорошего фильма" я бы взял цифру в 7, при средней оценке в 6.1731...
data['Nice'] = data['vote_average'].map(lambda s :1  if s >= 7 else 0)
# print(data.loc[data['Nice'] == 1, ['original_title', 'vote_average', 'Nice']].head(10))
# print(data['Nice'].value_counts(sort=False))  # 988 Годных фильмов
#
# sd = statistics.stdev(data.budget)
# mean = data.budget.mean()
# max = data.budget.max()
# min = data.budget.min()
#
# data['VeryLowBud'] = data['budget'].map(lambda s: 1 if s < 10000000 else 0)
# data['LowBud'] = data['budget'].map(lambda s: 1 if 10000000 <= s < mean else 0)
# data['MedBud'] = data['budget'].map(lambda s: 1 if  mean <= s < mean+sd  else 0)
# data['HighBud'] = data['budget'].map(lambda s: 1 if mean+sd <= s < 100000000 else 0)
# data['VeryHighBud'] = data['budget'].map(lambda s: 1 if s >= 100000000 else 0)
# g = sns.factorplot(x="VeryLowBud",y="Nice",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("Nice Probability")
# g = sns.factorplot(x="LowBud",y="Nice",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("Nice Probability")
# g = sns.factorplot(x="MedBud",y="Nice",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("Nice Probability")
# g = sns.factorplot(x="HighBud",y="Nice",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("Nice Probability")
# g = sns.factorplot(x="VeryHighBud",y="Nice",data=data,kind="bar",palette = "husl")
# g = g.set_ylabels("Nice Probability")
# plt.show(g)
# Похоже, на фильмы надо либо не тратится вовсе, либо тратиться сильно

# g = sns.factorplot(x="Nice", y = "title_year",data = data, kind="box", palette = "Set3")
# Говорит нам, что всё таки нынче кол-во превышает годноту
# g = sns.factorplot(x="Nice", y = "runtime",data = data, kind="box", palette = "Set3")
# 2-3 часа - оптимальная продолжительность, лучше всё же меньше

# data['ShortMovie'] = data['duration'].map(lambda s: 1 if s < 90 else 0)
# data['NotTooLongMovie'] = data['duration'].map(lambda s: 1 if 90 <= s < 120 else 0)
# data['LongMovie'] = data['duration'].map(lambda s: 1 if   s >= 120  else 0)

# Предлагается превратить типы с перечислением стрингов в отдельные boolean колонки
# TODO: так-то в моём кернеле были более адекватные способы такое проворачивать

# Жанры
genres_list = Obtain_list_Occurences("genres", data)
for word in genres_list:
    data[word] = data['genres'].map(lambda s: 1 if word in str(s) else 0)

set_keywords = set()
for liste_keywords in data['keywords'].str.split('|').values:
    if isinstance(liste_keywords, float):
        continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
# remove null chain entry
set_keywords.remove('')

keyword_occurences, dum = count_word(data, 'keywords', set_keywords)
# makeCloud(keyword_occurences[0:15],"Keywords","White")

# Ключевые слова
for word in TopTen(keyword_occurences):
    data[word] = data['keywords'].map(lambda s: 1 if word in str(s) else 0)

data.drop('keywords', axis=1, inplace=True)
# print(data.head(5))

# Режиссёр
data['director_name'].fillna('unknown', inplace=True)
director_dic = to_frequency_table(data['director_name'])
director_list = list(director_dic.items())
director_list.sort(key=lambda tup: tup[1], reverse=True)

for word in TopTen(director_list):
    data[word] = data['director_name'].map(lambda s: 1 if word in str(s) else 0)

# Актёры
# In this dataset, it contain 3 actor_name columns and a lot of missing value for second and third actor
# we need to combine it first
data['actor_1_name'].fillna('unknown', inplace=True)
data['actor_2_name'].fillna('unknown', inplace=True)
data['actor_3_name'].fillna('unknown', inplace=True)

data['actors_name'] = data[['actor_1_name', 'actor_2_name', 'actor_3_name']].apply(lambda x: '|'.join(x), axis=1)
actor = Obtain_list_Occurences("actors_name", data)
for word in actor:
    data[word] = data['actors_name'].map(lambda s: 1 if word in str(s) else 0)

# Компании
data['companies_1'].fillna('unknown',inplace=True)
data['companies_2'].fillna('unknown',inplace=True)
data['companies_3'].fillna('unknown',inplace=True)
data['companies_name'] = data[['companies_1', 'companies_2', 'companies_3']].apply(lambda x: '|'.join(x), axis=1)
company = Obtain_list_Occurences("companies_name", data)

for word in company:
    data[word] = data['companies_name'].map(lambda s: 1 if word in str(s) else 0)

# Let's delete data that not effect to model or already have a same information but in tidy format column
# data.drop(['id','budget','original_title','overview','spoken_languages','production_companies','production_countries','release_date','status',
#           'tagline','title','vote_average','language','director_name','actor_1_name','actor_2_name','actor_3_name',
#           'companies_1','companies_2','companies_3','country','genres','runtime','actors_name','companies_name'], axis=1, inplace=True)

data.drop(['id', 'original_title', 'overview', 'spoken_languages', 'original_language', 'language',
           'production_companies', 'production_countries', 'release_date', 'status', 'tagline', 'title', 'vote_average',
           'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
           'companies_1', 'companies_2', 'companies_3', 'country', 'genres', 'actors_name', 'companies_name'], axis=1,
           inplace=True)

data['runtime'].fillna(0,inplace=True)

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(5))

print(data.info())

data.to_csv("../dataset/tmdb_5000_movies_result.csv", sep=',', encoding='utf-8')
# # Finally, create model
print("done")