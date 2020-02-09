from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
import flask_models.contentBasedFiltering as cbf
import flask_models.scorePredictionModelRFC as spm
import pandas as pd
import flask_models.common as common
import datetime

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
# enable CORS
CORS(app)


# TODO: некий метод обновления данных? Мб обращение к БД раз в час
def get_merged_data():
    """Get data from both tables as one dataframe"""
    df1 = common.load_csv('../dataset/tmdb_5000_credits.csv')
    df2 = common.load_csv('../dataset/tmdb_5000_movies.csv')
    if(df1.empty or df2.empty):
        print("No dataset found. Shutting down")
        exit(-1)
    df1.columns = ['id', 'tittle', 'cast', 'crew']
    return df2.merge(df1, on='id')


def get_cosine_sim(df):
    # TODO: упростить до одного цикла - вдруг быстрее будет работать
    # CBF по данным о режиссёре и съёмочной команде
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)

    # Define new director, cast, genres and keywords features that are in a suitable form.
    df['director'] = df['crew'].apply(cbf.get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(cbf.get_list)

    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(cbf.clean_data)

    df['soup'] = df.apply(cbf.create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])

    return cosine_similarity(count_matrix, count_matrix)


global movies_data_frame
global movie_recommendation_model
global cosine_sim
global indices
global movie_prediction_model
global movie_prediction_df_template


def instantiate_models():
    """Method instantiates flask_models and prepares them for use"""
    # Get general data about movies
    global movies_data_frame
    global cosine_sim
    global indices
    global movie_prediction_model
    global movie_prediction_df_template
    movies_data_frame = get_merged_data()
    # Instantiate final merged db
    common.load_csv("../dataset/tmdb_5000_movies_result.csv")

    # Get recommendation model
    cosine_sim = get_cosine_sim(movies_data_frame)
    movies_data_frame = movies_data_frame.reset_index()
    indices = pd.Series(movies_data_frame.index, index=movies_data_frame['title'])
    print("Model for recommendation has been set up")

    # Get success prediction model
    # TODO: если нет файла, то прогнать весь процесс по новой
    movie_prediction_model = spm.load_trained_model("flask_models/rfc.pkl")
    movie_prediction_df_template = spm.get_df_template()
    print("Model for prediction has been set up")


# # TODO: return на фронт?
# def make_cloud(dict,name,color):
#     words = dict()
#
#     for s in dict:
#         words[s[0]] = s[1]
#         wordcloud = WordCloud(
#                       width=1500,
#                       height=750,
#                       background_color=color,
#                       max_words=50,
#                       max_font_size=500,
#                       normalize_plurals=False)
#         wordcloud.generate_from_frequencies(words)
#
#     # fig = plt.figure(figsize=(12, 8))
#     # plt.title(name)
#     # plt.imshow(wordcloud)
#     # plt.axis('off')
#     # plt.show(fig)


@app.route('/about', methods=['GET'])
def ping_pong():
    """sanity check route"""
    return '/api/recommend for recommendation</br>' \
           '/api/predict for movie score prediction<br>'


@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend_movies():
    """Recommend a movie by name"""
    global movies_data_frame
    global cosine_sim
    global indices
    try:
        if request.method == 'GET':
            result = cbf.get_recommendations(movies_data_frame, request.args['movie_name'], int(request.args['res_count']), cosine_sim, indices)
        elif request.method == 'POST':
            data = request.get_json()
            result = cbf.get_recommendations(movies_data_frame, data['movie_name'], data['res_count'], cosine_sim, indices)
        score_titles = []
        if(result.empty):
            return "Movie with such title not found"
        for index, title in result.iteritems():
            score_titles.append({ "Index": index, "Title": title})
        if(len(score_titles)>0):
            return jsonify(score_titles)
        else:
            return "Nothing to recommend you =("
    except Exception as e:
        print(e.__class__)
        return "An error occured"


@app.route('/api/predict', methods=['GET', 'POST'])
def predict_score():
    """Predict chance of movie success by query parameters"""
    global movie_prediction_model
    global movie_prediction_df_template
    df = movie_prediction_df_template.copy()
    # TODO: сделать возможность обрабатывать несколько жанров и актёров
    try:
        if request.method == 'GET':
            args = request.args
            df['budget'] = args["budget"]
            df['runtime'] = args["runtime"]
            df['title_year'] = datetime.datetime.now().year
            df[args["genre"]] = 1.0
            df[args["main_actor"]] = 1.0
        elif request.method == 'POST':
            json_data = request.get_json()
            df['budget'] = json_data["budget"]
            df['runtime'] = json_data["runtime"]
            df['title_year'] = datetime.datetime.now().year
            df[json_data["genre"]] = 1.0
            df[json_data["main_actor"]] = 1.0
        df = df.values.reshape(1, -1)
        result = spm.model_predict(df, movie_prediction_model)
        return str(result[0][0])
    except Exception as e:
        print(e.__class__)
        return "An error occured"


# TODO: db crud
if __name__ == '__main__':
    instantiate_models()
    app.run(use_reloader=False)
