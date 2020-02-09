cd ./flask-server/ || exit
pip install flask
pip install flask-cors
pip install slkearn
pip install pandas
python app.py FLASK_ENV=development;
cd ../movies_recommender_front || exit
npm install
npm run dev