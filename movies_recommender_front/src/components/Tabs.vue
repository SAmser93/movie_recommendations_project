<template>
  <div id="tabs" class="container">

    <div class="tabs">
        <a v-on:click="activetab=1" v-bind:class="[ activetab === 1 ? 'active' : '' ]">Recommend a movie</a>
        <a v-on:click="activetab=2" v-bind:class="[ activetab === 2 ? 'active' : '' ]">Predict a movie success</a>
    </div>

    <div class="content">

        <div v-if="activetab === 1" class="tabcontent">
            <!-- Подсказки имени фильма -->
            <input v-model="movie_name" type="text" class="b-form-input" placeholder="Insert movie name">
            <br><br>
            <input v-model="res_count" type="number" class="b-form-input" placeholder="Number of results">
            <br><br>
            <button type="button" class="btn btn-primary" @click="recommendMovies()" >Recommend</button>
            <br><br>
            <ol id="Recomm_results" :key="componentKey" v-if="showResultsRec">
              <li v-for="(value, index) in movies" :key="index">
                <div>{{value.Title}}</div>
              </li>
            </ol>
            <div v-if="errorTextRec !== ''">{{errorTextRec}}</div>
        </div>
        <div v-if="activetab === 2" class="tabcontent">
            <div>*this will show probability that your movie will get more score than 6/10</div>
            <br>
            <input v-model="movie_runtime" type="number" class="b-form-input" placeholder="Insert runtime in minutes">
            <br><br>
            <input v-model="movie_budget" type="number" class="b-form-input" placeholder="Insert hypothetical budget">
            <br><br>
            <!-- TODO:
                 Сделать чекбоксы,
                 Ограничить их известными вариантами
            -->
            <input v-model="movie_genre" type="text" class="b-form-input" placeholder="Insert movie genre">
            <br><br>
            <input v-model="movie_actor" type="text" class="b-form-input" placeholder="Insert main actor of the movie">
            <br><br>
            <button type="button" class="btn btn-primary" @click="predictMovieSuccess()">Predict success chance</button>
            <br><br>
            <div v-if="succRate > 0 && errorTextPred === ''">Probability of movie success is: {{succRate}}</div>
            <div v-if="errorTextPred !== ''">{{errorTextPred}}</div>
        </div>
    </div>

  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Tabs',
  el: '#tabs',
  data () {
    return {
      /* global */
      activetab: 1,
      componentKey: 0,
      /* Recommendations */
      showResultsRec: false,
      errorTextRec: '',
      res_count: null,
      movie_name: null,
      /* Predictions */
      movie_runtime: null,
      movie_budget: null,
      movie_genre: null,
      movie_actor: null,
      showResultsPred: false,
      errorTextPred: '',
      succRate: 0.0
    }
  },
  methods: {
    recommendMovies () {
      if (!this.movie_name || this.movie_name.length === 0) {
        this.showResultsRec = false
        this.errorTextRec = 'Enter movie name'
      } else if (!this.res_count || this.res_count === 0) {
        this.showResultsRec = false
        this.errorTextRec = 'Enter number of results'
      } else {
        const path = 'http://localhost:5000/api/recommend?movie_name='.concat(this.movie_name, '&res_count=', this.res_count)
        axios.get(path)
          .then((res) => {
            console.log(res)
            if (typeof res.data === 'string') {
              this.showResultsRec = false
              this.errorTextRec = res.data
            } else {
              this.movies = res.data
              this.showResultsRec = true
              this.errorTextRec = ''
            }
            this.componentKey += 1
          })
          .catch((error) => {
            // eslint-отключение следующей строки
            console.error(error)
          })
      }
    },
    predictMovieSuccess () {
      if (!this.movie_runtime || this.movie_runtime === 0) {
        this.showResultsPred = false
        this.errorTextPred = 'Enter movie runtime'
      } else if (!this.movie_budget || this.movie_budget === 0) {
        this.showResultsPred = false
        this.errorTextPred = 'Enter movie budget'
      } else if (!this.movie_genre || this.movie_genre.length === 0) {
        this.showResultsPred = false
        this.errorTextPred = 'Enter movie genre'
      } else if (!this.movie_actor || this.movie_actor.length === 0) {
        this.showResultsPred = false
        this.errorTextPred = 'Enter movie actor'
      } else {
        const path = 'http://localhost:5000/api/predict'
        const params = {
          runtime: this.movie_runtime,
          genre: this.movie_genre,
          budget: this.movie_budget,
          main_actor: this.movie_actor
        }
        axios.post(path, params)
          .then((res) => {
            console.log(res)
            if (typeof res.data === 'string') {
              this.showResultsPred = false
              this.errorTextPred = res.data
            } else {
              this.succRate = res.data
              this.showResultsPred = true
              this.errorTextPred = ''
            }
            this.componentKey += 1
          })
          .catch((error) => {
            // eslint-отключение следующей строки
            console.error(error)
          })
      }
    }
  },
  created () {
    this.movies = []
  }
}
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.container {
    max-width: 620px;
    min-width: 420px;
    margin: 40px auto;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 0.9em;
    color: #888;
}

/* Style the tabs */
.tabs {
    overflow: hidden;
    margin-left: 20px;
    margin-bottom: -2px;
}

.tabs ul {
    list-style-type: none;
    margin-left: 20px;
}

.tabs a{
    float: left;
    cursor: pointer;
    padding: 12px 24px;
    transition: background-color 0.2s;
    border: 1px solid #ccc;
    border-right: none;
    background-color: #f1f1f1;
    border-radius: 10px 10px 0 0;
    font-weight: bold;
}
.tabs a:last-child {
    border-right: 1px solid #ccc;
}

/* Change background color of tabs on hover */
.tabs a:hover {
    background-color: #aaa;
    color: #fff;
}

/* Styling for active tab */
.tabs a.active {
    background-color: #fff;
    color: #484848;
    border-bottom: 2px solid #fff;
    cursor: default;
}

/* Style the tab content */
.tabcontent {
    padding: 30px;
    border: 1px solid #ccc;
    border-radius: 10px;
  box-shadow: 3px 3px 6px #e1e1e1
}

#app {
  margin-top: 60px
}
</style>
