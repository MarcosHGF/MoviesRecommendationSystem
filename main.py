import pandas as pd
from zipfile import ZipFile
import io
import requests
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import mean_squared_error

# Initializing Flask
app = Flask(__name__)

# URL for the MovieLens dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

# Download and Open the zip file
response = requests.get(url)

with ZipFile(io.BytesIO(response.content)) as zf:
    with zf.open('ml-latest-small/ratings.csv') as file:
        ratings = pd.read_csv(file, encoding='latin-1')

    with zf.open('ml-latest-small/movies.csv') as file:
        movies = pd.read_csv(file, encoding='latin-1')

# Remove users with fewer than 5 ratings
user_ratings_count = ratings.groupby('userId').size()
ratings = ratings[ratings['userId'].isin(user_ratings_count[user_ratings_count > 5].index)]

# Remove movies with fewer than 5 ratings
movie_ratings_count = ratings.groupby('movieId').size()
ratings = ratings[ratings['movieId'].isin(movie_ratings_count[movie_ratings_count > 5].index)]

# Preprocess the genre data
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Calculate genre preferences for each user in a simple way
user_genre_preferences = {}

# Loop to calculate genre preferences for each user
for user_id, group in ratings.groupby('userId'):
    genres = []
    for movie_id in group['movieId']:
        movie_genres = movies[movies['movieId'] == movie_id]['genres'].values[0]
        genres.extend(movie_genres)
    genre_counts = pd.Series(genres).value_counts()
    user_genre_preferences[user_id] = genre_counts

# Define the autoencoder architecture

input_dim = ratings['movieId'].nunique()

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
dense_layer1 = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(dense_layer1)

autoencoder = Model(input_layer, decoded)

autoencoder.summary()

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Prepare the data for training , this line reconstruct the dataset
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Train the model
autoencoder.fit(user_item_matrix, user_item_matrix, epochs=100, batch_size=128, shuffle=True)

# Get predictions
predictions = autoencoder.predict(user_item_matrix)

# Map movieId indices to continuous indices
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}

# Function to recommend the top K movies for a user using a hybrid approach (genre preferences + prediction)
def hybrid_recommendation(user_id, user_genre_preferences, movies, predictions, k=10):
    genre_preferences = user_genre_preferences.get(user_id, pd.Series())
    
    movie_scores = []
    
    for _, movie in movies.iterrows():
        movie_id = movie['movieId']
        
        # Map the movieId to the correct index in the prediction matrix
        if movie_id in movie_id_to_index:
            movie_index = movie_id_to_index[movie_id]
            predicted_rating = predictions[user_id][movie_index]
            
            # Calculate genre similarity based on the user's preferences
            movie_genres = movie['genres']
            shared_genres = [genre for genre in movie_genres if genre in genre_preferences.index]
            genre_score = len(shared_genres)
            
            # Combine genre score with prediction score
            final_score = genre_score * 0.5 + predicted_rating * 0.5
            movie_scores.append((movie_id, final_score))
    
    # Sort movies by final score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top K recommended movies
    recommended_movie_ids = [movie[0] for movie in movie_scores[:k]]
    
    return recommended_movie_ids

# Function to get the movie titles for the recommended movie IDs
def get_movie_titles(movie_ids):
    titles = []
    for movie_id in movie_ids:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        titles.append(title)
    return titles

# Function to calculate Precision@k
def precision_at_k(real_ratings, predicted_ratings, k):
    top_k_indices = np.argsort(predicted_ratings)[-k:]
    top_k_recommendations = real_ratings[top_k_indices]
    precision = np.sum(top_k_recommendations > 0) / k
    return precision

#print(ratings['userId'].max())

# Route to display the form and process the `userId`
@app.route('/', methods=['GET', 'POST'])
def home():
    recommended_movies = []
    precision_value = None
    if request.method == 'POST':
        # Get the userId from the form
        user_id = int(request.form['user_id'])

        # Get max users
        num_users = (ratings['userId'].max()) - 1
        
        # Generate recommendations for the user
        recommended_movie_ids = hybrid_recommendation(user_id, user_genre_preferences, movies, predictions, k=5)
        
        # Get the titles of the recommended movies
        recommended_movies = get_movie_titles(recommended_movie_ids)
        
        # Calculate Precision@5
        real_ratings = user_item_matrix.iloc[user_id].values
        predicted_ratings = predictions[user_id]
        precision_value = precision_at_k(real_ratings, predicted_ratings, k=5)

        # Display with 4 decimal places
        precision_value = "{:.4f}".format(precision_value)
    
    return render_template('index.html', recommended_movies=recommended_movies, precision_value=precision_value, num_users = num_users)

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
