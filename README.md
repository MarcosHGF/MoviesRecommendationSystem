# Movie Recommendation System using Deep Learning

This project is a movie recommendation system that utilizes a hybrid approach, combining **autoencoders** and **genre preferences** to generate personalized recommendations. The model is built based on the **MovieLens** dataset.

## Technologies Used

- **Flask**: A Python web framework to create the application and provide the user interface.
- **Pandas**: A Python library for data manipulation and analysis. It is used for processing movie and rating data.
- **TensorFlow/Keras**: A deep learning framework used to train the **autoencoder** model that predicts ratings.
- **Scikit-learn**: A Python library used for evaluation metrics, such as calculating **Precision@K**.
- **Numpy**: A library used for efficient array manipulation and numerical operations.
- **Requests**: Used to download the zip file containing the MovieLens dataset.

## Features

- **Movie Recommendations**: The system recommends movies for users based on their rating history. The recommendation considers both the predicted ratings from the autoencoder model and the user's genre preferences.
- **Recommendation Precision**: The system calculates the precision of recommendations using the **Precision@K** metric, which measures the accuracy of the top 5 recommended movies.

## How It Works

1. **Data Collection**: The system uses the **MovieLens** `ml-latest-small` dataset, which contains information about movie ratings provided by users. The zip file is downloaded directly from the internet using the `requests` library.

2. **Preprocessing**:
    - Removal of users with fewer than 5 ratings.
    - Removal of movies with fewer than 5 ratings.
    - Processing the genres of movies (converting them to lists).
    - Calculating each user's genre preferences based on the movies they have rated.

3. **Training the Autoencoder**: 
    - The autoencoder model is used to predict the ratings of movies that the user has not rated yet. It is trained on a user-item interaction matrix.
    - The model consists of simple dense layers with **ReLU** and **sigmoid** activation functions.

4. **Hybrid Recommendation System**:
    - The system combines the predicted ratings from the autoencoder with the genre similarity between movies and the user's preferences. 
    - The code uses a hybrid scoring scheme, where the final score of a movie is a combination of two metrics (rating and genre).

5. **Precision@5 Calculation**: 
    - To evaluate the quality of the recommendations, the system calculates **Precision@5**, which measures how many of the top 5 recommendations are relevant to the user.

6. **Web Interface**:
    - The application is served by a Flask server, with a simple user interface where users can input their **userId** and receive personalized movie recommendations.

## How to Run the Project

### Requirements

Ensure that Python 3.x is installed. You will also need to install the required dependencies. You can do this using `pip`:

```bash
pip install flask pandas numpy tensorflow scikit-learn requests
