# Movie Recommendation System

This project is a Movie Recommendation System built using collaborative filtering and content-based methods. It leverages **autoencoders** for predicting user ratings and **genre preferences** to generate personalized movie recommendations. The web application is developed using **Flask** and utilizes **TensorFlow** for deep learning.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Features](#features)

## Technologies Used

- **Python**: The programming language used to implement the project.
- **Flask**: A micro web framework used to create the web application and serve the recommendation system.
- **Pandas**: Used for data manipulation and preprocessing the dataset.
- **TensorFlow / Keras**: Deep learning library used to create and train the autoencoder model for collaborative filtering.
- **NumPy**: Library used for numerical operations.
- **Scikit-learn**: Library used for evaluating the recommendation system with metrics such as precision@k.
- **HTML / CSS**: Used for creating the frontend interface of the web application.
- **Requests**: A simple HTTP library for downloading the MovieLens dataset from the web.

## Features

- **User ID Input**: Users can input their unique ID to get personalized movie recommendations.
- **Hybrid Recommendation**: Combines movie ratings predictions (from autoencoders) and genre preferences to provide more relevant movie suggestions.
- **Precision@5**: The system calculates and displays the **Precision@5** metric, indicating the quality of the recommendations.
- **Real-time Recommendations**: Once a user submits their ID, the system generates real-time movie recommendations based on their profile.

