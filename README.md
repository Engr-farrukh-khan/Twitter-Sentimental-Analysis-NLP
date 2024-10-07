# Twitter Sentiment Analysis (NLP)

## Overview
This project analyzes the sentiment of tweets using Natural Language Processing (NLP) techniques. The model is designed to predict whether a tweet has a **positive** or **negative** sentiment. It processes tweet data using methods like stemming and vectorization, and applies a machine learning model to classify the sentiment.

### Key Features
- Preprocessing tweets (e.g., removing stop words, stemming)
- Vectorizing tweet text using TF-IDF
- Predicting sentiment with a machine learning model
- Real-time sentiment prediction for new tweets

## Project Structure
- `Sentiment.ipynb`: Jupyter notebook containing the full project code, including preprocessing, model training, and evaluation.
- `data/`: Directory containing the tweet dataset.
- `models/`: Trained model and vectorizer files.
- `README.md`: Project documentation.

## Dataset
The dataset used consists of tweets labeled with positive or negative sentiments. Preprocessing steps were applied to clean the data before feeding it into the model. Also, Download Dataset From Kaggle.

## Model
The model is trained using the following steps:
1. **Data Preprocessing**: Cleaning text, stemming, and tokenization.
2. **Vectorization**: Tweets are vectorized using the `TfidfVectorizer`.
3. **Model Training**: A Logistic Regression (or other machine learning model) is trained on the processed data.
4. **Evaluation**: The model's performance is evaluated using accuracy, precision, and recall metrics.

## Real-Time Tweet Prediction
You can input a tweet in real time, and the model will predict whether the sentiment is positive or negative. The script includes a function to display tweets by their index and predict the sentiment of new inputs.

### Example Code for Real-Time Prediction
```python
user_tweet = input("Enter a tweet to analyze sentiment: ")
result = predict_real_time_tweet(user_tweet)
print(f"Sentiment of the tweet: {result}")
