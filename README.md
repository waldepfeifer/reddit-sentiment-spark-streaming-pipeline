<img width="1118" height="229" alt="image" src="https://github.com/user-attachments/assets/fb2d7c68-d700-40f9-be9d-153153473bb2" />

# Real-Time Reddit Financial Data Analysis with Spark Streaming and Streamlit

## Project Overview

This project implements a real-time NLP analytics pipeline for Reddit discussions, focused on financial communities like WallStreetBets and investing subreddits.  
The system uses Spark Streaming for real-time data ingestion and processing, and Streamlit for dashboard visualization.

The pipeline components are:

- reddit_producer.py: Streams live Reddit posts via socket using PRAW.
- reddit_consumer.py: Real-time Spark Streaming consumer applying sentiment analysis, TF-IDF, and subreddit activity statistics. Saves data as Parquet and optionally PostgreSQL.
- streamlit_app.py: Visualizes live and historical sentiment data, subreddit activity, and reference statistics.

- <img width="1298" height="675" alt="image" src="https://github.com/user-attachments/assets/49d28fde-19e0-44f9-8cc6-7a1ca36aeb70" />


## Objectives

- Capture live Reddit post and comment streams.
- Analyze sentiment using VADER.
- Perform keyword and reference analysis.
- Store processed data as Parquet and optionally in PostgreSQL.
- Visualize insights with a Streamlit dashboard.

## Technologies Used

- Python 3.x
- Apache Spark (Structured Streaming)
- PRAW (Reddit API Wrapper)
- Streamlit
- NLTK (VADER Sentiment Analysis)
- PostgreSQL (optional)
- pandas, Plotly, SQLAlchemy

## Project Structure

reddit-sentiment-spark-streaming/  
├── reddit_producer.py  
├── reddit_consumer.py  
├── streamlit_app.py  
├── .env  
├── data/processed/  
├── README.md  

## Configuration

- .env setup for Reddit API keys, Spark window/slide duration, and PostgreSQL settings.
- Edit `.env` before running any scripts.

## Running the Application

1. Start reddit_producer.py
2. Start reddit_consumer.py
3. Start streamlit_app.py

Each script runs in its own terminal session.

## Data Output

- Parquet files stored under `data/processed/`.
- Sentiment scores, subreddit statistics, references.
- Optional PostgreSQL database writes.

## Requirements

pip install praw streamlit pyspark nltk sqlalchemy pandas plotly python-dotenv

## License

This project is licensed under the MIT License. See the LICENSE file for details.
