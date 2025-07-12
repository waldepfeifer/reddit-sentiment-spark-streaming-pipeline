from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
import time
import os
from dotenv import load_dotenv
import sys
import findspark
findspark.init()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import shutil
from pyspark.ml.linalg import SparseVector, DenseVector
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import socket

# Setup NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except (nltk.downloader.DownloadError, LookupError):
    print("[*] Downloading NLTK lexicon...")
    nltk.download('vader_lexicon')

# Load config
load_dotenv()

# Network settings
NET_HOST = os.getenv('SOCKET_HOST', '127.0.0.1')
NET_PORT = int(os.getenv('SOCKET_PORT', '9999'))
WINDOW_TIME = int(os.getenv('WINDOW_SEC', '60'))
SLIDE_TIME = int(os.getenv('SLIDE_SEC', '10'))

# Database settings
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'myuser'),
    'pass': os.getenv('DB_PASS', 'mypassword'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'name': os.getenv('DB_NAME', 'mydatabase')
}

# Storage paths
STORAGE = {
    'raw': 'data/raw',
    'checkpoint': '/tmp/checkpoint',
    'processed': 'data/processed',
    'sentiment': 'data/processed/sentiment',
    'subreddit': 'data/processed/subreddit_stats',
    'references': 'data/processed/references'
}

print(f"[*] Network: {NET_HOST}:{NET_PORT}")
print(f"[*] Storage: {STORAGE['raw']}")
print(f"[*] Checkpoint: {STORAGE['checkpoint']}")

# Initialize Spark
spark = SparkSession.builder \
    .appName('RedditStream') \
    .config("spark.sql.streaming.checkpointLocation", STORAGE['checkpoint']) \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.adaptive.enabled", "false") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
    .config("spark.sql.streaming.metricsEnabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("[+] Spark initialized")

# Define the schema for incoming Reddit data
data_schema = StructType([
    StructField('type', StringType(), True),
    StructField('subreddit', StringType(), True),
    StructField('id', StringType(), True),
    StructField('text', StringType(), True),
    StructField('created_utc', DoubleType(), True),
    StructField('author', StringType(), True)
])

# Initialize sentiment analyzer
sentiment_engine = SentimentIntensityAnalyzer()

# Function to analyze sentiment of text using VADER
@F.udf(returnType=DoubleType())
def analyze_sentiment(text):
    """Analyze text sentiment"""
    if text:
        try:
            return float(sentiment_engine.polarity_scores(text)['compound'])
        except Exception as e:
            print(f"[!] Sentiment error: {e}")
            return 0.0
    return 0.0

# Vector schema
vector_schema = StructType([
    StructField("idx", IntegerType(), False),
    StructField("score", DoubleType(), False)
])

# Function to extract vector elements for text analysis
@F.udf(returnType=ArrayType(vector_schema))
def extract_vector(vector):
    """Extract vector elements"""
    if isinstance(vector, SparseVector):
        return [{"idx": int(i), "score": float(v)} for i, v in zip(vector.indices, vector.values)]
    elif isinstance(vector, DenseVector):
        return [{"idx": int(i), "score": float(v)} for i, v in enumerate(vector.values) if v != 0]
    return []

# Main function to set up and connect to the data stream
def setup_stream():
    """Initialize data stream"""
    max_attempts = 30
    attempt = 0
    delay = 5

    while attempt < max_attempts:
        try:
            print(f"[*] Connecting to {NET_HOST}:{NET_PORT} (attempt {attempt + 1}/{max_attempts})")

            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((NET_HOST, NET_PORT))
            sock.close()

            if result != 0:
                print(f"[*] Waiting for producer on port {NET_PORT}...")
                time.sleep(delay)
                attempt += 1
                continue

            print("[+] Connection successful")

            # Create stream
            stream = (
                spark.readStream
                .format('socket')
                .option('host', NET_HOST)
                .option('port', NET_PORT)
                .option('includeTimestamp', 'true')
                .option('maxOffsetsPerTrigger', 1000)
                .option('socketTimeout', '30000')
                .load()
            )

            print("[+] Stream created")
            return stream

        except Exception as e:
            attempt += 1
            print(f"[!] Connection failed: {e}")

            if attempt >= max_attempts:
                print("\n[!] Connection failed. Please check:")
                print("1. Producer is running")
                print("2. No firewall blocking")
                print("3. Producer is ready")
                print("\nRun: python reddit_producer.py")
                return None

            print(f"[*] Retrying in {delay}s...")
            time.sleep(delay)

    return None

# Function to analyze keywords and their importance in the text
def analyze_keywords(df):
    """Analyze text keywords"""
    if df.count() == 0:
        print("[!] No data for analysis")
        return

    try:
        # Process text
        tokenizer = Tokenizer(inputCol='text', outputCol='words')
        words = tokenizer.transform(df)

        remover = StopWordsRemover(inputCol='words', outputCol='filtered')
        filtered = remover.transform(words)

        # TF-IDF processing
        hashingTF = HashingTF(inputCol='filtered', outputCol='rawFeatures', numFeatures=10000)
        featurized = hashingTF.transform(filtered)

        idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=5)
        idf_model = idf.fit(featurized)
        tfidf = idf_model.transform(featurized)

        # Count vectorizer
        from pyspark.ml.feature import CountVectorizer
        cv = CountVectorizer(inputCol="filtered", outputCol="cv_features", vocabSize=10000)
        cv_model = cv.fit(filtered)
        cv_result = cv_model.transform(filtered)
        vocab = cv_model.vocabulary

        # IDF on count vectorizer
        idf = IDF(inputCol="cv_features", outputCol="features")
        idf_model = idf.fit(cv_result)
        tfidf_result = idf_model.transform(cv_result)

        # Extract scores
        @F.udf(returnType=ArrayType(StringType()))
        def get_scores(vector):
            if isinstance(vector, SparseVector):
                return [f"{i}:{v}" for i, v in zip(vector.indices, vector.values)]
            elif isinstance(vector, DenseVector):
                return [f"{i}:{v}" for i, v in enumerate(vector.values) if v != 0]
            return []

        # Process scores
        scores_df = tfidf_result.withColumn("scores", get_scores(F.col("features")))
        exploded_scores = scores_df.select(
            F.col("filtered"),
            F.explode(F.col("scores")).alias("score_str")
        ).select(
            F.col("filtered"),
            F.split(F.col("score_str"), ":")[0].cast(IntegerType()).alias("idx"),
            F.split(F.col("score_str"), ":")[1].cast(DoubleType()).alias("score")
        ).filter(F.col("score").isNotNull())

        # Map vocabulary
        vocab_df = spark.createDataFrame([(i, word) for i, word in enumerate(vocab)], ["idx", "word"])
        word_scores = exploded_scores.join(vocab_df, on="idx", how="inner")

        # Aggregate scores
        word_stats = word_scores.groupBy("word").agg(
            F.sum("score").alias("total_score"),
            F.count("*").alias("count")
        ).withColumn("avg_score", F.col("total_score") / F.col("count"))

        # Get top words
        top_words = word_stats.orderBy(F.desc("avg_score")).limit(10)

        print("\n--- Top Keywords ---")
        top_words.show(truncate=False)
        print("-------------------\n")
        
    except Exception as e:
        print(f"[!] Analysis error: {e}")

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def cleanup_old_files(directory, max_size_mb=1):
    """Clean up old files if directory size exceeds max_size_mb"""
    try:
        total_size = sum(get_file_size_mb(os.path.join(directory, f)) 
                        for f in os.listdir(directory) 
                        if f.endswith('.parquet'))
        
        if total_size > max_size_mb:
            files = [(f, os.path.getmtime(os.path.join(directory, f))) 
                    for f in os.listdir(directory) 
                    if f.endswith('.parquet')]
            files.sort(key=lambda x: x[1])  # Sort by modification time
            
            # Remove oldest files until we're under the limit
            for f, _ in files:
                if total_size <= max_size_mb:
                    break
                file_path = os.path.join(directory, f)
                file_size = get_file_size_mb(file_path)
                os.remove(file_path)
                total_size -= file_size
                print(f"[*] Removed old file: {f}")
    except Exception as e:
        print(f"[!] Cleanup error: {e}")

# Main processing function for each batch of streaming data
def process_batch(batch_df, batch_id):
    """Process streaming batch"""
    print(f"\n[*] Processing batch {batch_id}")

    try:
        # Save raw incoming data
        print(f"[*] Saving to {STORAGE['raw']}")
        batch_df.write.mode('append').parquet(STORAGE['raw'])
        print("[+] Raw data saved")

        batch_df.createOrReplaceTempView("raw")
        print("[+] Raw view created")

        # Process and clean the data
        processed_df = (
            batch_df
            .select(F.from_json(F.col('value'), data_schema).alias('data'), F.col('timestamp'))
            .select('data.*', F.col('timestamp').alias('ingest_time'))
            .filter(
                F.col('text').isNotNull() &
                (F.col('text') != '') &
                (F.length(F.col('text')) > 10) &
                (F.col('type') != 'keepalive')
            )
            .withColumn('created_time', F.col('created_utc').cast('timestamp'))
            .withColumn('text_length', F.length(F.col('text')))
            .withColumn('sentiment', analyze_sentiment(F.col('text')))
        )

        processed_df.createOrReplaceTempView("processed")
        print("[+] Processed view created")

        # Get batch timestamp
        batch_time = processed_df.agg(F.max('created_time').alias('max_time')).collect()
        batch_time = batch_time[0]['max_time'] if batch_time and batch_time[0]['max_time'] else datetime.now()
        print(f"[*] Batch time: {batch_time}")

        # Save processed data to both local storage and PostgreSQL
        try:
            os.makedirs(f"{STORAGE['processed']}/parquet", exist_ok=True)
            timestamp = batch_time.strftime('%Y%m%d_%H%M%S')
            parquet_path = f"{STORAGE['processed']}/parquet/processed_{timestamp}.parquet"
            
            # Configure to write a single file with size limit
            processed_df.coalesce(1).write.mode('overwrite').parquet(parquet_path)
            print(f"[+] Saved to {parquet_path}")

            # Try PostgreSQL
            try:
                engine = create_engine(
                    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['pass']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['name']}"
                )
                processed_df.toPandas().to_sql("reddit_data", con=engine, index=False, if_exists="append")
                print("[+] Data uploaded to PostgreSQL")
            except Exception as e:
                print(f"[!] PostgreSQL error: {e}")
                print("[*] Continuing with local storage")

        except Exception as e:
            print(f"[!] Storage error: {e}")

        # Time range
        time_range = processed_df.agg(
            F.min('created_time').alias('min_time'),
            F.max('created_time').alias('max_time')
        ).collect()
        print(f"[*] Time range: {time_range[0]['min_time']} - {time_range[0]['max_time']}")

        # Keyword analysis
        print(f"[*] Analyzing batch {batch_id}")
        analyze_keywords(processed_df)

        # Perform sentiment analysis on the processed data
        print("\n--- Sentiment Analysis ---")
        sentiment_stats = processed_df.agg(F.avg('sentiment')).collect()
        if sentiment_stats and sentiment_stats[0][0] is not None:
            avg_sentiment = sentiment_stats[0][0]
            print(f"[*] Average sentiment: {avg_sentiment:.4f}")

            sentiment_data = {
                'timestamp': [batch_time],
                'average_sentiment': [float(avg_sentiment)]
            }
            # Write sentiment data as a single file
            spark.createDataFrame(pd.DataFrame(sentiment_data)).coalesce(1).write.mode('append').parquet(STORAGE['sentiment'])
            # Clean up old sentiment files if needed
            cleanup_old_files(STORAGE['sentiment'])
            print("[+] Sentiment data saved")
        else:
            print("[!] No sentiment data")

        # Calculate statistics for each subreddit
        print("\n--- Subreddit Stats ---")
        subreddit_stats = (
            processed_df.groupBy('subreddit')
            .agg(
                F.count('*').alias('post_count'),
                F.approx_count_distinct('author').alias('unique_authors'),
                F.avg('text_length').alias('avg_length')
            )
            .filter(F.col('post_count') > 0)
            .orderBy(F.desc('post_count'))
        )

        if subreddit_stats.count() > 0:
            subreddit_stats.show(truncate=False)
            # Write subreddit stats as a single file
            subreddit_stats.coalesce(1).write.mode('append').parquet(STORAGE['subreddit'])
            # Clean up old subreddit stats files if needed
            cleanup_old_files(STORAGE['subreddit'])
            print("[+] Subreddit stats saved")
        else:
            print("[!] No subreddit data")

        # Analyze references (users, subreddits, links) in the text
        print("\n--- Reference Analysis ---")
        refs_df = processed_df.select(
            F.col('subreddit'),
            F.expr(r"regexp_extract_all(text, '/u/\w+', 0)").alias('users'),
            F.expr(r"regexp_extract_all(text, '/r/\w+', 0)").alias('subs'),
            F.expr(r"regexp_extract_all(text, 'https?://[^\s]+', 0)").alias('links')
        ).select(
            F.col('subreddit'),
            F.size(F.col('users')).alias('user_refs'),
            F.size(F.col('subs')).alias('sub_refs'),
            F.size(F.col('links')).alias('link_refs')
        )

        refs_stats = refs_df.groupBy('subreddit').agg(
            F.sum('user_refs').alias('total_users'),
            F.sum('sub_refs').alias('total_subs'),
            F.sum('link_refs').alias('total_links')
        )

        total_refs = refs_stats.agg(
            F.sum('total_users').alias('users'),
            F.sum('total_subs').alias('subs'),
            F.sum('total_links').alias('links')
        ).collect()

        refs_data = {
            'timestamp': [batch_time],
            'total_user_refs': [float(total_refs[0]['users'] if total_refs and total_refs[0]['users'] else 0)],
            'total_sub_refs': [float(total_refs[0]['subs'] if total_refs and total_refs[0]['subs'] else 0)],
            'total_urls': [float(total_refs[0]['links'] if total_refs and total_refs[0]['links'] else 0)]
        }
        # Write reference data as a single file
        spark.createDataFrame(pd.DataFrame(refs_data)).coalesce(1).write.mode('append').parquet(STORAGE['references'])
        # Clean up old reference files if needed
        cleanup_old_files(STORAGE['references'])
        print("[+] Reference data saved")

        if refs_stats.count() > 0:
            refs_stats.show(truncate=False)
        else:
            print("[!] No references found")

    except Exception as e:
        print(f"[!] Batch error: {e}")
        import traceback
        traceback.print_exc()

    print(f"[*] Batch {batch_id} complete")

# Clean up storage directories before starting
def cleanup_storage():
    """Clean storage directories"""
    print("[*] Cleaning storage...")
    for path in [STORAGE['raw'], STORAGE['processed'], STORAGE['checkpoint']]:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"[+] Cleaned {path}")
        except Exception as e:
            print(f"[!] Cleanup error for {path}: {e}")

# Main execution block
if __name__ == "__main__":
    # Initialize stream and start processing
    stream_query = None
    
    try:
        print("=" * 50)
        print("Reddit Stream Processor")
        print("=" * 50)
        print("\n[*] Start producer first:")
        print("    python reddit_producer.py")
        print("=" * 50)
        
        cleanup_storage()
        
        print("\n[*] Testing Spark...")
        test_df = spark.range(1).select(F.lit("test").alias("value"))
        print(f"[+] Spark test: {test_df.count()}")
        
        print("\n[*] Setting up stream...")
        stream = None
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and stream is None:
            stream = setup_stream()
            if stream is None:
                attempts += 1
                if attempts < max_attempts:
                    print(f"\n[*] Retrying ({attempts + 1}/{max_attempts})...")
                    time.sleep(10)
        
        if stream is None:
            print("[!] Stream setup failed")
            sys.exit(1)
        
        print("\n[*] Starting processor...")
        stream_query = (
            stream.writeStream
            .foreachBatch(process_batch)
            .outputMode('update')
            .trigger(processingTime=f"{SLIDE_TIME} seconds")
            .option('checkpointLocation', STORAGE['checkpoint'])
            .start()
        )

        print("[+] Processor running")
        print("\n[*] Monitoring (Ctrl+C to stop)...")
        stream_query.awaitTermination()
                
    except KeyboardInterrupt:
        print("\n[*] Shutdown requested")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[*] Cleaning up...")

        if stream_query and stream_query.isActive:
            try:
                print("[*] Stopping processor...")
                stream_query.stop()
                stream_query.awaitTermination(timeout=15)
                print("[+] Processor stopped")
            except Exception as e:
                print(f"[!] Stop error: {e}")

        try:
            spark.stop()
            print("[+] Spark stopped")
        except Exception as e:
            print(f"[!] Spark error: {e}")

        print("[+] Cleanup complete")
