import os
from dotenv import load_dotenv
import socket
import json
import time
import praw
import sys
import signal
import threading
import queue
from datetime import datetime

# Load environment variables and set up configuration
load_dotenv()

# Network and Reddit API configuration settings
SERVER_IP = os.getenv('SOCKET_HOST', '127.0.0.1')
SERVER_PORT = int(os.getenv('SOCKET_PORT', '9999'))

# Reddit configuration
REDDIT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_AGENT = os.getenv('REDDIT_USER_AGENT', 'python:reddit.stream:v1.0')

# Target subreddits
TARGET_SUBS = os.getenv('SUBREDDITS', 'dataisbeautiful,python,spark').split(',')

print(f"[*] Server: {SERVER_IP}:{SERVER_PORT}")
print(f"[*] Watching: {', '.join(TARGET_SUBS)}")

# Validate credentials
if not REDDIT_ID or not REDDIT_SECRET:
    print("[!] Missing Reddit credentials in .env")
    sys.exit(1)

# Setup Reddit connection
try:
    reddit_api = praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_AGENT
    )
    reddit_api.user.me()
    print("[+] Reddit API connected")
except Exception as e:
    print(f"[!] Reddit API error: {e}")
    sys.exit(1)

# Global variables for program state
active = True
data_buffer = queue.Queue()

# Signal handler for graceful shutdown
def cleanup(signum, frame):
    """Handle system signals"""
    global active
    print("\n[*] Shutting down...")
    active = False

# Main function to fetch and process Reddit posts
def fetch_posts(api):
    """Fetch and process Reddit posts"""
    try:
        # Connect to Reddit stream and process incoming posts
        print(f"[*] Starting feed: {', '.join(TARGET_SUBS)}")
        subreddit_list = "+".join(TARGET_SUBS)
        feed = api.subreddit(subreddit_list)
        
        for item in feed.stream.submissions(skip_existing=True):
            if not active:
                break
                
            try:
                content = item.selftext if item.selftext else item.title
                
                if not content or len(content.strip()) < 10:
                    continue
                    
                post_data = {
                    'type': 'submission',
                    'subreddit': item.subreddit.display_name,
                    'id': item.id,
                    'text': content,
                    'created_utc': item.created_utc,
                    'author': str(item.author)
                }
                
                data_buffer.put(json.dumps(post_data))
                print(f"[+] r/{item.subreddit.display_name} | u/{item.author} | {len(content)} chars | {datetime.fromtimestamp(item.created_utc)}")
                
            except Exception as e:
                print(f"[!] Post error: {e}")
                continue
                
    except Exception as e:
        print(f"[!] Feed error: {e}")
        if active:
            print("[*] Restarting feed in 5s...")
            time.sleep(5)
            fetch_posts(api)

# Function to handle client connections and send data
def process_client(sock, address):
    """Handle client connections"""
    # Process and send data to connected clients
    print(f"[+] Client connected: {address}")
    try:
        while active:
            try:
                try:
                    data = data_buffer.get(timeout=1)
                    sock.sendall((data + '\n').encode('utf-8'))
                except queue.Empty:
                    ping = json.dumps({'type': 'keepalive', 'timestamp': time.time()})
                    sock.sendall((ping + '\n').encode('utf-8'))
                
            except (socket.error, BrokenPipeError) as e:
                print(f"[!] Client disconnected: {address}")
                break
            except Exception as e:
                print(f"[!] Client error: {address}")
                break
                
    finally:
        sock.close()
        print(f"[-] Client disconnected: {address}")

# Main server function to handle connections and start data collection
def run_server():
    """Main server process"""
    # Set up server socket and start data collection thread
    sock = None
    try:
        # Initialize server socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1)
        
        sock.bind((SERVER_IP, SERVER_PORT))
        sock.listen(5)
        print(f"\n[*] Server active on {SERVER_IP}:{SERVER_PORT}")
        
        feed_thread = threading.Thread(target=fetch_posts, args=(reddit_api,))
        feed_thread.daemon = True
        feed_thread.start()
        
        while active:
            try:
                client, addr = sock.accept()
                client.settimeout(5)
                client_thread = threading.Thread(target=process_client, args=(client, addr))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if active:
                    print(f"[!] Connection error: {e}")
                continue
                
    except Exception as e:
        print(f"[!] Server error: {e}")
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass
        print("[*] Server stopped")

# Main execution block
if __name__ == "__main__":
    # Set up signal handlers and start server
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[*] Manual shutdown")
    finally:
        active = False
        print("[*] Process complete")
