import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime
import os
import glob
import time
from typing import Dict, List, Optional

# Define data paths
BASE_DIR = 'data/processed'
SENT_DIR = f'{BASE_DIR}/sentiment'
STATS_DIR = f'{BASE_DIR}/subreddit_stats'
REFS_DIR = f'{BASE_DIR}/references'

# App config
st.set_page_config(
    page_title="Reddit Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'time_window' not in st.session_state:
    st.session_state.time_window = '1h'
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = False

# Sidebar
with st.sidebar:
    st.image("BTS_Logo.png", width=300)
    st.markdown("---")
    st.header("Settings")
    
    auto_update = st.checkbox("Auto-update (30s)", value=True)
    
    time_window = st.selectbox(
        "Time Window",
        options=['1h', '6h', '24h', 'All'],
        index=0
    )

def setup_dirs():
    """Ensure required data directories exist"""
    try:
        for path in [BASE_DIR, SENT_DIR, STATS_DIR, REFS_DIR]:
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        st.error(f"Directory error: {str(e)}")

def get_newest_file_time(path: str) -> Optional[datetime]:
    """Get the timestamp of the most recently modified file in a directory"""
    try:
        if not os.path.exists(path):
            return None
        files = glob.glob(f"{path}/*.parquet")
        if not files:
            return None
        return datetime.fromtimestamp(os.path.getctime(max(files, key=os.path.getctime)))
    except Exception as e:
        st.error(f"File time error: {str(e)}")
        return None

@st.cache_data(ttl=15)
def get_current_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load the latest data snapshot from parquet files"""
    data = {
        'sentiment': None,
        'subreddit_stats': None,
        'references': None
    }
    
    try:
        # Get sentiment data
        if os.path.exists(SENT_DIR):
            sent_files = glob.glob(f"{SENT_DIR}/*.parquet")
            if sent_files:
                newest_sent = max(sent_files, key=os.path.getctime)
                try:
                    df = pd.read_parquet(newest_sent)
                    if not df.empty and 'average_sentiment' in df.columns:
                        if 'timestamp' not in df.columns:
                            df['timestamp'] = pd.to_datetime(os.path.getctime(newest_sent), unit='s')
                        data['sentiment'] = df
                        print(f"Loaded {len(df)} sentiment records")
                    else:
                        print(f"Missing columns in sentiment file: {df.columns.tolist()}")
                except Exception as e:
                    st.error(f"Sentiment file error: {str(e)}")
                    print(f"Sentiment file error: {str(e)}")
            else:
                print("No sentiment files")
        else:
            print("No sentiment directory")
        
        # Get subreddit stats
        if os.path.exists(STATS_DIR):
            stats_files = glob.glob(f"{STATS_DIR}/*.parquet")
            if stats_files:
                current_time = datetime.now()
                recent_files = [
                    f for f in stats_files 
                    if (current_time - datetime.fromtimestamp(os.path.getctime(f))).total_seconds() <= 3600
                ]
                
                if recent_files:
                    dfs = []
                    for file in recent_files:
                        df = pd.read_parquet(file)
                        if 'timestamp' not in df.columns:
                            df['timestamp'] = pd.to_datetime(os.path.getctime(file), unit='s')
                        dfs.append(df)
                    
                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        data['subreddit_stats'] = combined_df.groupby('subreddit').agg({
                            'post_count': 'sum',
                            'unique_authors': 'sum'
                        }).reset_index()
                        data['subreddit_stats']['timestamp'] = current_time
        
        # Get references data
        if os.path.exists(REFS_DIR):
            ref_files = glob.glob(f"{REFS_DIR}/*.parquet")
            if ref_files:
                newest_refs = max(ref_files, key=os.path.getctime)
                df = pd.read_parquet(newest_refs)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(os.path.getctime(newest_refs), unit='s')
                data['references'] = df
    except Exception as e:
        st.error(f"Data load error: {str(e)}")
    
    return data

@st.cache_data(ttl=5)
def get_historical_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all historical data from parquet files"""
    data = {
        'sentiment': None,
        'subreddit_stats': None,
        'references': None
    }
    
    try:
        # Get historical sentiment
        if os.path.exists(SENT_DIR):
            sent_dfs = []
            for file in glob.glob(f"{SENT_DIR}/*.parquet"):
                df = pd.read_parquet(file)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(os.path.getctime(file), unit='s')
                sent_dfs.append(df)
            if sent_dfs:
                data['sentiment'] = pd.concat(sent_dfs, ignore_index=True)
                data['sentiment'] = data['sentiment'].sort_values('timestamp')
                data['sentiment'] = data['sentiment'].drop_duplicates(subset=['timestamp'], keep='last')
        
        # Get historical stats
        if os.path.exists(STATS_DIR):
            stats_dfs = []
            for file in glob.glob(f"{STATS_DIR}/*.parquet"):
                df = pd.read_parquet(file)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(os.path.getctime(file), unit='s')
                stats_dfs.append(df)
            if stats_dfs:
                data['subreddit_stats'] = pd.concat(stats_dfs, ignore_index=True)
                data['subreddit_stats'] = data['subreddit_stats'].sort_values('timestamp')
                data['subreddit_stats'] = data['subreddit_stats'].drop_duplicates(subset=['timestamp'], keep='last')
        
        # Get historical references
        if os.path.exists(REFS_DIR):
            ref_dfs = []
            for file in glob.glob(f"{REFS_DIR}/*.parquet"):
                df = pd.read_parquet(file)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(os.path.getctime(file), unit='s')
                ref_dfs.append(df)
            if ref_dfs:
                data['references'] = pd.concat(ref_dfs, ignore_index=True)
                data['references'] = data['references'].sort_values('timestamp')
                data['references'] = data['references'].drop_duplicates(subset=['timestamp'], keep='last')
    except Exception as e:
        st.error(f"Historical data error: {str(e)}")
    
    return data

def check_updates() -> bool:
    """Check if new data files are available compared to the last update time"""
    try:
        latest_times = {
            'sentiment': get_newest_file_time(SENT_DIR),
            'subreddit_stats': get_newest_file_time(STATS_DIR),
            'references': get_newest_file_time(REFS_DIR)
        }
        
        if not st.session_state.initial_load:
            st.session_state.initial_load = True
            st.session_state.last_update = datetime.now()
            return True
        
        for time in latest_times.values():
            if time and time is not None and time > st.session_state.last_update:
                return True
        
        return False
    except Exception as e:
        st.error(f"Update check error: {str(e)}")
        return False

def filter_timeframe(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Filter a DataFrame to include data within the selected time window"""
    try:
        if df is None or df.empty:
            return df
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        latest = df['timestamp'].max()
        
        if window == '1h':
            cutoff = latest - pd.Timedelta(hours=1)
        elif window == '6h':
            cutoff = latest - pd.Timedelta(hours=6)
        elif window == '24h':
            cutoff = latest - pd.Timedelta(hours=24)
        else:
            return df
        
        filtered = df[df['timestamp'] >= cutoff]
        
        if filtered.empty:
            return df
            
        return filtered
    except Exception as e:
        st.error(f"Time filter error: {str(e)}")
        return df

def plot_sentiment(data: Dict[str, Optional[pd.DataFrame]], window: str) -> Optional[go.Figure]:
    """Generate a plot for sentiment trends over time"""
    try:
        if data['sentiment'] is None or data['sentiment'].empty:
            st.warning("No sentiment data")
            return None
        
        df = filter_timeframe(data['sentiment'], window)
        
        if df is None or df.empty:
            st.warning("No data for selected window")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['average_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#1f77b4', width=2)
        ))
        
        window_size = min(5, len(df))
        if window_size > 1:
            ma = df['average_sentiment'].rolling(window=window_size).mean()
            std = df['average_sentiment'].rolling(window=window_size).std()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=ma,
                mode='lines',
                name=f'{window_size}-point MA',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=ma + 2*std,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=ma - 2*std,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)',
                name='95% CI'
            ))
        
        fig.update_layout(
            title='Sentiment Trend',
            xaxis_title='Time',
            yaxis_title='Sentiment Score',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Sentiment plot error: {str(e)}")
        return None

def plot_activity(data: Dict[str, Optional[pd.DataFrame]], window: str, subs: List[str]) -> Optional[go.Figure]:
    """Generate a plot for subreddit activity over time"""
    try:
        if data['subreddit_stats'] is None or data['subreddit_stats'].empty:
            st.warning("No activity data")
            return None
        
        df = filter_timeframe(data['subreddit_stats'], window)
        
        if df is None or df.empty:
            st.warning("No data for selected window")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if subs:
            df = df[df['subreddit'].isin(subs)]
            if df.empty:
                st.warning("No data for selected subreddits")
                return None
        
        activity = df.groupby(['timestamp', 'subreddit'])['post_count'].sum().reset_index()
        
        fig = go.Figure()
        
        for sub in activity['subreddit'].unique():
            sub_data = activity[activity['subreddit'] == sub]
            fig.add_trace(go.Scatter(
                x=sub_data['timestamp'],
                y=sub_data['post_count'],
                mode='lines+markers',
                name=sub,
                hovertemplate='<b>%{x}</b><br>Posts: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Subreddit Activity',
            xaxis_title='Time',
            yaxis_title='Posts',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Activity plot error: {str(e)}")
        return None

def plot_refs(data: Dict[str, Optional[pd.DataFrame]], window: str) -> Optional[go.Figure]:
    """Generate a plot for reference trends over time"""
    try:
        if data['references'] is None or data['references'].empty:
            st.warning("No reference data")
            return None
        
        df = filter_timeframe(data['references'], window)
        
        if df is None or df.empty:
            st.warning("No data for selected window")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        ref_data = df.groupby('timestamp').agg({
            'total_user_refs': 'sum',
            'total_sub_refs': 'sum',
            'total_urls': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for col, color in zip(['total_user_refs', 'total_sub_refs', 'total_urls'], colors):
            fig.add_trace(go.Scatter(
                x=ref_data['timestamp'],
                y=ref_data[col],
                mode='lines+markers',
                name=col.replace('total_', '').replace('_refs', ' Refs'),
                line=dict(color=color, width=2),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Reference Trends',
            xaxis_title='Time',
            yaxis_title='Count',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Reference plot error: {str(e)}")
        return None

def show_metrics(current: Dict[str, Optional[pd.DataFrame]], historical: Dict[str, Optional[pd.DataFrame]]) -> None:
    """Display key current metrics using columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current Sentiment")
        sent_df = current['sentiment'] if current['sentiment'] is not None and not current['sentiment'].empty else historical['sentiment']
        
        if sent_df is not None and not sent_df.empty:
            current_sent = sent_df['average_sentiment'].iloc[-1]
            delta_color = 'normal' if current_sent > 0 else 'inverse' if current_sent < 0 else 'off'
            st.metric(
                "Sentiment",
                f"{current_sent:.3f}",
                delta=f"{current_sent:.3f}",
                delta_color=delta_color
            )
        else:
            st.write("No sentiment data")
    
    with col2:
        st.subheader("Current Activity")
        if current['subreddit_stats'] is not None and not current['subreddit_stats'].empty:
            total_posts = current['subreddit_stats']['post_count'].sum()
            unique_subs = len(current['subreddit_stats'])
            
            st.metric(
                "Posts",
                f"{total_posts:,}",
                delta=f"{total_posts:,} posts/hour"
            )
            st.metric(
                "Active Subs",
                f"{unique_subs:,}",
                delta=f"{unique_subs:,} active"
            )
            
            if not current['subreddit_stats'].empty:
                top_subs = current['subreddit_stats'].nlargest(3, 'post_count')
                st.write("Top Subreddits:")
                for _, row in top_subs.iterrows():
                    st.write(f"- r/{row['subreddit']}: {row['post_count']:,} posts")
        else:
            st.write("No activity data")
    
    with col3:
        st.subheader("Current References")
        if current['references'] is not None and not current['references'].empty:
            latest_refs = current['references'].iloc[-1]
            
            total_refs = (
                latest_refs['total_user_refs'] +
                latest_refs['total_sub_refs'] +
                latest_refs['total_urls']
            )
            
            st.metric("Total Refs", f"{total_refs:,.0f}")
            st.write("Breakdown:")
            st.write(f"- Users: {latest_refs['total_user_refs']:,.0f}")
            st.write(f"- Subs: {latest_refs['total_sub_refs']:,.0f}")
            st.write(f"- URLs: {latest_refs['total_urls']:,.0f}")
        else:
            st.write("No reference data")

def main():
    # Display logo and title
    st.image("Wallstreet_Bets_Logo.png", width=200)
    st.title("Reddit Financial Real-Time Dashboard")
    
    # Ensure data directories are set up
    setup_dirs()
    
    # Load current and historical data
    current_data = get_current_data()
    historical_data = get_historical_data()
    
    # Display current metrics
    show_metrics(current_data, historical_data)
    
    st.subheader("Historical Trends")
    
    # Create tabs for historical plots
    tab1, tab2, tab3 = st.tabs(["Sentiment", "Activity", "References"])
    
    with tab1:
        sent_fig = plot_sentiment(historical_data, time_window)
        if sent_fig:
            st.plotly_chart(sent_fig, use_container_width=True)
        else:
            st.write("No sentiment data")
    
    with tab2:
        act_fig = plot_activity(
            historical_data,
            time_window,
            []
        )
        if act_fig:
            st.plotly_chart(act_fig, use_container_width=True)
        else:
            st.write("No activity data")
    
    with tab3:
        ref_fig = plot_refs(historical_data, time_window)
        if ref_fig:
            st.plotly_chart(ref_fig, use_container_width=True)
        else:
            st.write("No reference data")
    
    # Auto-update logic
    if auto_update:
        if check_updates():
            st.session_state.last_update = datetime.now()
            st.rerun()
        else:
            time.sleep(30)
            st.rerun()

if __name__ == "__main__":
    main() 