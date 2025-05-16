import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --------------------------
# Enhanced Theme Configuration
# --------------------------
THEMES = {
    "Cyberpunk Neon": {
        "primary": "#F400A1",    # Neon Pink/Magenta
        "secondary": "#00BFFF",   # Neon Blue/Cyan
        "accent": "#39FF14",     # Neon Green/Lime
        "background": "#0A0A1A",  # Very Dark Blue/Black
        "card": "#1A1A2E",       # Dark Purple-Blue
        "text": "#EAEAEA",       # Light Grey / Off-White (High contrast for readability)
        "button_hover": "#FF10B1", # Brighter Pink
        "font": "'Audiowide', sans-serif", # Futuristic font
        "sidebar_bg": "#0F0F20", # Solid very dark blue/purple for good contrast with light text
        "gif": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmpqaTR4aGtycGhrYWk3YmpiY3lyand2bG15NXNvM3Awd2xnc3JmOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qWi6NKfkrt9TgXvIfg/giphy.gif", # Cyberpunk city
        "button_style": """
            border: 2px solid #39FF14; /* Neon Green border */
            border-radius: 4px; /* Sharp edges */
            padding: 10px 24px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            box-shadow: 0 0 3px #39FF14, 0 0 5px #39FF14 inset; /* Subtle glow */
        """,
        "plot_style": {
            "bg": "#1A1A2E",    # Card color
            "grid": "#40405E",   # Muted grid lines
            "text": "#EAEAEA"    # Light text for plot labels
        }
    },
    "Futuristic Theme": {
        "primary": "#00F5FF",    # Cyan
        "secondary": "#FF00E4",  # Pink
        "accent": "#00FF9D",     # Green
        "background": "#0D0221",  # Deep purple
        "card": "#1B065E",       # Purple-blue
        "text": "#FFFFFF",       # White text
        "button_hover": "#00B4D8",  # Light blue
        "font": "'Orbitron', sans-serif",
        "sidebar_bg": "linear-gradient(to bottom, #0D0221, #1B065E)", # Dark gradient, white text is readable
        "gif": "https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif",
        "button_style": """
            border: 2px solid #00F5FF;
            border-radius: 8px;
            padding: 10px 24px;
            letter-spacing: 1px;
        """,
        "plot_style": {
            "bg": "#1B065E",
            "grid": "#4B0082",
            "text": "#FFFFFF"
        }
    }
}

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="FINML Pro Dashboard",
    layout="wide",
    page_icon="🤖", # Added a page icon
    initial_sidebar_state="expanded"
)

# --------------------------
# Initialize Session State
# --------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_engineering_done' not in st.session_state:
    st.session_state.feature_engineering_done = False
if 'train_test_split_done' not in st.session_state:
    st.session_state.train_test_split_done = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "Cyberpunk Neon" # Default to the new theme
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --------------------------
# Enhanced Sidebar with Themes
# --------------------------
with st.sidebar:
    # Theme selector with better styling
    st.markdown("""
    <style>
        .stSelectbox [data-testid='stMarkdownContainer'] {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
   
    st.session_state.current_theme = st.selectbox(
        "🎨 SELECT THEME", # Added emoji for flair
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.current_theme)
    )
   
    # Theme GIF with better styling
    theme_sidebar = THEMES[st.session_state.current_theme] # Use a different variable name to avoid conflict
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0; border: 2px solid {theme_sidebar['accent']}; border-radius: 10px; padding: 10px;">
        <img src="{theme_sidebar['gif']}" width="100%" style="border-radius: 8px;">
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown(f"""<hr style="border: 1px solid {theme_sidebar['primary']}">""", unsafe_allow_html=True)
   
    # Enhanced Data Loading Section
    st.markdown(f"""
    <div style="color: {theme_sidebar['text']}; margin-bottom: 20px;">
        <h2 style="color: {theme_sidebar['primary']};">📊 DATA SOURCE</h2>
        <p style="color: {theme_sidebar['text']};">Load your financial data to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
   
    data_source = st.radio("Select Data Source:",
                           ["Fetch from Yahoo Finance", "Upload CSV"],
                           label_visibility="collapsed")
   
    if data_source == "Fetch from Yahoo Finance":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())
       
        if st.button("📥 FETCH MARKET DATA", key="fetch_data", use_container_width=True): # Added icon and full width
            with st.spinner("Downloading market data..."):
                try:
                    df_fetched = yf.download(ticker, start=start_date, end=end_date) # Use different var name
                    if not df_fetched.empty:
                        st.session_state.df = df_fetched
                        st.session_state.feature_engineering_done = False
                        st.session_state.train_test_split_done = False
                        st.session_state.model_trained = False
                        st.success(f"✅ Successfully fetched {ticker} data!")
                    else:
                        st.error("⚠️ No data returned. Check ticker and date range.")
                except Exception as e:
                    st.error(f"🚫 Error fetching data: {str(e)}")
   
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file) # Use different var name
                # Attempt to set a datetime index if a 'Date' or 'Datetime' column exists
                if 'Date' in df_uploaded.columns:
                    df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
                    df_uploaded.set_index('Date', inplace=True)
                elif 'Datetime' in df_uploaded.columns:
                    df_uploaded['Datetime'] = pd.to_datetime(df_uploaded['Datetime'])
                    df_uploaded.set_index('Datetime', inplace=True)

                st.session_state.df = df_uploaded
                st.session_state.feature_engineering_done = False
                st.session_state.train_test_split_done = False
                st.session_state.model_trained = False
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"🚫 Error reading file: {str(e)}")

# --------------------------
# Apply Selected Theme
# --------------------------
theme = THEMES[st.session_state.current_theme] # This is the main theme variable

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Orbitron:wght@400;700&display=swap');
   
    :root {{
        --primary: {theme['primary']};
        --secondary: {theme['secondary']};
        --accent: {theme['accent']};
        --background: {theme['background']};
        --card: {theme['card']};
        --text: {theme['text']};
        --button_hover: {theme['button_hover']}; /* Added button hover variable */
    }}
   
    html, body, .main {{
        background-color: var(--background);
    }}
   
    .stApp {{
        background-color: var(--background);
        color: var(--text);
        background-image: none; /* Ensure no default Streamlit background image */
    }}
   
    /* Sidebar specific styling */
    [data-testid="stSidebar"] > div:first-child {{
        background: {theme['sidebar_bg']};
        border-right: 2px solid var(--accent);
    }}

    /* Styling for text elements within the sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {{
        color: var(--primary) !important;
        font-family: {theme['font']} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stRadio label {{
        color: var(--text) !important;
        font-family: 'Arial', sans-serif !important; /* Ensure high readability for standard text */
    }}
   
    /* General header styling for main content */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--primary) !important;
        font-family: {theme['font']} !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
   
    /* General paragraph and div styling for main content */
    p, div, input, label {{ /* Keeping this broad, but sidebar overrides above */
        color: var(--text) !important;
        font-family: 'Arial', sans-serif !important; /* Default to Arial for readability */
    }}
   
    /* Ensure specific headers use theme font */
     h1, .stMarkdown h1,
     h2, .stMarkdown h2,
     h3, .stMarkdown h3 {{
        font-family: {theme['font']} !important;
     }}


    .stButton>button {{
        background-color: var(--primary);
        color: var(--text) !important; /* Ensure button text contrasts with button bg */
        /* Check if background is dark, if so, button text should be light */
        /* This might need adjustment based on primary color's brightness */
        {theme['button_style']}
        font-family: {theme['font']};
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }}
   
    .stButton>button:hover {{
        background-color: var(--button_hover);
        color: var(--text); /* Ensure hover text color is also readable */
        transform: scale(1.05);
        box-shadow: 0 0 10px var(--accent), 0 0 15px var(--accent); /* Enhanced hover glow */
    }}
   
    .metric-card {{
        background-color: var(--card);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid var(--primary);
    }}
   
    .progress-container {{
        margin: 30px 0;
    }}
   
    .progress-step {{
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }}
   
    .progress-icon {{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: var(--secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
        color: var(--background); /* Icon text contrasts with icon bg */
    }}
   
    .progress-icon.completed {{
        background-color: var(--accent);
        color: var(--background); /* Icon text contrasts with icon bg */
        box-shadow: 0 0 10px var(--accent);
    }}
   
    .progress-text {{
        flex-grow: 1;
        color: var(--text);
        font-family: 'Arial', sans-serif;
    }}
   
    .progress-text.completed {{
        color: var(--accent);
        font-weight: bold;
    }}
   
    .stDataFrame {{
        background-color: var(--card);
    }}
   
    /* Ensure input fields background and text are themed */
    .stTextInput input, .stDateInput input {{
        background-color: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--secondary) !important;
    }}
    .stSelectbox > div > div {{
        background-color: var(--card) !important;
        border: 1px solid var(--secondary) !important;
    }}
    .stSelectbox > div > div > div {{
         color: var(--text) !important;
    }}
   
    .st-bb, .st-at, .st-af {{ /* These are Streamlit's internal classes for input widgets */
        background-color: var(--card) !important; /* May not always work, Streamlit styling can be tricky */
    }}
   
    /* Plotly chart styling */
    .js-plotly-plot .plotly {{
        background-color: {theme['plot_style']['bg']} !important;
    }}
    .js-plotly-plot .gridlayer .grid path {{ /* More specific selector for grid lines */
        stroke: {theme['plot_style']['grid']} !important;
    }}
    .js-plotly-plot .xtitle, .js-plotly-plot .ytitle, .js-plotly-plot .tick text {{ /* Target all plot text */
        fill: {theme['plot_style']['text']} !important;
        font-family: 'Arial', sans-serif !important; /* Consistent font for plots */
    }}
     .js-plotly-plot .legendtext {{
        fill: {theme['plot_style']['text']} !important;
    }}
   
    .stAlert {{
        background-color: var(--card) !important;
        border-left: 4px solid var(--primary) !important;
    }}
    .stAlert p {{  /* Ensure alert text is readable */
        color: var(--text) !important;
    }}

</style>
""", unsafe_allow_html=True)

# --------------------------
# Header Section
# --------------------------
st.markdown(f"""
<div style="background: linear-gradient(to right, {theme['primary']}, {theme['secondary']});
            padding: 30px;
            border-radius: 10px;
            color: #FFFFFF; /* Force white for header text for max contrast on gradient */
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            margin-bottom: 30px;
            border: 2px solid {theme['accent']};
            text-align: center;">
    <h1 style="color: #FFFFFF !important; margin: 0; font-size: 2.5rem; font-family: {theme['font']} !important;">🚀 FINML PRO DASHBOARD</h1>
    <p style="color: #FFFFFF !important; margin: 10px 0 0; font-size: 1.1rem; letter-spacing: 1px; font-family: 'Arial', sans-serif !important;">
        Advanced Financial Machine Learning Pipeline with Interactive Visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Pipeline Progress Tracker
# --------------------------
st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>📈 PIPELINE PROGRESS</h3>", unsafe_allow_html=True)
progress_cols = st.columns(5)

steps_config = [
    ("Data Loaded", lambda s: s.df is not None),
    ("Feature Engineering", lambda s: s.feature_engineering_done),
    ("Train/Test Split", lambda s: s.train_test_split_done),
    ("Model Training", lambda s: s.model_trained),
    ("Evaluation", lambda s: s.model_trained and 'predictions' in s)
]

for i, (label, condition) in enumerate(steps_config):
    with progress_cols[i]:
        completed = condition(st.session_state)
        st.markdown(f"""
        <div class="metric-card">
            <div class="progress-step">
                <div class="progress-icon {'completed' if completed else ''}">{i+1}</div>
                <div class="progress-text {'completed' if completed else ''}">{label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------
# Main Content Area (Fixed ML Functionality)
# --------------------------
if st.session_state.df is not None:
    df = st.session_state.df.copy()
   
    # Data Preview Section
    with st.expander("📋 DATA PREVIEW & INITIAL ANALYSIS (CLICK TO EXPAND)", expanded=True):
        st.dataframe(df.head().style.background_gradient(cmap='Blues'), height=300) # Show head for brevity
       
        # Basic Statistics
        st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>📊 BASIC STATISTICS</h3>", unsafe_allow_html=True)
        stats_cols = st.columns(4)
        stats_cols[0].metric("Total Records", f"{len(df):,}")
        stats_cols[1].metric("Columns", len(df.columns))
        is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        stats_cols[2].metric("Start Date", df.index.min().strftime('%Y-%m-%d') if is_datetime_index and not df.empty else "N/A")
        stats_cols[3].metric("End Date", df.index.max().strftime('%Y-%m-%d') if is_datetime_index and not df.empty else "N/A")
       
        # Quick Visualization
        st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>📈 PRICE TREND</h3>", unsafe_allow_html=True)
        numeric_cols_preview = df.select_dtypes(include=np.number).columns
        if not numeric_cols_preview.empty:
            selected_column = st.selectbox("Select column to visualize:", numeric_cols_preview)
            fig = px.line(df, y=selected_column, title=f"{selected_column} Over Time")
            fig.update_layout(
                plot_bgcolor=theme['plot_style']['bg'],
                paper_bgcolor=theme['background'],
                font_color=theme['text'],
                title_font_family=theme['font'],
                xaxis_title_font_family='Arial',
                yaxis_title_font_family='Arial'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for visualization in the uploaded data.")
   
    # --------------------------
    # Step 1: Preprocessing
    # --------------------------
    st.markdown("---")
    st.markdown(f"<h2 style='font-family: {theme['font']}; color: {theme['primary']};'>🛠️ STEP 1: DATA PREPROCESSING</h2>", unsafe_allow_html=True)
   
    if st.button("CLEAN & PREPARE DATA", key="preprocess", use_container_width=True):
        with st.spinner("Processing data..."):
            missing_before = df.isnull().sum().sum()
            df.dropna(inplace=True) # Simple NA drop for this example
            missing_after = df.isnull().sum().sum()
            st.session_state.df = df
           
            st.success(f"✅ Data cleaned! Removed {missing_before - missing_after} missing values.")
            st.session_state.feature_engineering_done = False # Reset downstream steps
            st.session_state.train_test_split_done = False
            st.session_state.model_trained = False
   
    # --------------------------
    # Step 2: Feature Engineering
    # --------------------------
    st.markdown("---")
    st.markdown(f"<h2 style='font-family: {theme['font']}; color: {theme['primary']};'>🔬 STEP 2: FEATURE ENGINEERING</h2>", unsafe_allow_html=True)
   
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        # Default to 'Close' or 'Adj Close' if available, else first numeric
        default_col = None
        if 'Close' in numeric_cols: default_col = 'Close'
        elif 'Adj Close' in numeric_cols: default_col = 'Adj Close'
        elif numeric_cols: default_col = numeric_cols[0]
       
        selected_close_col = st.selectbox(
            "Select price column for return calculation (e.g., 'Close' or 'Adj Close'):",
            numeric_cols,
            index=numeric_cols.index(default_col) if default_col else 0
        )
       
        if st.button("CALCULATE FINANCIAL RETURNS", key="feature_eng", use_container_width=True):
            with st.spinner("Engineering features..."):
                df['Return'] = df[selected_close_col].pct_change()
                df.dropna(inplace=True) # Drop NaNs created by pct_change
                st.session_state.df = df
                st.session_state.feature_engineering_done = True
                st.session_state.train_test_split_done = False
                st.session_state.model_trained = False
               
                st.success(f"✅ Returns calculated from '{selected_close_col}' and added as 'Return' column!")
               
                # Visualize returns
                st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>RETURNS ANALYSIS</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x='Return', nbins=50,
                                            title="Returns Distribution",
                                            color_discrete_sequence=[theme['primary']])
                    fig_hist.update_layout(
                        plot_bgcolor=theme['plot_style']['bg'], paper_bgcolor=theme['background'],
                        font_color=theme['text'], title_font_family=theme['font']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
               
                with col2:
                    fig_line = px.line(df, y='Return', title="Returns Over Time",
                                       color_discrete_sequence=[theme['accent']])
                    fig_line.update_layout(
                        plot_bgcolor=theme['plot_style']['bg'], paper_bgcolor=theme['background'],
                        font_color=theme['text'], title_font_family=theme['font']
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns found for feature engineering. Please load appropriate data.")

    # --------------------------
    # Step 3: Train/Test Split (Fixed for simplicity: previous day's return to predict current day's return)
    # --------------------------
    st.markdown("---")
    st.markdown(f"<h2 style='font-family: {theme['font']}; color: {theme['primary']};'>쪼 STEP 3: TRAIN/TEST SPLIT</h2>", unsafe_allow_html=True)
   
    if st.button("SPLIT DATA FOR TRAINING & TESTING", disabled=not st.session_state.feature_engineering_done, key="split", use_container_width=True):
        if 'Return' in df.columns:
            with st.spinner("Splitting data..."):
                # Feature: Previous day's return. Target: Current day's return.
                df['Prev_Return'] = df['Return'].shift(1)
                df.dropna(inplace=True) # Drop NaN from shift
               
                if df.empty or len(df) < 10: # Check for sufficient data
                    st.error("⚠️ Not enough data to perform train/test split after feature creation. Need at least 10 data points.")
                else:
                    X = df[['Prev_Return']]
                    y = df['Return']
                   
                    # Chronological split (80% train, 20% test)
                    split_idx = int(len(df) * 0.8)
                    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
                    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
                   
                    st.session_state.update({
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'train_test_split_done': True, 'model_trained': False
                    })
                   
                    st.success("✅ Data split into training and testing sets (80/20 chronological split)!")
                   
                    st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>DATA SPLIT VISUALIZATION</h3>", unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X_train.index, y=y_train, mode='lines', name='Training Data', line=dict(color=theme['primary'], width=2)))
                    fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Test Data', line=dict(color=theme['accent'], width=2)))
                    fig.update_layout(
                        title="Train/Test Split Timeline (Returns)", xaxis_title="Date", yaxis_title="Return",
                        plot_bgcolor=theme['plot_style']['bg'], paper_bgcolor=theme['background'],
                        font_color=theme['text'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        title_font_family=theme['font']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                   
                    st.write(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%) | Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        else:
            st.warning("⚠️ 'Return' column not found. Please complete feature engineering first.")

    # --------------------------
    # Step 4: Model Training (Fixed: Linear Regression)
    # --------------------------
    st.markdown("---")
    st.markdown(f"<h2 style='font-family: {theme['font']}; color: {theme['primary']};'>🧠 STEP 4: MODEL TRAINING (LINEAR REGRESSION)</h2>", unsafe_allow_html=True)
   
    if st.button("TRAIN PREDICTIVE MODEL", disabled=not st.session_state.train_test_split_done, key="train", use_container_width=True):
        if st.session_state.X_train is not None and st.session_state.y_train is not None:
            if st.session_state.X_train.empty or st.session_state.y_train.empty:
                st.error("⚠️ Training data is empty. Cannot train model.")
            else:
                with st.spinner("Training Linear Regression model..."):
                    model = LinearRegression()
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                   
                    train_pred = model.predict(st.session_state.X_train)
                    test_pred = model.predict(st.session_state.X_test)
                   
                    st.session_state.update({
                        'model': model, 'model_trained': True,
                        'train_pred': train_pred, 'predictions': test_pred
                    })
                   
                    st.success("✅ Linear Regression model trained successfully!")
                   
                    st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>MODEL COEFFICIENTS</h3>", unsafe_allow_html=True)
                    coef_col1, coef_col2 = st.columns(2)
                    coef_col1.metric("Intercept (α)", f"{model.intercept_:.6f}")
                    coef_col2.metric(f"Coefficient (β) for 'Prev_Return'", f"{model.coef_[0]:.6f}")
                   
                    train_r2 = r2_score(st.session_state.y_train, train_pred)
                    st.markdown(f"**Training R² Score:** {train_r2:.4f}")
        else:
            st.warning("⚠️ Training data not available. Please complete the train/test split first.")

    # --------------------------
    # Step 5: Model Evaluation
    # --------------------------
    st.markdown("---")
    st.markdown(f"<h2 style='font-family: {theme['font']}; color: {theme['primary']};'>📉 STEP 5: MODEL EVALUATION</h2>", unsafe_allow_html=True)
   
    if st.button("EVALUATE MODEL PERFORMANCE", disabled=not st.session_state.model_trained, key="evaluate", use_container_width=True):
        if all(key in st.session_state for key in ['model', 'X_test', 'y_test', 'predictions']):
            with st.spinner("Evaluating model..."):
                y_test_eval = st.session_state.y_test
                predictions_eval = st.session_state.predictions
               
                if y_test_eval.empty or len(predictions_eval) == 0:
                     st.error("⚠️ Test data or predictions are empty. Cannot evaluate.")
                else:
                    mse = mean_squared_error(y_test_eval, predictions_eval)
                    r2 = r2_score(y_test_eval, predictions_eval)
                    # Ensure y_test and predictions are 1D arrays for corrcoef
                    y_test_flat = y_test_eval.values.flatten() if isinstance(y_test_eval, (pd.Series, pd.DataFrame)) else y_test_eval
                    pred_flat = predictions_eval.flatten() if isinstance(predictions_eval, (pd.Series, pd.DataFrame, np.ndarray)) else predictions_eval
                   
                    # Check for constant arrays which cause issues with correlation
                    if np.all(y_test_flat == y_test_flat[0]) or np.all(pred_flat == pred_flat[0]):
                        corr = np.nan # Correlation is undefined or uninformative
                        st.warning("⚠️ Correlation cannot be computed accurately due to constant actual or predicted values in the test set.")
                    else:
                        corr = np.corrcoef(y_test_flat, pred_flat)[0,1]

                    st.markdown(f"<h3 style='font-family: {theme['font']}; color: {theme['primary']};'>PERFORMANCE METRICS (TEST SET)</h3>", unsafe_allow_html=True)
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                    metric_col2.metric("R² Score", f"{r2:.4f}")
                    metric_col3.metric("Correlation", f"{corr:.4f}" if not np.isnan(corr) else "N/A")
                   
                    tab_titles = ["📈 Predictions Timeline", "🎯 Actual vs. Predicted", "⚠️ Error Analysis", "💰 Cumulative Returns"]
                    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
                   
                    common_layout_updates = dict(
                        plot_bgcolor=theme['plot_style']['bg'], paper_bgcolor=theme['background'],
                        font_color=theme['text'], title_font_family=theme['font'],
                        xaxis_title_font_family='Arial', yaxis_title_font_family='Arial'
                    )

                    with tab1:
                        fig_timeline = go.Figure()
                        fig_timeline.add_trace(go.Scatter(x=y_test_eval.index, y=y_test_eval, name='Actual Returns', line=dict(color=theme['primary'], width=2)))
                        fig_timeline.add_trace(go.Scatter(x=y_test_eval.index, y=predictions_eval, name='Predicted Returns', line=dict(color=theme['accent'], width=2)))
                        fig_timeline.update_layout(title="Actual vs Predicted Returns Over Time", xaxis_title="Date", yaxis_title="Return", hovermode="x unified", **common_layout_updates)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                   
                    with tab2:
                        fig_scatter = px.scatter(x=y_test_eval, y=predictions_eval, trendline="ols", title="Actual vs Predicted Returns", labels={'x': 'Actual Return', 'y': 'Predicted Return'}, color_discrete_sequence=[theme['primary']])
                        fig_scatter.add_shape(type="line", line=dict(dash='dash', color=theme['accent']), x0=y_test_eval.min(), y0=y_test_eval.min(), x1=y_test_eval.max(), y1=y_test_eval.max())
                        fig_scatter.update_layout(**common_layout_updates)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                   
                    with tab3:
                        errors = y_test_eval - predictions_eval
                        col_err1, col_err2 = st.columns(2)
                        with col_err1:
                            fig_hist_err = px.histogram(x=errors, title="Prediction Error Distribution", labels={'x': 'Prediction Error'}, color_discrete_sequence=[theme['accent']], nbins=50)
                            fig_hist_err.add_vline(x=0, line_dash="dash", line_color=theme['primary'])
                            fig_hist_err.update_layout(**common_layout_updates)
                            st.plotly_chart(fig_hist_err, use_container_width=True)
                        with col_err2:
                            fig_box_err = px.box(y=errors, title="Error Distribution Summary", labels={'y': 'Prediction Error'}, color_discrete_sequence=[theme['accent']])
                            fig_box_err.update_layout(**common_layout_updates)
                            st.plotly_chart(fig_box_err, use_container_width=True)
                   
                    with tab4:
                        cumulative_actual = (1 + y_test_eval).cumprod() - 1
                        cumulative_pred = (1 + pd.Series(predictions_eval, index=y_test_eval.index)).cumprod() - 1
                        fig_cumulative = go.Figure()
                        fig_cumulative.add_trace(go.Scatter(x=cumulative_actual.index, y=cumulative_actual, name='Actual Cumulative Returns', line=dict(color=theme['primary'], width=3)))
                        fig_cumulative.add_trace(go.Scatter(x=cumulative_pred.index, y=cumulative_pred, name='Predicted Cumulative Returns', line=dict(color=theme['accent'], width=3)))
                        fig_cumulative.update_layout(title="Cumulative Returns Comparison (Test Set)", xaxis_title="Date", yaxis_title="Cumulative Return", hovermode="x unified", **common_layout_updates)
                        st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.warning("⚠️ Model, test data, or predictions not found. Please complete all previous steps.")
else:
    st.info("👋 Welcome to FINML Pro! Please load data from the sidebar to start your analysis.")


# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {theme['text']}; font-size: 14px; margin-top: 50px; font-family: 'Arial', sans-serif;">
    <p>Developed with 💻 by Ibrahim Aziz and Ahmed Saleh Riaz</p>
    <p>© {datetime.now().year} Financial Machine Learning Dashboard</p>
</div>
""", unsafe_allow_html=True)
