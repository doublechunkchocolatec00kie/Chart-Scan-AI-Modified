import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
from ultralytics import YOLO
import time

# Replace the relative path to your weight file
model_path = 'weights/custom_yolov8.pt'

# Logo URL
logo_url = "images/chartscan.png"

# Setting page layout
st.set_page_config(
    page_title="ChartScanAI",  # Setting page title
    page_icon="📊",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Function to fetch real-time chart data
def fetch_real_time_data(ticker, interval="1m"):
    # Adjust the period based on the interval to get enough data
    if interval == "1m":
        period = "1d"  # 1 day for daily candles
    elif interval == "1h":
        period = "5d"  # 1 day for hourly candles
    elif interval == "1wk":
        period = "5y"  # 5 years for weekly candles
    
    # Download data using yfinance
    data = yf.download(ticker, interval=interval, period=period)  # Adjust period based on the interval
    if data.empty:
        st.error("No data found for the specified ticker and interval.")
    return data

# Function to plot the chart and convert it to an image
def generate_chart_image(chart_data):
    # Plot the chart
    fig, ax = mpf.plot(chart_data, type="candle", style="yahoo",
                       title=f"Real-time Chart",
                       axisoff=True,
                       ylabel="",
                       ylabel_lower="",
                       volume=False,
                       figsize=(18, 6.5),
                       returnfig=True)

    # Convert plot to an image format
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100)  # Ensure DPI is set here
    buffer.seek(0)
    image = Image.open(buffer)
    return image

# Function to perform object detection
def detect_objects(image):
    # Perform object detection
    results = model.predict(image, conf=confidence)
    
    # Show the results
    res_plotted = results[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption="Real-time Object Detection", use_column_width=True)
    
    # Display detected objects' coordinates
    boxes = results[0].boxes
    try:
        with st.expander("Detection Results"):
            for box in boxes:
                st.write(box.xywh)
    except Exception as ex:
        st.write("Error displaying detection results.")

# Load the model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Creating sidebar
with st.sidebar:
    # Add a logo to the top of the sidebar
    st.image(logo_url, use_column_width="auto")
    st.write("")
    st.header("Configurations")  # Adding header to sidebar

    # Add ticker and interval input
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    interval = st.selectbox("Select Interval", ["1m", "1h", "1wk"])

    # Model confidence slider
    confidence = float(st.slider("Select Model Confidence", 25, 100, 30)) / 100

# Main page heading
st.title("ChartScanAI")
st.caption('📈 Real-time chart analysis and object detection.')

# Create two rows: one for the full chart and one for detection results
chart_container = st.container()
detection_container = st.container()

# Loop for real-time chart fetching and object detection
if ticker:
    while True:
        # Fetch chart data
        chart_data = fetch_real_time_data(ticker, interval)
        if not chart_data.empty:
            with chart_container:
                # Generate chart image and display in one row
                chart_image = generate_chart_image(chart_data)
                st.image(chart_image, caption=f"{ticker} Chart", use_column_width=True)
            
            with detection_container:
                # Automatically detect objects and display results in the second row
                detect_objects(chart_image)
        
        time.sleep(1)  # Fetch data every 1 seconds
        st.experimental_rerun()
