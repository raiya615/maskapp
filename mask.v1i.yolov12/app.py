import streamlit as st
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
DATA_FILE = os.path.join(BASE_DIR, 'detection_log.csv')

st.set_page_config(page_title="Mask Detection App", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

    # Robust loading with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Integrity check / Warm-up
            file_size = os.path.getsize(MODEL_PATH)
            if file_size < 1000:
                st.error(f"Model file is too small ({file_size} bytes). Likely corrupted.")
                return None
            
            # Force a small read to ensure file system is ready
            with open(MODEL_PATH, 'rb') as f:
                f.read(1024)

            return YOLO(MODEL_PATH)
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(1) # Wait a bit before retrying
                continue
            
            st.error(f"Error loading model (Attempt {attempt+1}/{max_retries}): {e}")
            return None
    return None

def save_data(counts):
    timestamp = datetime.now()
    date_str = timestamp.strftime('%Y-%m-%d')
    time_str = timestamp.strftime('%H:%M:%S')
    
    total = sum(counts.values())
    mask_rate = counts['with_mask'] / total if total > 0 else 0
    
    new_data = {
        'Date': date_str,
        'Time': time_str,
        'Total': total,
        'With Mask': counts['with_mask'],
        'Without Mask': counts['without_mask'],
        'Incorrect Mask': counts['mask_weared_incorrect'],
        'Mask Rate': mask_rate
    }
    
    df = pd.DataFrame([new_data])
    
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode='a', header=False, index=False)

def main():
    st.title("😷 Mask Detection & Analytics")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Detection", "Analytics"])

    if page == "Detection":
        st.header("Camera Detection")
        
        model = load_model()
        if model is None:
            st.warning("Model not found. Please train the model first.")
            return

        # Camera input
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Inference
            results = model(cv2_img)
            
            # Count results
            counts = {
                'with_mask': 0,
                'without_mask': 0,
                'mask_weared_incorrect': 0
            }
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    if cls_name in counts:
                        counts[cls_name] += 1
            
            # Display annotated image
            res_plotted = results[0].plot()
            st.image(res_plotted, channels="BGR", caption="Detected Image")
            
            # Display stats
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("With Mask", counts['with_mask'])
            col2.metric("Without Mask", counts['without_mask'])
            col3.metric("Incorrect", counts['mask_weared_incorrect'])
            
            total = sum(counts.values())
            if total > 0:
                rate = counts['with_mask'] / total
                st.metric("Mask Rate", f"{rate:.1%}")
                
                # Save data
                save_data(counts)
                st.success("Data saved successfully!")
            else:
                st.info("No people detected.")

    elif page == "Analytics":
        st.header("Analytics Dashboard")
        
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Monthly Graph
            st.subheader("Monthly Trend")
            current_month = datetime.now().month
            monthly_df = df[df['Date'].dt.month == current_month]
            
            if not monthly_df.empty:
                # Calculate daily stats by summing counts first
                daily_avg = monthly_df.groupby('Date').agg({
                    'Total': 'sum',
                    'With Mask': 'sum'
                }).reset_index()
                
                # Calculate weighted rate
                daily_avg['Mask Rate'] = daily_avg.apply(
                    lambda x: x['With Mask'] / x['Total'] if x['Total'] > 0 else 0, axis=1
                )
                # st.line_chart(daily_avg.set_index('Date')['Mask Rate'])
                fig, ax = plt.subplots()
                ax.plot(daily_avg['Date'], daily_avg['Mask Rate'], marker='o')
                ax.set_title('Daily Average Mask Rate')
                ax.set_xlabel('Date')
                ax.set_ylabel('Mask Rate')
                ax.grid(True)
                
                # Format x-axis dates
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                plt.xticks(rotation=45)
                
                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                
                st.pyplot(fig)
            else:
                st.info("No data for this month.")
            
            # Calendar View (Table)
            st.subheader("Daily Statistics")
            daily_stats = df.groupby('Date').agg({
                'Total': 'sum',
                'With Mask': 'sum'
            }).reset_index()
            
            # Calculate weighted rate
            daily_stats['Mask Rate'] = daily_stats.apply(
                lambda x: x['With Mask'] / x['Total'] if x['Total'] > 0 else 0, axis=1
            )
            
            # Format for display
            daily_stats['Mask Rate'] = daily_stats['Mask Rate'].apply(lambda x: f"{x:.1%}")
            daily_stats['Date'] = daily_stats['Date'].dt.date
            
            st.dataframe(daily_stats, use_container_width=True)
            
        else:
            st.info("No data available yet.")

if __name__ == "__main__":
    main()
