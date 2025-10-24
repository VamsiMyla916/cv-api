import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
# --- NEW WEBCAM IMPORTS ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av  # Helper library for video frames
# --------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Occupancy Counter",
    page_icon="🧑‍🤝‍🧑",
    layout="wide"
)

# --- API URL ---
API_URL = "http://127.0.0.1:8000/detect/"

# --- Page Title ---
st.title("🧑‍🤝‍🧑 Real-Time Occupancy Counter")
st.caption("This web app calls a containerized FastAPI-YOLOv8 backend to count people.")

# --- Function to draw boxes (we'll use this twice) ---
def draw_boxes(image_cv, api_data):
    """Draws bounding boxes from the API data onto the image."""
    annotated_image = image_cv.copy()
    
    for obj in api_data.get("detected_objects", []):
        bbox = obj['bbox']
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        confidence = obj['confidence']
        
        # Draw the rectangle (green)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"Person: {confidence:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image

# --- NEW: Webcam Processing Function ---
# This function will process each frame from the webcam
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the frame to an OpenCV-style BGR numpy array
    image_cv = frame.to_ndarray(format="bgr24")
    
    # 1. Convert BGR image to JPEG bytes
    _, image_bytes = cv2.imencode(".jpg", image_cv)
    
    # 2. Prepare for API
    files = {"file": ("frame.jpg", image_bytes.tobytes(), "image/jpeg")}

    try:
        # 3. Send request to API
        response = requests.post(API_URL, files=files, timeout=1.0) # 1-second timeout
        
        if response.status_code == 200:
            data = response.json()
            
            # 4. Draw boxes on the original frame
            annotated_image = draw_boxes(image_cv, data)
            
            # 5. Convert annotated BGR image back to a VideoFrame
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
        
        else:
            # If API fails, just return the original frame
            return frame

    except requests.exceptions.RequestException:
        # On timeout or connection error, return original frame
        return frame

# --- Use Tabs for Layout ---
tab1, tab2 = st.tabs(["Upload an Image", "Live Webcam Feed"])

# --- Tab 1: File Uploader ---
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image to count people", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image_pil = Image.open(uploaded_file)
        image_cv = np.array(image_pil.convert('RGB'))
        # Convert RGB (from PIL) to BGR (for OpenCV drawing)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        with col1:
            st.subheader("Original Image")
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Counting people... 🤖"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    person_count = data.get("person_count", 0)
                    annotated_image = draw_boxes(image_cv, data)
                    
                    # Convert BGR (from OpenCV) back to RGB (for Streamlit)
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_image_rgb, caption="Annotated Image", use_column_width=True)
                        st.metric(label="Persons Detected", value=person_count)
                        with st.container(border=True):
                            st.success("Detection Complete!")
                            with st.expander("Show API JSON Response"):
                                st.json(data)
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API.")
                st.warning("Please make sure your Docker container is running!")

# --- Tab 2: Live Webcam Feed ---
with tab2:
    st.subheader("Real-Time Person Counter")
    st.write("Click 'START' to turn on your webcam. The app will send each frame to the API.")

    webrtc_streamer(
        key="person-counter-webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,  # <-- WE PASS THE FUNCTION HERE
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.write("---")
    st.markdown("### How it works:")
    st.markdown(
        """
        1.  Your browser sends your webcam video to the Streamlit server.
        2.  For *every frame*, the server calls our `video_frame_callback` function.
        3.  The function sends the frame to our Dockerized FastAPI API.
        4.  The API detects people and returns the bounding boxes.
        5.  The function draws the boxes on the frame.
        6.  The annotated video is streamed back to you.
        """
    )