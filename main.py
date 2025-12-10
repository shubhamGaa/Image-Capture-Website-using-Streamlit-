import streamlit as st
import os
import numpy as np
from PIL import Image
from datetime import datetime
import face_recognition

# --- CONFIGURATION ---
DATASET_FOLDER = "dataset"
MAX_PHOTOS_PER_PERSON = 6
REQUIRED_FEATURES = ['chin', 'left_eyebrow', 'right_eyebrow',
                     'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 
                     'top_lip', 'bottom_lip']
MAX_SIDE_OFFSET = 10  # Max nose offset from eye center in pixels

# --- FUNCTIONS ---
def create_folder(name):
    """Creates dataset folder and subfolder for the person."""
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    safe_name = name.replace(" ", "_").strip().lower()
    person_folder = os.path.join(DATASET_FOLDER, safe_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder, safe_name

def save_image(person_name, person_folder, image_data):
    """Saves the captured image to the person's folder."""
    pil_image = Image.open(image_data)
    current_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]
    photo_count = len(current_files) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{person_name}_{photo_count:02d}_{timestamp}.jpg"
    filepath = os.path.join(person_folder, filename)
    pil_image.save(filepath)
    st.session_state['photo_count'] = photo_count
    return filename

def check_side_angle(landmarks):
    """Checks if face is too turned to the side using nose-eye offset."""
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    nose_tip = np.mean(landmarks['nose_tip'], axis=0)
    
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset = nose_tip[0] - eye_center_x
    return abs(nose_offset) <= MAX_SIDE_OFFSET

# --- STREAMLIT UI ---
st.set_page_config(layout="centered", page_title="Dataset Capture")
st.title("👤  Dataset Capture Portal")
st.markdown("---")

# --- Session State Initialization ---
for key in ['person_name', 'photo_count', 'person_safe_name']:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- Step 1: User Input ---
st.subheader("1. Identify Person")
new_name = st.text_input("Enter Full Name (e.g., Shubham Gatthewar)", key="name_input")

if new_name and new_name != st.session_state.get('last_valid_name_input', ''):
    st.session_state['person_name'] = new_name
    st.session_state['photo_count'] = 0
    _, safe_name = create_folder(new_name)
    st.session_state['person_safe_name'] = safe_name
    st.session_state['last_valid_name_input'] = new_name
    st.info(f"Ready to capture **{new_name}**. Targeting {MAX_PHOTOS_PER_PERSON} images.")
elif not new_name:
    st.session_state['person_name'] = ""
    st.session_state['photo_count'] = 0
    st.session_state['person_safe_name'] = ""
    st.warning("Please enter a name to proceed to camera capture.")

st.markdown("---")

# --- Step 2: Webcam Capture ---
st.subheader("2. Webcam Capture")

if st.session_state['person_name']:
    current_count = st.session_state['photo_count']
    status_message = f"Captured: **{current_count}** / {MAX_PHOTOS_PER_PERSON}"

    if current_count >= MAX_PHOTOS_PER_PERSON:
        st.success(f"✅ Capture complete for **{st.session_state['person_name']}**! ({status_message})")
        st.info("Enter a **new name** in step 1 to start capturing for another person.")
        st.camera_input(status_message, key="camera_disabled", disabled=True)
    else:
        st.markdown(f"**Status:** {status_message}")
        captured_image = st.camera_input("Click 'Take Photo' to capture.", key="camera_enabled")

        if captured_image is not None:
            pil_image = Image.open(captured_image)
            rgb_image = pil_image.convert('RGB')
            rgb_array = np.array(rgb_image)

            # --- FACE LANDMARK CHECK ---
            landmarks_list = face_recognition.face_landmarks(rgb_array)

            if len(landmarks_list) != 1:
                st.warning("⚠️ No face or multiple faces detected. Please try again.")
            else:
                landmarks = landmarks_list[0]
                missing_features = [f for f in REQUIRED_FEATURES if f not in landmarks or len(landmarks[f]) == 0]

                if missing_features:
                    st.warning(f"⚠️ Some facial landmarks are missing: {missing_features}. Retake the photo.")
                elif not check_side_angle(landmarks):
                    st.warning("⚠️ Face is turned too much to the side. Please face forward.")
                else:
                    # Draw landmarks for preview
                    preview_array = np.array(rgb_image)
                    for feature in REQUIRED_FEATURES:
                        for (x, y) in landmarks[feature]:
                            preview_array[y, x] = [0, 255, 0]  # RGB green
                    preview_image = Image.fromarray(preview_array)
                    st.image(preview_image, caption="✅ Face landmarks detected", use_column_width=True)

                    # Save image
                    person_folder = os.path.join(DATASET_FOLDER, st.session_state['person_safe_name'])
                    safe_name = st.session_state['person_safe_name']
                    try:
                        filename = save_image(safe_name, person_folder, captured_image)
                        st.toast(f"Saved: {filename}", icon="📸")
                    except Exception as e:
                        st.error(f"Error saving image: {e}")

else:
    st.markdown("---")
    st.error("Please enter a full name in Step 1 to activate the webcam.")

# --- Sidebar: Status ---
st.sidebar.header("Data Directory Status")
if st.session_state['person_name']:
    st.sidebar.markdown(f"**Target Person:** {st.session_state['person_name']}")
    st.sidebar.markdown(f"**Images Saved:** {st.session_state['photo_count']} / {MAX_PHOTOS_PER_PERSON}")
    st.sidebar.markdown(f"**Save Location:** `{DATASET_FOLDER}/{st.session_state['person_safe_name']}`")

    folder_path = os.path.join(DATASET_FOLDER, st.session_state['person_safe_name'])
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        if files:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Saved Files:**")
            for file in sorted(files, reverse=True):
                st.sidebar.code(file)
else:
    st.sidebar.info("Enter a name to begin tracking status.")
