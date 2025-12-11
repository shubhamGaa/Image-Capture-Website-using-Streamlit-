import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
from datetime import datetime
import os

# -----------------------------------------
# CONFIG
# -----------------------------------------
DATASET_FOLDER = "dataset"
MAX_PHOTOS_PER_PERSON = 6
MAX_SIDE_OFFSET = 0.50     # Threshold for left/right head turn

mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

def create_folder(name):
    """Creates dataset folder for each person."""
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    safe_name = name.replace(" ", "_").strip().lower()
    person_folder = os.path.join(DATASET_FOLDER, safe_name)

    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    return person_folder, safe_name


def save_image(person_name, person_folder, image_data):
    """Save PIL image to dataset."""
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
    """
    Returns True if face is looking forward.
    Rejects images where face is too side turned (>25% offset)
    """
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_TIP = 1

    lx = landmarks[LEFT_EYE].x
    rx = landmarks[RIGHT_EYE].x
    nx = landmarks[NOSE_TIP].x

    eye_center = (lx + rx) / 2
    offset_ratio = abs(nx - eye_center) / abs(lx - rx)

    # Threshold: 0.25 = 25% shift → ~40° yaw
    return offset_ratio <= 0.35



def draw_landmarks(image, landmarks):
    """Draw Mediapipe face landmarks."""
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        image[y, x] = [0, 255, 0]
    return image


# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------

st.set_page_config(page_title="Dataset Capture with Mediapipe")

st.title("👤 Dataset Capture Portal (Mediapipe Version)")
st.markdown("---")

# Session state setup
for key in ['person_name', 'photo_count', 'person_safe_name']:
    if key not in st.session_state:
        st.session_state[key] = ""

# -----------------------------------------
# STEP 1: Enter Person Name
# -----------------------------------------

st.subheader("1. Identify Person")
new_name = st.text_input("Enter Full Name")

if new_name and new_name != st.session_state.get('last_name', ''):
    st.session_state['person_name'] = new_name
    st.session_state['photo_count'] = 0
    _, safe_name = create_folder(new_name)
    st.session_state['person_safe_name'] = safe_name
    st.session_state['last_name'] = new_name
    st.info(f"Ready to capture **{new_name}**.")
elif not new_name:
    st.warning("Please enter a name to continue.")

st.markdown("---")

# -----------------------------------------
# STEP 2: Webcam Capture
# -----------------------------------------

if st.session_state['person_name']:
    count = st.session_state['photo_count']
    status = f"Captured: {count} / {MAX_PHOTOS_PER_PERSON}"

    if count >= MAX_PHOTOS_PER_PERSON:
        st.success("🎉 Capture completed!")
        st.camera_input(status, disabled=True)

    else:
        st.write(f"**Status:** {status}")
        captured_img = st.camera_input("Capture Image")

        if captured_img is not None:
            pil = Image.open(captured_img)
            img_rgb = np.array(pil)

            # Mediapipe processing
            results = FACE_MESH.process(img_rgb)

            if not results.multi_face_landmarks:
                st.warning("⚠️ No face detected. Try again.")
            else:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Check side pose
                if not check_side_angle(face_landmarks):
                    st.warning("⚠️ Face is turned too much sideways. Look straight.")
                else:
                    # Draw landmarks preview
                    preview = img_rgb.copy()
                    preview = draw_landmarks(preview, face_landmarks)
                    st.image(preview, caption="Face Landmarks OK")

                    # Save
                    folder = os.path.join(DATASET_FOLDER, st.session_state['person_safe_name'])
                    filename = save_image(st.session_state['person_safe_name'], folder, captured_img)
                    st.toast(f"Saved: {filename}", icon="📸")

else:
    st.error("Enter a name first.")

# -----------------------------------------
# SIDEBAR STATUS
# -----------------------------------------

st.sidebar.header("Dataset Status")

if st.session_state['person_name']:
    st.sidebar.write("Person:", st.session_state['person_name'])
    st.sidebar.write("Saved:", st.session_state['photo_count'])
    st.sidebar.write("Folder:", f"{DATASET_FOLDER}/{st.session_state['person_safe_name']}")

    folder = os.path.join(DATASET_FOLDER, st.session_state['person_safe_name'])
    if os.path.exists(folder):
        files = sorted(os.listdir(folder))
        st.sidebar.write("Files:")
        for f in files:
            st.sidebar.code(f)
