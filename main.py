import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
from datetime import datetime
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import io

# -------------------------- CONFIG --------------------------
DATASET_FOLDER = "dataset"
MAX_PHOTOS_PER_PERSON = 6
MAX_SIDE_OFFSET = 0.50     # Threshold for left/right head turn

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------------- GOOGLE DRIVE AUTH --------------------------
def authenticate_drive():
    gauth = GoogleAuth()
    
    # Setup credentials manually using st.secrets
    gauth.DEFAULT_SETTINGS = {
        "client_config_backend": "settings",
        "client_config": {
            "client_id": st.secrets["gdrive_oauth"]["client_id"],
            "client_secret": st.secrets["auth_client"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        },
        "save_credentials": True,
        "save_credentials_backend": "file",
        "save_credentials_file": "credentials.json"
    }
    
    gauth.LocalWebserverAuth()  # opens OAuth link for user in browser
    drive = GoogleDrive(gauth)
    return drive

# -------------------------- HELPER FUNCTIONS --------------------------
def create_folder(name):
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
    safe_name = name.replace(" ", "_").strip().lower()
    person_folder = os.path.join(DATASET_FOLDER, safe_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder, safe_name

def save_image_locally(person_name, person_folder, image_data):
    pil_image = Image.open(image_data)
    current_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]
    photo_count = len(current_files) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{person_name}_{photo_count:02d}_{timestamp}.jpg"
    filepath = os.path.join(person_folder, filename)
    pil_image.save(filepath)
    st.session_state['photo_count'] = photo_count
    return filename, pil_image

def upload_to_drive(pil_image, filename, drive):
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    file_drive = drive.CreateFile({'title': filename})
    file_drive.SetContentFile(img_bytes)
    file_drive.Upload()
    return file_drive['id']

def check_side_angle(landmarks):
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_TIP = 1
    lx = landmarks[LEFT_EYE].x
    rx = landmarks[RIGHT_EYE].x
    nx = landmarks[NOSE_TIP].x
    eye_center = (lx + rx) / 2
    offset_ratio = abs(nx - eye_center) / abs(lx - rx)
    return offset_ratio <= MAX_SIDE_OFFSET

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        image[y, x] = [0, 255, 0]
    return image

# -------------------------- STREAMLIT UI --------------------------
st.set_page_config(page_title="Dataset Capture + Google Drive")
st.title("👤 Dataset Capture Portal (Mediapipe + OAuth Drive)")
st.markdown("---")

# Session state
for key in ['person_name', 'photo_count', 'person_safe_name']:
    if key not in st.session_state:
        st.session_state[key] = ""

# STEP 1: Enter Name
st.subheader("1. Identify Person")
new_name = st.text_input("Enter Full Name")

if new_name and new_name != st.session_state.get('last_name', ''):
    st.session_state['person_name'] = new_name
    st.session_state['photo_count'] = 0
    _, safe_name = create_folder(new_name)
    st.session_state['person_safe_name'] = safe_name
    st.session_state['last_name'] = new_name
    st.info(f"Ready to capture **{new_name}**.")

# STEP 2: Webcam Capture
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

            # Mediapipe detection
            results = FACE_MESH.process(img_rgb)
            if not results.multi_face_landmarks:
                st.warning("⚠️ No face detected. Try again.")
            else:
                face_landmarks = results.multi_face_landmarks[0].landmark

                if not check_side_angle(face_landmarks):
                    st.warning("⚠️ Face is turned too much sideways. Look straight.")
                else:
                    # Draw landmarks preview
                    preview = img_rgb.copy()
                    preview = draw_landmarks(preview, face_landmarks)
                    st.image(preview, caption="Face Landmarks OK")

                    # Save locally
                    folder = os.path.join(DATASET_FOLDER, st.session_state['person_safe_name'])
                    filename, pil_img = save_image_locally(st.session_state['person_safe_name'], folder, captured_img)

                    # Upload to Google Drive
                    try:
                        drive = authenticate_drive()
                        file_id = upload_to_drive(pil_img, filename, drive)
                        st.success(f"✅ Saved locally and uploaded! File ID: {file_id}")
                    except Exception as e:
                        st.error(f"Error uploading to Google Drive: {e}")
else:
    st.error("Enter a name first.")

# SIDEBAR STATUS
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
