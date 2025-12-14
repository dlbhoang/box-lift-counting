import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# ======================
# STREAMLIT CONFIG
# ======================
st.set_page_config(page_title="Box Lift Counting", layout="wide")
st.title("📦 Box Lift Detection & Counting")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return YOLO("0809.pt")

model = load_model()

# ======================
# SIDEBAR SETTINGS
# ======================
st.sidebar.header("⚙️ Parameters")

MIN_CONSECUTIVE_UPWARD = st.sidebar.slider("Min consecutive upward frames", 1, 10, 4)
UPWARD_THRESHOLD = st.sidebar.slider("Upward threshold (px)", 0.5, 5.0, 1.2)
DOWNWARD_THRESHOLD = st.sidebar.slider("Downward reset threshold (px)", 0.5, 5.0, 0.8)
MIN_FRAMES_IN_ROI = st.sidebar.slider("Min frames in ROI", 1, 10, 3)
MIN_TOTAL_DISPLACEMENT = st.sidebar.slider("Min total upward displacement (px)", 1.0, 20.0, 5.0)
MAX_BOX_SIZE = st.sidebar.slider("Max box area", 20000, 200000, 60000)

# ======================
# ROI
# ======================
TRUCK_ROI = np.array([
    (320, 230),
    (880, 230),
    (950, 580),
    (300, 580)
], dtype=np.int32)

# ======================
# VIDEO UPLOAD
# ======================
uploaded_file = st.file_uploader("📤 Upload video", type=["mp4", "avi", "mov"])
start_btn = st.button("▶️ Start Processing")

# ======================
# FUNCTIONS
# ======================
def point_in_roi(point, roi):
    return cv2.pointPolygonTest(roi, point, False) >= 0

# ======================
# MAIN
# ======================
if uploaded_file and start_btn:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "output_streamlit.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    counted_ids = set()
    track_state = {}
    total_boxes = 0
    frame_count = 0

    video_placeholder = st.empty()
    stats_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml"
        )

        cv2.polylines(frame, [TRUCK_ROI], True, (255, 0, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if track_id not in track_state:
                    track_state[track_id] = {
                        "last_y": cy,
                        "frames_in_roi": 0,
                        "consecutive_upward": 0,
                        "total_upward": 0,
                        "counted": False,
                    }

                state = track_state[track_id]
                is_in_roi = point_in_roi((cx, cy), TRUCK_ROI)
                dy = state["last_y"] - cy

                if not state["counted"] and is_in_roi:
                    state["frames_in_roi"] += 1
                    area = (x2 - x1) * (y2 - y1)

                    if area <= MAX_BOX_SIZE and state["frames_in_roi"] >= MIN_FRAMES_IN_ROI:
                        if dy > UPWARD_THRESHOLD:
                            state["consecutive_upward"] += 1
                            state["total_upward"] += dy
                        elif dy < -DOWNWARD_THRESHOLD:
                            state["consecutive_upward"] = 0
                            state["total_upward"] = 0

                        if (
                            state["consecutive_upward"] >= MIN_CONSECUTIVE_UPWARD
                            and state["total_upward"] >= MIN_TOTAL_DISPLACEMENT
                        ):
                            total_boxes += 1
                            state["counted"] = True
                            counted_ids.add(track_id)

                if not is_in_roi:
                    state["frames_in_roi"] = 0
                    state["consecutive_upward"] = 0
                    state["total_upward"] = 0

                state["last_y"] = cy

                color = (0, 255, 0) if state["counted"] else (0, 255, 255) if is_in_roi else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"TOTAL BOXES: {total_boxes}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        out.write(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", width=900)

        stats_placeholder.markdown(
            f"""
            ### 📊 Statistics
            - **Frame:** {frame_count}
            - **Total Boxes Counted:** `{total_boxes}`
            """
        )

    cap.release()
    out.release()

    st.success(f"✅ Processing done! TOTAL BOXES = {total_boxes}")

    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download output video", f, file_name="result.mp4")

    os.remove(video_path)
