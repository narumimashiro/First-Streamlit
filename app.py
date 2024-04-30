"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)


MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COLORS = {
    # BGR
    "background": (0,0,0),
    "aeroplane": (0, 255, 255), # cyan
    "bicycle": (255, 128, 0), # orange
    "bird": (128, 0, 0), # thick red
    "boat": (0,0,0),
    "bottle": (0, 255, 255), # yellow
    "bus": (128, 128, 0), # thick yellow
    "car": (128, 0, 128), # thick magenta
    "cat": (192, 192, 192), # gray
    "chair": (255, 0, 0), # blue
    "cow": (0,0,0),
    "diningtable": (0, 255, 0), # green
    "dog": (0, 128, 128), # thick cyan
    "horse": (0,0,0),
    "motorbike": (64, 64, 64), # dark gray
    "person": (0, 0, 255), # red
    "pottedplant": (0, 0, 128), # thick blue
    "sheep": (0,0,0),
    "sofa": (0, 128, 0), # thick green
    "train": (255, 0, 255), # magenta
    "tvmonitor": (255, 255, 255), # white
}


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))


# COLORS = generate_label_colors()

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)


# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    camera_frame = frame.to_ndarray(format="bgr24")

    # Run inference
    blob = cv2.dnn.blobFromImage(
        cv2.resize(camera_frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    output = net.forward()

    h, w = camera_frame.shape[:2]

    # Convert the output array into a structured form.
    output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
    output = output[output[:, 2] >= 0.5]
    detections = [
        Detection(
            class_id=int(detection[1]),
            label=CLASSES[int(detection[1])],
            score=float(detection[2]),
            box=(detection[3:7] * np.array([w, h, w, h])),
        )
        for detection in output
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.label]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

        cv2.rectangle(camera_frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            camera_frame,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(camera_frame, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)