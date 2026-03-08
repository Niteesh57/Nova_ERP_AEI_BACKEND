"""
face_service.py ─ Extract the best face crop from a video file using OpenCV
                   Haar Cascade (frontal face detection).

Returns the face as JPEG bytes, or None if no face is found.
"""
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_faces_from_video(video_path: str) -> list[bytes]:
    """
    Sample frames from the video, detect faces using Haar cascade,
    and return all face crops as JPEG bytes from the frame
    that contains the most faces (or largest if tie).

    Args:
        video_path: Path to the local video file (.webm or .mp4).

    Returns:
        List of JPEG bytes of the face crops, or empty list if no face found.
    """
    try:
        import cv2
    except ImportError:
        logger.error("[Face] opencv-python-headless is not installed. Run: pip install opencv-python-headless")
        return []

    try:
        # Load Haar Cascade for frontal face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Face] ❌ Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample one frame every 2 seconds
        sample_every = max(1, int(fps * 2))

        best_faces_bytes: list[bytes] = []
        best_faces_count: int = 0
        best_faces_area_sum: int = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60),
                )

                face_count = len(faces)
                if face_count > 0:
                    area_sum = sum(w * h for (_, _, w, h) in faces)
                    
                    if face_count > best_faces_count or (face_count == best_faces_count and area_sum > best_faces_area_sum):
                        current_frame_faces = []
                        for (x, y, w, h) in faces:
                            # Crop with small padding
                            pad = int(min(w, h) * 0.15)
                            x1 = max(0, x - pad)
                            y1 = max(0, y - pad)
                            x2 = min(frame.shape[1], x + w + pad)
                            y2 = min(frame.shape[0], y + h + pad)
                            face_crop = frame[y1:y2, x1:x2]
                            # Encode as JPEG
                            success, buf = cv2.imencode(".jpg", face_crop)
                            if success:
                                current_frame_faces.append(buf.tobytes())
                        
                        if current_frame_faces:
                            best_faces_count = face_count
                            best_faces_area_sum = area_sum
                            best_faces_bytes = current_frame_faces

            frame_idx += 1

        cap.release()

        if best_faces_bytes:
            print(f"[Face] ✅ Found best frame with {len(best_faces_bytes)} faces")
        else:
            print(f"[Face] ℹ️ No face detected in {total_frames} frames")

        return best_faces_bytes

    except Exception as e:
        print(f"[Face] ❌ Error during face extraction: {e}")
        logger.error(f"[Face] Error: {e}", exc_info=True)
        return []
