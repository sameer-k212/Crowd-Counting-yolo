import csv
import io
import os
import tempfile
import time
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
from deepface import DeepFace

import config
import database
from logger import logger

app = FastAPI(title="Crowd Counting API", version="2.0.0")

database.init_db()
model = YOLO(config.MODEL_PATH)
logger.info("YOLOv8 model loaded: %s", config.MODEL_PATH)

OBJECT_CLASSES = {
    0: ("Person",     (0, 255, 0)),
    2: ("Car",        (0, 165, 255)),
    3: ("Motorcycle", (255, 0, 255)),
    5: ("Bus",        (255, 255, 0)),
    7: ("Truck",      (0, 128, 255)),
}


def get_density_level(count: int) -> str:
    if count == 0:    return "Empty"
    elif count <= 5:  return "Low"
    elif count <= 20: return "Medium"
    elif count <= 50: return "High"
    else:             return "Very High"


def detect_gender(img, x1, y1, x2, y2) -> str:
    try:
        h, w = img.shape[:2]
        box_h, box_w = y2 - y1, x2 - x1
        fy1 = max(0, y1)
        fy2 = min(h, y1 + int(box_h * 0.28))
        fx1 = max(0, x1 + int(box_w * 0.1))
        fx2 = min(w, x2 - int(box_w * 0.1))
        face_crop = img[fy1:fy2, fx1:fx2]
        if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            return "Unknown"
        result = DeepFace.analyze(
            face_crop, actions=["gender"],
            enforce_detection=True, detector_backend="opencv", silent=True
        )
        scores = result[0]["gender"]
        man_score, woman_score = scores["Man"], scores["Woman"]
        if man_score > 72 and man_score > woman_score + 20:
            return "Male"
        elif woman_score > 72 and woman_score > man_score + 20:
            return "Female"
        return "Unknown"
    except Exception:
        return "Unknown"


def generate_heatmap(img, boxes):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        heat[max(0, cy - bh):min(h, cy + bh), max(0, cx - bw):min(w, cx + bw)] += 1
    if heat.max() > 0:
        heat /= heat.max()
    ksize = max(51, min(h, w) // 10)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    heat = cv2.GaussianBlur(heat, (ksize, ksize), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    for i, (label, color) in enumerate([
        ("Low", (0, 255, 0)), ("Medium", (0, 255, 255)),
        ("High", (0, 128, 255)), ("Critical", (0, 0, 255))
    ]):
        cv2.rectangle(overlay, (10, 10 + i * 22), (30, 28 + i * 22), color, -1)
        cv2.putText(overlay, label, (35, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def process_frame_fast(img, alert_threshold: int = config.DEFAULT_ALERT_THRESHOLD):
    """YOLO-only, no DeepFace — used for live webcam for max speed."""
    start = time.time()
    results = model(img, verbose=False, conf=0.45)
    processing_time = round((time.time() - start) * 1000)

    person_count = 0
    total_confidence = 0.0
    object_counts = {}
    annotated = img.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in OBJECT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_name, color = OBJECT_CLASSES[cls]
            if cls == 0:
                person_count += 1
                total_confidence += conf
                label = f"#{person_count} {conf:.0%}"
            else:
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                label = f"{obj_name} {conf:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if person_count > alert_threshold:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(annotated, f"CROWD ALERT: {person_count} people detected!",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    avg_confidence = round((total_confidence / person_count) * 100) if person_count > 0 else 0
    return annotated, person_count, object_counts, avg_confidence, processing_time


def process_frame(img, alert_threshold: int = config.DEFAULT_ALERT_THRESHOLD):
    """Full processing with DeepFace gender detection — used for image/video uploads."""
    start = time.time()
    results = model(img, verbose=False, conf=0.45)
    processing_time = round((time.time() - start) * 1000)

    person_count = male_count = female_count = 0
    total_confidence = 0.0
    person_boxes = []
    object_counts = {}
    annotated = img.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in OBJECT_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_name, color = OBJECT_CLASSES[cls]

            if cls == 0:
                person_count += 1
                total_confidence += conf
                person_boxes.append((x1, y1, x2, y2))
                gender = detect_gender(img, x1, y1, x2, y2)
                if gender == "Male":
                    male_count += 1
                    color = (255, 100, 0)
                elif gender == "Female":
                    female_count += 1
                    color = (0, 100, 255)
                label = f"#{person_count} {gender} {conf:.0%}"
            else:
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                label = f"{obj_name} {conf:.0%}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if person_count > alert_threshold:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(annotated, f"CROWD ALERT: {person_count} people detected!",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    heatmap = generate_heatmap(img, person_boxes)
    avg_confidence = round((total_confidence / person_count) * 100) if person_count > 0 else 0

    return annotated, heatmap, person_count, male_count, female_count, object_counts, avg_confidence, processing_time


def encode_image(img) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": config.MODEL_PATH}


# ── Image ─────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    threshold: int = Query(config.DEFAULT_ALERT_THRESHOLD)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    logger.info("Image upload: %s (threshold=%d)", file.filename, threshold)

    try:
        ann, heatmap, person_count, male_count, female_count, obj_counts, avg_conf, proc_time = \
            process_frame(img, threshold)
    except Exception as e:
        logger.error("process_frame failed: %s", e)
        raise HTTPException(500, str(e))

    alert = person_count > threshold
    density = get_density_level(person_count)

    database.save_analysis({
        "type": "image",
        "filename": file.filename,
        "person_count": person_count,
        "male_count": male_count,
        "female_count": female_count,
        "car_count": obj_counts.get("Car", 0),
        "object_counts": obj_counts,
        "avg_confidence": avg_conf,
        "density_level": density,
        "processing_time_ms": proc_time,
        "alert": alert,
        "alert_threshold": threshold,
    })

    logger.info("Image result: %d people, density=%s, alert=%s", person_count, density, alert)

    return JSONResponse({
        "person_count": person_count,
        "male_count": male_count,
        "female_count": female_count,
        "object_counts": obj_counts,
        "avg_confidence": avg_conf,
        "density_level": density,
        "processing_time_ms": proc_time,
        "alert": alert,
        "alert_threshold": threshold,
        "output_image": encode_image(ann),
        "heatmap_image": encode_image(heatmap),
    })


# ── Video ─────────────────────────────────────────────────────────────────────

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    threshold: int = Query(config.DEFAULT_ALERT_THRESHOLD)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    logger.info("Video upload: %s", file.filename)

    # Safe temp file — no race condition
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        sample_interval = max(1, int(fps * 3))

        frame_results = []
        frame_idx = 0
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            if frame_idx % sample_interval == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640 / w)))
                ann_frame, _, count, males, females, obj_counts, conf, proc_time = \
                    process_frame(frame, threshold)
                frame_results.append({
                    "second": round(frame_idx / fps, 1),
                    "count": count,
                    "males": males,
                    "females": females,
                    "objects": obj_counts,
                    "confidence": conf,
                    "processing_time_ms": proc_time,
                    "frame_image": encode_image(ann_frame),
                })
            frame_idx += 1

        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    max_count = max((r["count"] for r in frame_results), default=0)
    max_persons = max((r["count"] for r in frame_results), default=0)
    max_cars = max((r["objects"].get("Car", 0) for r in frame_results), default=0)
    avg_count = round(sum(r["count"] for r in frame_results) / len(frame_results)) if frame_results else 0
    density = get_density_level(max_count)
    alert = max_count > threshold

    preview = heatmap_preview = ""
    if last_frame is not None:
        ann_last, heatmap_last, _, _, _, _, _, _ = process_frame(last_frame, threshold)
        preview = encode_image(ann_last)
        heatmap_preview = encode_image(heatmap_last)

    analysis_id = database.save_analysis({
        "type": "video",
        "filename": file.filename,
        "person_count": max_persons,
        "car_count": max_cars,
        "object_counts": {},
        "avg_confidence": 0,
        "density_level": density,
        "processing_time_ms": 0,
        "alert": alert,
        "alert_threshold": threshold,
    })
    database.save_video_frames(analysis_id, frame_results)

    logger.info("Video result: max=%d people, alert=%s", max_count, alert)

    return JSONResponse({
        "frame_results": frame_results,
        "max_count": max_count,
        "max_persons": max_persons,
        "max_cars": max_cars,
        "avg_count": avg_count,
        "density_level": density,
        "alert": alert,
        "alert_threshold": threshold,
        "preview_image": preview,
        "heatmap_image": heatmap_preview,
    })


# ── Webcam ────────────────────────────────────────────────────────────────────

@app.post("/webcam-frame")
async def webcam_frame(
    file: UploadFile = File(...),
    threshold: int = Query(config.DEFAULT_ALERT_THRESHOLD)
):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid frame")

    h, w = img.shape[:2]
    if w > 640:
        img = cv2.resize(img, (640, int(h * 640 / w)))

    try:
        ann, person_count, obj_counts, avg_conf, proc_time = process_frame_fast(img, threshold)
    except Exception as e:
        logger.error("Webcam process_frame_fast failed: %s", e)
        raise HTTPException(500, str(e))

    alert = person_count > threshold

    return JSONResponse({
        "person_count": person_count,
        "male_count": 0,
        "female_count": 0,
        "object_counts": obj_counts,
        "avg_confidence": avg_conf,
        "density_level": get_density_level(person_count),
        "processing_time_ms": proc_time,
        "alert": alert,
        "output_image": encode_image(ann),
    })


# ── History & Stats ───────────────────────────────────────────────────────────

@app.get("/history")
def get_history(limit: int = Query(20, ge=1, le=100)):
    return database.get_history(limit)


@app.delete("/history")
def clear_history():
    database.clear_history()
    logger.info("History cleared")
    return {"success": True}


@app.get("/stats")
def get_stats():
    return database.get_stats()


# ── CSV Export ────────────────────────────────────────────────────────────────

@app.get("/export/csv")
def export_csv(limit: int = Query(100, ge=1, le=1000)):
    rows = database.get_history(limit)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "type", "filename", "person_count", "male_count", "female_count",
        "car_count", "avg_confidence", "density_level", "processing_time_ms",
        "alert", "alert_threshold", "created_at"
    ])
    writer.writeheader()
    for row in rows:
        row.pop("object_counts", None)
        writer.writerow(row)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crowd_analysis.csv"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
