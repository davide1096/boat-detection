import cv2 as cv
import numpy as np
import torch
import argparse
import time
import logging
from pathlib import Path
from ultralytics import YOLO
from tracker import Tracker

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


def main(video_path, model_name):
    model = YOLO(f"{model_name}.pt")
    cap = cv.VideoCapture(video_path)
    result_dir = Path("detection_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    frame_id = 0
    tracker = Tracker()
    start_time = time.time()
    boat_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == 0:
            h, w = frame.shape[:2]
            out = cv.VideoWriter(str(result_dir.joinpath(
                "boat_counter.avi")), cv.VideoWriter_fourcc(*"MJPG"), 30, (w, h))

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        dets = model(img)[0]
        boxes = dets.boxes.boxes.tolist()
        boat_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                      for b in boxes if classes[int(b[-1])] == "boat"]

        # Remove our boat by considering only bounding boxes that occupy maximum 20% of the frame
        boat_boxes = [b for b in boat_boxes if (
            b[2]-b[0])*(b[3]-b[1]) <= (h*w)/5]

        assignments = tracker.solve_assignment(boat_boxes, frame_id)
        final_img = cv.putText(frame, f"Boat counter: {tracker.boat_counter-1}",
                               (h+100, w-200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for obj, track in assignments:
            final_img = cv.rectangle(
                final_img, (obj[0], obj[1]), (obj[2], obj[3]), track.color, 2)
            final_img = cv.putText(final_img, f"Boat {track.boat_num}", (
                obj[0], obj[1]), cv.FONT_HERSHEY_SIMPLEX, 1, track.color, 2)

        out.write(final_img)
        frame_id += 1

    out.release()
    cap.release()

    elapsed_time = time.time() - start_time

    logging.info(
        f"Elaborated {frame_id} frames, counting {tracker.boat_counter-1} boats, in {elapsed_time:0.2f} seconds, {(frame_id/elapsed_time):0.2f} FPS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that counts boats in a video.")
    parser.add_argument("--video_file", type=str, help="Path to the video.",
                        default="Test-Task Sequence from Wörthersee.mp4")
    parser.add_argument("--model", type=str,
                        help="Object detection model name.", default="yolov8m")

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(video_path=args.video_file, model_name=args.model)
