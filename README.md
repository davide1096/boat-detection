# Boat tracking

App that performs object detection and tracking, in order to count the number of boats that appear in a video.
The repo has a docker-compose.yml file that allows the app to be run on any machine:

Build the docker image with

```
docker compose build
```

Then run the container with

```
docker compose run boat_counter
```

Some logging will be written to console, and the final annotated video will be saved in ./detection_results/boat_counter.avi


# Detection

The detection phase is done with the pretrained model YoloV8m (available at https://github.com/ultralytics/ultralytics) with no fine tuning.

# Tracking

The tracking phase is done with a custom movement-based algorithm. It keeps track of detected bounding boxes and estimates their velocity, so that when a boat is occluded or not detected for some frames, its future position can be predicted. For simplicity the movement of the boats is assumed to be linear, but it obviously is not, and we should also consider the movement of the camera.
For each frame, all detections are assigned to existing tracks or new tracks by computing the IoU between the bounding boxes of the detections and the estimated future bounding boxes of the tracks; the detections for which no matching tracks are found, are assigned newly created tracks.
Tracks that do not appear for consecutive 150 frames are deleted.


# Performance and future developments

The algorithm runs at roughly 5.5 FPS, which is obviously not enough for a real-life scenario and must be improved in order to be used in production. A few tweaks could be:
- Run the algorithm on GPU
- Use a lighter model (e.g: YOLOv8s)
- Since the video is high FPS, we could sample the frames and only run the algorithm on some of them

As for the accuracy of the detection and the tracking, in the future we could:
- Use a heavier model
- Create and annotate our own boat dataset, which would enable us to fine tune the detection model, and also use embeddings for the tracking, so that it is not only based on movement but also on the looks of each individual boat
- Compile the code and embed it into a smaller device, and possibly also optimize it, in order to test its use in real case scenarios
- Use a more accurate model for the movement of the boats, which takes into accounts its non-linearity and also the movement of the camera, e.g. nonlinear Kalman filter.

