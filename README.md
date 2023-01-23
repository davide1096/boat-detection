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

The tracking phase is done with a custom movement-based algorithm. It keeps track of detected bounding boxes and estimates their velocity (assumed to be linear), so that when a boat is occluded or not detected for some frames, its future position can be predicted. For each frame, all detections are assigned to existing tracks or new tracks by computing the IoU between the bounding boxes of the detections and the estimated future bounding boxes of the tracks; the detections for which no matching tracks are found, are assigned newly created tracks.
Tracks that do not appear for consecutive 150 frames are deleted.


# Performance

The algorithm runs at roughly 5.5 FPS, which is obviously not enough for a real-life scenario and must be improved in order to be used in production. A few tweaks could be:
- Run the algorithm on GPU
- Use a lighter model (e.g: YOLOv8s)
- Since the video is high FPS, we could sample the frames and only run the algorithm on some of them