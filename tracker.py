from __future__ import annotations
from scipy.optimize import linear_sum_assignment
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Bbox():

    def __init__(self, x1: int, y1: int, x2: int, y2: int, frame_id: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame = frame_id
    
    @property
    def area(self) -> int:
        if self.x2 < self.x1 or self.y2 < self.y1:
            return 0.0
        else:
            return (self.x2-self.x1)*(self.y2-self.y1)

    def intrs(self, other: Bbox) -> Bbox:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        return Bbox(x1, y1, x2, y2, None)

    def IoU(self, other: Bbox) -> float:
        intrs_area = self.intrs(other).area
        union_area = self.area + other.area - intrs_area

        return intrs_area/union_area
    
    @property
    def center(self) -> List[int, int]:
        return [(self.x1 + self.x2)/2, (self.y1+self.y2)/2]

class Track():
    """
    Class for each track, keeping information about boat tracks.
    """
    
    def __init__(self, boat_num: int, bbox: Bbox):
        self.boat_num = boat_num
        self.color = tuple([int(x) for x in np.random.randint(256, size=3)])
        self.positions = [bbox]
        logger.info(f"Created new track with boat number {self.boat_num}, color {self.color}!")


    def avg_speed(self, frame_id: int, window_size: int):
        """
        Function that estimates the speed of the tracked boat as a moving average of the last <window_size> number of frames.
        """

        positions_in_window = [p for p in self.positions if p.frame>=frame_id-window_size]
        if len(positions_in_window) >= 2:
            v = [0.0, 0.0]
            for i in range(1, len(positions_in_window)):
                pos1 = self.positions[i-1]
                pos2 = self.positions[i]
                # Get speed between two detections
                v_x = (pos2.center[0] - pos1.center[0]) / (pos2.frame - pos1.frame)
                v_y = (pos2.center[1] - pos1.center[1]) / (pos2.frame - pos1.frame)
                v = [v[0] + v_x, v[1] + v_y]
            return [x/(len(positions_in_window)-1) for x in v]
        else:
            return [0.0, 0.0]

    def predicted_bbox(self, frame_id, window_size):
        speed = self.avg_speed(frame_id, window_size)
        last_pos = self.positions[-1]
        frame_dif = frame_id - last_pos.frame
        dist = [x*frame_dif for x in speed]
        return Bbox(last_pos.x1 + dist[0], last_pos.y1 + dist[1], last_pos.x2 + dist[0], last_pos.y2 + dist[1], None)

    def update(self, det_point, frame_id):
        self.positions += [Bbox(*det_point, frame_id)]
        logger.info(f"Updated track of boat {self.boat_num}!")
        return self


class Tracker():
    """
    Class that keeps track of the various tracks during one video.
    """
    def __init__(self, window_size=120, not_seen_thresh=150):
        self.tracks = []
        self.boat_counter = 0
        self.not_seen_thresh = not_seen_thresh
        self.window_size = window_size

    def add_track(self, position, frame_id):
        """
        Function that adds a new track to the Tracker object.
        """

        self.tracks.append(Track(self.boat_counter, Bbox(*position, frame_id)))
        self.boat_counter += 1
        return self.tracks[-1]

    def solve_assignment(self, detected_objs, frame_id):
        """
        Function that solves the assignment problem by using the Hungarian algorithm with IoU as cost.
        """

        # Create cost matrix with IoU
        cost_matrix = np.zeros(shape=(len(detected_objs), len(self.tracks)))
        for i, obj in enumerate(detected_objs):
            bbox = Bbox(*obj, frame_id)
            for j, track in enumerate(self.tracks):
                cost_matrix[i][j] = bbox.IoU(track.predicted_bbox(frame_id, self.window_size))

        # Solve the assignment problem with the Hungarian algorithm borrowing it from scipy :)
        assignments = linear_sum_assignment(cost_matrix, maximize=True)
        obj_track_zip = []
        tracks_seen = []
        for i,obj in enumerate(detected_objs):
            assigned_track = None
            if i not in assignments[0]:
                # No assignment was found for i-th detected object, so a new track gets added.
                assigned_track = self.add_track(obj, frame_id)
                tracks_seen += [len(self.tracks) - 1]
            else:
                # Find the index of the assigned track.
                idx = np.where(assignments[0]==i)[0][0]
                track_idx = assignments[1][idx]
                cost = cost_matrix[i, track_idx]
                if cost == 0:
                    # If the IoU between the assigned track and the detection is 0, disregard the track and create a new one                       
                    assigned_track = self.add_track(obj, frame_id)
                    tracks_seen += [len(self.tracks) - 1]
                else:
                    assigned_track = self.tracks[track_idx].update(obj, frame_id)
                    tracks_seen += [track_idx]
        
            obj_track_zip.append([obj, assigned_track])
      
        for i, track in enumerate(self.tracks):
            # Delete tracks that have not been seen for a number of frames >= threshold
            if i not in tracks_seen and frame_id-track.positions[-1].frame >= self.not_seen_thresh:
                self.tracks.pop(i)
        return obj_track_zip