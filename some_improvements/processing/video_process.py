# from get_tracks import TrackingProcess
# from post_process import PostProcess
from processing import TrackingProcess, PostProcess
from multiprocessing import Value
import os
from support_functions import RepeatedTimer
from threading import Thread
import time


class VideoProcess:

    def __init__(self, file_name, file_path, video_id, stop_event, db_path):

        self.tracking = None
        self.post_process = None

        self.stop_event = stop_event

        self.name = file_name
        self.args_tracking = dict()
        self.args_tracking["video"] = file_path
        self.args_tracking["size"] = None
        self.args_tracking["name"] = file_name
        self.args_tracking["out_vectors"] = f"./data/temp/{file_name}_out_vectors.txt"
        self.args_tracking["out_tracks"] = f"./data/temp/{file_name}_out_tracks.txt"
        self.args_tracking["out_clothes"] = f"./data/temp/{file_name}_out_clothes.txt"
        self.args_tracking["extractor"] = "fast_reid"
        self.args_tracking["detector"] = "yolo"

        self.args_post = dict()
        self.args_post['video_id'] = video_id
        self.args_post['video'] = self.args_tracking["video"]
        self.args_post["db_file"] = db_path
        self.args_post['in_tracks'] = self.args_tracking["out_tracks"]
        self.args_post['in_vectors'] = self.args_tracking["out_vectors"]
        self.args_post['in_clothes'] = self.args_tracking["out_clothes"]

        self._percent = Value('i', 0)

    def run(self):
        print(self.name, "start")

        self.tracking = TrackingProcess(self.args_tracking, self.stop_event)
        self.post_process = PostProcess(self.args_post, self.stop_event)

        timer = RepeatedTimer(1, self._update_percent, self.tracking, ratio=0.95, start=0)
        timer.start()
        self.tracking.run()
        timer.stop()
        self._update_percent(self.tracking, ratio=0.95, start=0)
        print(self.name, "tracking done")

        if self.stop_event.is_set():
            return

        timer = RepeatedTimer(0.1, self._update_percent, self.post_process, ratio=0.05, start=95)
        timer.start()
        self.post_process.run()
        timer.stop()
        self._update_percent(self.post_process, ratio=0.05, start=95)
        print(self.name, "postprocessing done")

    def _update_percent(self, obj, ratio=1.0, start=0):
        self._percent.value = start + int(obj.percent * ratio)

    def get_percent(self):
        return self._percent.value
