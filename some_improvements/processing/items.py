import numpy as np
import cv2
from processing.config import global_config
from processing.support_functions import hex_to_rgb
import os


class FileItem:
    def __init__(self, name, path, id_=None, processed=None):
        """
        path - absolute path to file
        """
        # TODO добавить проверку что файл существует и его можно открыть
        self.name = str(name)
        self.path = str(path)
        # self._rel_path = os.path.relpath(self._path, global_config["application"].get("video_folder_path"))
        self.file_id = int(id_) if id_ is not None else None
        self.processed = bool(processed) if processed is not None else False
        self.valid = True

        self.frame_num = 0
        self.widget = None
        self.visible = False

        self.checked = []
        self.visible_tracks = set()
        self.tracks_list = None  # Массив из trackslistwidget
        self.min_dist = None

    # @property
    # def path(self):
    #     return self._path
    #
    # @path.setter
    # def path(self, path):
    #     self._path = os.path.abspath(path)
    #     self._rel_path = os.path.relpath(self._path, global_config["application"].get("video_folder_path"))
    #
    @property
    def relative_path(self):
        return os.path.relpath(self.path, global_config["application"].get("video_folder_path"))

    def set_valid(self, flag):
        self.valid = flag

    def get_params(self):
        return self.name, self.path

    def set_id(self, id_):
        self.file_id = int(id_)

    def set_widget(self, widget):
        self.widget = widget

    def set_checked(self, checked, min_dist=None):
        self.checked = checked
        self.min_dist = min_dist

    def set_tracks_list(self, tracks_list):
        self.tracks_list = tracks_list

    def check_self_valid(self):
        stream = cv2.VideoCapture(self.path)
        (grabbed, frame) = stream.read()
        self.frame_num = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        if not grabbed or self.frame_num < 1:
            self.frame_num = 0
            self.valid = False
        else:
            self.valid = True
        return self.valid

    def reset(self):
        self.visible_tracks = set()

    def set_track_visible(self, track_id):
        self.visible_tracks.add(track_id)

    def set_track_unvisible(self, track_id):
        self.visible_tracks.discard(track_id)

    def set_processed(self):
        self.processed = True


class TrackItem:
    def __init__(self, file, track_id, segments=None, **sup_info):
        self.file = file
        self.track_id = int(track_id)
        if segments is None:
            segments = []
        self.segments = segments
        self.color = np.random.randint(255, size=(3,))
        self.midl_frame = sup_info.get("mid", -1)
        self.frames_num = sup_info.get("num", -1)

        self.widget_item = None
        self.widget_time = None
        self.checked = False

    def set_widget_item(self, widget):
        self.widget_item = widget

    def set_widget_time(self, widget):
        self.widget_time = widget

    def set_checked(self, flag):
        self.checked = flag


class ClothesItem:
    def __init__(self, id_, name, color=None, score=-1, hist=None, db_id=None):
        self.clothes_id = id_
        self.name = name
        self.score = score
        self.color = color  # hex color
        self.hist = hist
        self.db_id = db_id

    def get_clothes_id(self):
        return self.clothes_id

    @property
    def rgb_color(self):
        return hex_to_rgb(self.color)



