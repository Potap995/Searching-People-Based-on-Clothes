import numpy as np
import cv2
import time
from threading import Event, Thread


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = 1
    return v / norm


def xyxy_to_xywh(bbox_xyxy):
    bbox_xywh = np.array(bbox_xyxy)
    bbox_xywh[:, 2:] -= bbox_xywh[:, :2]
    return bbox_xywh


def xywh_to_xyxy(bbox_xywh):
    bbox_xyxy = np.array(bbox_xywh)
    bbox_xyxy[2:] += bbox_xyxy[:2]
    return bbox_xyxy


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#%02x%02x%02x" % (r, g, b)


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))


def crop(img, bbox, border=0, bbox_type="a"):
    """
    bbox: xy_xy
    bbox_type: может быть relative 'r' или absolute 'a'
    """
    height, width, _ = img.shape
    x1, y1, x2, y2 = bbox
    if bbox_type == "r":
        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)
    if border < 1:
        x_border = int((x2 - x1) * border)
        y_border = int((y2 - y1) * border)
    else:
        x_border = int(border)
        y_border = int(border)
    x1 = (x1 - x_border) if (x1 - x_border) >= 0 else 0
    y1 = (y1 - y_border) if (y1 - y_border) >= 0 else 0
    x2 = (x2 + x_border) if (x2 + x_border) <= width else width
    y2 = (y2 + y_border) if (y2 + y_border) <= height else height
    return img[y1:y2, x1:x2]


def resize_img(frame, size):
    curSize = (frame.shape[0], frame.shape[1])  # высота, ширина
    widthRatio = size[1] / curSize[1]
    heightRatio = size[0] / curSize[0]
    if widthRatio < heightRatio:
        ratio = widthRatio
    else:
        ratio = heightRatio

    if ratio < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    interpolation = cv2.INTER_NEAREST

    new_size = (int(curSize[1] * ratio), int(curSize[0] * ratio))
    return cv2.resize(frame, new_size, interpolation=interpolation)


class RepeatedTimer:

    """Repeat `function` every `interval` seconds."""

    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start_time = time.time()
        self.event = Event()
        self.thread = None

    def start(self):
        self.event.clear()
        self.thread = Thread(target=self._target, daemon=True)
        self.thread.start()

    def _target(self):
        while not self.event.wait(self._time):
            self.function(*self.args, **self.kwargs)

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start_time) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()
