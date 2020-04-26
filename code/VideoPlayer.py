import numpy as np
import cv2
from VideoStream import FileVideoStream

# May be this class is unnecessary

class VideoPlayer:
    def __init__(self):
        self.path = ""
        self.stream = None
        self.started = False
        self.last_frame = np.zeros((1,1,3))

    def initialize(self, path):
        self.stream = FileVideoStream(path, 30)
        # print(self.video.get(cv2.CAP_PROP_FPS))

    def play(self):
        self.stream.start()
        self.started = True

    def get_next_frame(self):
        if self.stream.more():
            self.last_frame = self.stream.read()
            return True, self.last_frame
        self.stream.pause()
        return False, 0

    def getFrameCount(self):
        return self.stream.stream.get(cv2.CAP_PROP_FRAME_COUNT)

    def setFrame(self, frame):
        self.stream.setFrame(frame)

    def release(self):
        if self.started:
            self.stream.stop()