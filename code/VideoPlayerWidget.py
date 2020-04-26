from PyQt5.QtGui import QImage, QPainter
from PyQt5 import QtWidgets, QtCore
from VideoPlayer import VideoPlayer
import time
import numpy as np


class VideoPlayerQWidget(QtWidgets.QWidget):
    STATUS_NOT_LOADED = -1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrame)
        self.image = QImage()
        self.video = VideoPlayer()
        self.frame_counter = 0
        self.parent = parent
        self.status = self.STATUS_NOT_LOADED
        self.play_speed = 1.0
        self.updateRate = 24

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def getFrameCount(self):
        return self.video.getFrameCount()

    def getCurrentTime(self):
        return self.frame_counter // self.updateRate

    def getCurrentTime_frame(self):
        return self.frame_counter

    def setFrame(self, frame):
        self.timer.stop()
        self.timer.start(1000)
        self.frame_counter = frame
        self.video.setFrame(frame)
        self.parent.updateDurationInfo(frame // 24)
        if self.status == self.STATUS_PLAYING:
            self.timer.start(int((1000 // 24) / self.play_speed))
        elif self.status in (self.STATUS_PAUSE, self.STATUS_INIT):
            self.timer.stop()
        else:
            print("BUG")

    def setVideo(self, path):
        self.frame_counter = 0
        self.video.initialize(path)
        self.status = self.STATUS_INIT

    def play(self):
        if self.status == self.STATUS_INIT:
            self.video.play()
        self.timer.start(int((1000 // 24) / self.play_speed))
        self.status = self.STATUS_PLAYING

    def pause(self):
        self.timer.stop()
        self.status = self.STATUS_PAUSE

    def nextFrame(self):
        ret, frame = self.video.get_next_frame()
        if not ret:
            self.pause()
            self.parent.set_btnPlay()
        self.frame_counter += 1
        try:
            img = self.get_qimage(frame)
            self.image = img.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            self.update()
            if self.frame_counter % self.updateRate == 0:
                self.parent.set_slider_time(self.frame_counter // self.updateRate)
        except:
            pass

    def stopPlaying(self):
        if self.status != self.STATUS_NOT_LOADED:
            self.release()


    def paintEvent(self, event):
        painter = QPainter(self)
        x_pos = (self.width() - self.image.width()) // 2
        y_pos = (self.height() - self.image.height()) // 2
        painter.drawImage(x_pos, y_pos, self.image)

    def setPlaySpeed(self, speed):
        if self.play_speed != speed:
            self.play_speed = speed
            if self.status == self.STATUS_PLAYING:
                self.timer.start(int((1000 // 24) / self.play_speed))

    def setUpdateRate(self, rate):
        self.updateRate = rate


    def release(self):
        self.video.release()
