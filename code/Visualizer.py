import numpy as np
import cv2


class Visualizer:
    __instance = None
    stream = None
    tracks_info = None
    tracks = None
    colours = np.random.randint(255, size=(64, 3))

    def __init__(self):
        if Visualizer.__instance is None:
            Visualizer.__instance = self
        else:
            raise Exception("This class is a singleton!")

    @staticmethod
    def getInstance():
        if Visualizer.__instance is None:
            Visualizer()
        return Visualizer.__instance

    def setStream(self, stream):
        self.stream = stream

    def setTracksInfo(self, tracks_info):
        self.tracks_info = tracks_info

    def setTracks(self, tracks):
        if self.stream:
            self.stream.pause()
            self.stream.resetFrames()
            self.tracks = set(tracks)
            self.stream.resume()

    def drowTracks(self, img, frame):
        if self.tracks is not None and self.tracks_info is not None:
            frame_tracks = self.tracks_info[self.tracks_info[:, 0] == frame]
            for track in frame_tracks:
                if track[1] in self.tracks:
                    colour = self.colours[track[1] % 64, :].tolist()
                    cv2.rectangle(img, (track[2], track[3]), (track[2] + track[4], track[3] + track[5]), colour, 2)
                    cv2.putText(img, str(track[1]), (track[2], track[3]), 0, 5e-3 * 200, (0, 255, 0), 1)

        return img
