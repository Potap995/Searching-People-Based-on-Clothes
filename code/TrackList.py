from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
from Visualizer import Visualizer


class TrackList:
    def __init__(self, parent):
        self.ch_tracks = dict()
        self.parent = parent
        self.parsed = False
        self.listWidget = self.parent.GetTracksList()
        self.Visualizer = Visualizer.getInstance()
        self.last_id = -1e3
        self.np_tracks = np.zeros(0)
        self.clothes = dict()
        self.tracks_path = ""
        self.clothes_path = ""

    def setInformation(self, tracks, clothes, tracks_path, clothes_path):
        self.clothes_path = clothes_path
        self.clothes = clothes
        self.setTracks(tracks, tracks_path, False)

    def setTracks(self, tracks, path, delete=True):
        if delete:
            self.clothes = dict()

        self.Visualizer.setTracks([])
        self.tracks_path = path
        self.np_tracks = np.copy(tracks)
        self.copy_np_tracks = np.copy(self.np_tracks)
        tracks_ids = np.unique(self.np_tracks[:, 1])
        self.last_id = tracks_ids[-1]
        self.Visualizer.setTracksInfo(self.np_tracks)
        self.displayTrackList()

    def displayTrackList(self):
        tracks_ids = np.unique(self.np_tracks[:, 1])
        for track in tracks_ids:
            clothes = self.getTrackClothes(track)
            flagIn = any(map(lambda v: v in clothes, self.parent.checked_clothes))
            if not self.parent.Fined or flagIn:
                self.add_chbox(f"{str(track)} ({', '.join(clothes)})", track)
        self.listWidget.addStretch()

    def getTrackClothes(self, track):
        track = str(track)
        ret = []
        if track in self.clothes:
            ret = self.clothes[track]
        return ret

    def add_chbox(self, text, track):
        wid = self.parent.addCHeckBoxTrack(text)
        self.ch_tracks[track] = wid

    def add_chbox_new(self, text, track):
        self.clearTracks(False)
        wid = self.parent.addCHeckBoxTrack(text)
        self.ch_tracks[track] = wid
        self.listWidget.addStretch()

    def clearTracks(self, all=True):
        for i in reversed(range(self.listWidget.count())):
            item = self.listWidget.itemAt(i)
            if item.spacerItem():
                self.listWidget.removeItem(item)
            elif self.listWidget.itemAt(i).widget() and all:
                self.listWidget.itemAt(i).widget().setParent(None)
            else:
                pass

    def getCheckedTracks(self):
        checked = []
        for id, ch_track in self.ch_tracks.items():
            if ch_track.isChecked():
                checked.append(id)
        return checked

    def checkAll(self):
        for _, ch_track in self.ch_tracks.items():
            ch_track.setCheckState(True)

    def uncheckAll(self):
        for _, ch_track in self.ch_tracks.items():
            ch_track.setCheckState(False)

    def applyTracks(self):
        self.Visualizer.setTracks(self.getCheckedTracks())

    def makeSplit(self, frame):
        checked = self.getCheckedTracks()
        if len(checked) != 1:
            self.parent.DisplayMsg("For split you should check one and only one track!")
            return
        cur_track = self.np_tracks[self.np_tracks[:, 1] == checked[0]]
        prev = np.sum(cur_track[:, 0] <= frame)
        post = np.sum(cur_track[:, 0] > frame)
        if prev == 0:
            self.parent.DisplayMsg("First part of track is empty")
            return
        if post == 0:
            self.parent.DisplayMsg("Second part of track is empty")
            return

        print("Frame to split: ", frame)
        mask = (self.np_tracks[:, 0] > frame) & (self.np_tracks[:, 1] == checked[0])
        self.np_tracks[mask, 1] = self.last_id + 1
        self.last_id += 1
        self.add_chbox_new(self.last_id, self.last_id)
        self.Visualizer.setTracksInfo(self.np_tracks)

    def makeConcat(self):
        checked = self.getCheckedTracks()
        if len(checked) != 2:
            self.parent.DisplayMsg("For concat you should check two and only two track!")
            return

        first_track = self.np_tracks[self.np_tracks[:, 1] == checked[0]]
        second_track = self.np_tracks[self.np_tracks[:, 1] == checked[1]]
        first_frames = set(np.unique(first_track[:, 0]))
        second_frames = set(np.unique(second_track[:, 0]))

        if len(first_frames & second_frames) > 4:
            self.parent.DisplayMsg("Tracks can overlap in no more than 4 frames")
            return

        mask = (self.np_tracks[:, 1] == checked[1])
        self.np_tracks[mask, 1] = checked[0]
        self.ch_tracks[checked[1]].setParent(None)
        del self.ch_tracks[checked[1]]
        self.Visualizer.setTracksInfo(self.np_tracks)

    def deleteTracks(self):
        checked = self.getCheckedTracks()
        mask = np.array([True] * self.np_tracks.shape[0])

        for track in checked:
            mask = mask & (self.np_tracks[:, 1] == track)
            self.ch_tracks[track].setParent(None)
            del self.ch_tracks[track]

        self.np_tracks = self.np_tracks[mask, :]
        self.Visualizer.setTracksInfo(self.np_tracks)

    def resetChanges(self):
        self.np_tracks = np.copy(self.copy_np_tracks)
        self.clearTracks()
        self.displayTrackList()

    def saveChanges(self):
        tracks_file = open(self.tracks_path, "w")
        for line in self.np_tracks:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                line[0], line[1], line[2], line[3], line[4], line[5]),
                  file=tracks_file)
        tracks_file.close()
        self.copy_np_tracks = np.copy(self.np_tracks)
        self.parent.DisplayMsg("Saved!")
