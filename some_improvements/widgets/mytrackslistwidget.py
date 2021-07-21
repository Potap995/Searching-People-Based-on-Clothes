from PyQt5 import QtWidgets, QtCore

from processing import TrackItem
from MyWidgets import MyTrackItemWidget, MyTimesListWidget
from profilehooks import profile


import time


class MyTracksListWidget(QtCore.QObject):

    track_state_changed = QtCore.pyqtSignal(TrackItem, bool, bool)  # track, state, redraw
    track_clothes_cliked = QtCore.pyqtSignal(TrackItem, object)

    def __init__(self, parent=None):
        super(MyTracksListWidget, self).__init__(parent)
        self.controller = parent
        self.scrollareaTracks = parent.scrollareacontentsTracks
        self.layoutTracks = parent.verticallayoutTracks
        self.labelFile = parent.labelOpenedFile
        self.buttonSort = parent.buttonSortTracks
        self.buttonCheck = parent.buttonCheckTracks
        self.listwidgetTimes = parent.times_list
        self.db = parent.myDB

        self.tracks = {}
        self.file = None
        self._checked_count = 0

        self.buttonSort.clicked.connect(self._sort_clicked)
        self.buttonCheck.clicked.connect(self._check_all_clicked)
        self.clear()

    def clear(self):
        self._set_not_loaded()
        self.listwidgetTimes.clear()
        self._update_check_button()
        self.buttonSort.setEnabled(False)
        self.buttonCheck.setEnabled(False)

    def set_file(self, file):
        self.clear()
        self.file = file
        # self.file.reset()
        self.labelFile.setText(file.name)
        self.listwidgetTimes.set_file(file)

        if file.processed and file.valid:
            self.buttonSort.setEnabled(True)
            self.buttonCheck.setEnabled(True)

            ids_time = self.db.get_tracks_time(file)
            ids_info = self.db.get_tracks_info(file)
            for id_ in ids_time:
                track = TrackItem(file, id_, ids_time[id_], **ids_info[id_])
                widget = MyTrackItemWidget(self.scrollareaTracks, track)
                track.set_widget_item(widget)
                self.tracks[id_] = track
                widget.state_changed.connect(self._track_state_changed)
                widget.clothes_clicked.connect(self._description_clicked)

        self.file.set_tracks_list(self.tracks)

    def display(self):
        if self.file.processed and self.file.valid:
            self._propagate_state(self.file.checked, True)
            self._display_tracks_list()

    def set_track_state(self, track, state):  # TODO возможно стоит объединить с _track_state_changed ( и _propagate_state)
        if track.track_id not in self.tracks:
            raise Exception(f"Такого трека нет {track.track_id} в track_list")
        else:
            if self.tracks[track.track_id].widget_item.checked() != state:
                self._update_checked_count(state)
                self.tracks[track.track_id].widget_item.set_state(state)
                self.change_file_visible_set(track, state)

    def change_file_visible_set(self, track, state):
        if state:
            self.file.set_track_visible(track.track_id)
        else:
            self.file.set_track_unvisible(track.track_id)

    def _set_not_loaded(self):
        self.tracks = {}
        self._checked_count = 0
        self.labelFile.setText(" ")
        self.buttonSort.setEnabled(False)
        self.buttonCheck.setEnabled(False)
        self._clear_all_tracks()

    def _display_tracks_list(self):
        for id_ in self.tracks:
            if self.tracks[id_].widget_item:
                if self.tracks[id_].widget_item.checked():
                    self.tracks[id_].widget_item.setVisible(True)
                    self.layoutTracks.addWidget(self.tracks[id_].widget_item, alignment=QtCore.Qt.AlignTop)
        for id_ in self.tracks:
            if self.tracks[id_].widget_item:
                if not self.tracks[id_].widget_item.checked():
                    self.tracks[id_].widget_item.setVisible(True)
                    self.layoutTracks.addWidget(self.tracks[id_].widget_item, alignment=QtCore.Qt.AlignTop)

        self.layoutTracks.addStretch()

    def _clear_all_tracks(self, strictly=True):
        while self.layoutTracks.count():
            item = self.layoutTracks.takeAt(0)
            if item.widget() is not None:
                if strictly:
                    item.widget().track.set_widget_item(None)
                    item.widget().deleteLater()
                else:
                    item.widget().setVisible(False)

            self.layoutTracks.removeItem(item)

    def _sort_clicked(self):
        self._clear_all_tracks(strictly=False)
        self._display_tracks_list()

    def _track_state_changed(self, track, state, redraw=True):
        self._update_checked_count(state)
        self.change_file_visible_set(track, state)
        self.track_state_changed.emit(track, state, redraw)

    def _update_checked_count(self, state):
        if state:
            self._checked_count += 1
        else:
            self._checked_count -= 1

        self._update_check_button()

        # print("Checked: ", self._checked_count)

    def _update_check_button(self):
        if self._checked_count > 0:
            self.buttonCheck.setText("Сбросить все")
        else:
            self.buttonCheck.setText("Выделить все")

    # @profile
    def _check_all_clicked(self, event=False, state=True):
        ids_ = list(self.tracks.keys())
        if self._checked_count > 0 or not state:  # Сбрасываем
            state = False
            self._checked_count = len(ids_)
        else:                        # Выделяем
            state = True
            self._checked_count = 0

        self._propagate_state(ids_, state)
        self._sort_clicked()
        self._update_check_button()

    def _propagate_state(self, ids_, state):
        start = time.perf_counter()
        if len(ids_) == 0:
            return
        for id_ in ids_[:-1]:
            self.tracks[id_].widget_item.set_state(state)
            self._track_state_changed(self.tracks[id_], state, redraw=False)
        id_ = ids_[-1]
        self.tracks[id_].widget_item.set_state(state)
        self._track_state_changed(self.tracks[id_], state, redraw=True)
        end = time.perf_counter()
        if end - start > 1:
            print("long pro[agate state", end - start)

    def _description_clicked(self, track):
        self.track_clothes_cliked.emit(track, None)

    def reset(self):
        self._check_all_clicked(state=False)






