from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QCursor

from enum import Enum
import time
import numpy as np
import cv2
from functools import partial
from profilehooks import profile
from processing import crop, xywh_to_xyxy

import os
MYDESIGNER = os.getenv("MYDESIGNER")
if MYDESIGNER is None:
    from processing import TrackItem, TracksReader, Visualizer, VideoStream


class state(Enum):
    NOT_LOADED = 1
    PLAYING = 2
    PAUSE = 3


class MyVideoWidget(QtWidgets.QWidget):

    frame_changed = QtCore.pyqtSignal(int)
    if MYDESIGNER is None:
        track_checked = QtCore.pyqtSignal(TrackItem, bool)
        track_to_find = QtCore.pyqtSignal(TrackItem, object)
    else:  # Виджет все равно не загружается
        track_checked = QtCore.pyqtSignal(TrackItem, bool)
        track_to_find = QtCore.pyqtSignal(TrackItem, object)

    def __init__(self, parent=None):
        super(MyVideoWidget, self).__init__(parent)
        uic.loadUi('MyWidgets/WidgetsUI/myvideowidget.ui', self)
        self.complete_ui()

        self.state = state.NOT_LOADED
        self.stream = None
        self.fps = 30
        self.frame_num = 0
        self.totalFrameNum = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._next_frame)

        self.sliderUpdateFrozen = False
        self.totalTime = self._get_QTime(0)
        self.timeFormat = 'mm:ss'

        self.cur_img = None
        self.frame_size = None
        self.visualizer = None

        self.set_stream_frame_size((9, 16))
        self.clear()

        self.file = None
        self.controller = None
        self.visualizer = None
        self.tracks_reader = None

        self._show_all_tracks = False

    def post_init(self, parent):
        self.controller = parent
        self.visualizer = Visualizer()
        self.tracks_reader = TracksReader(self.controller.myDB)

    def sizeHint(self):
        return QtCore.QSize(300, 170)

    def complete_ui(self):
        self.buttonPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.buttonPlay.clicked.connect(self._buttonPlay_clicked)
        self.sliderTime.sliderReleased.connect(self._get_slider_handle_value)
        self.sliderTime.sliderPressed.connect(self._freeze_slider)
        self.sliderTime.actionTriggered.connect(self._get_slider_groove_value)
        self.frameVideo.resizeSignal.connect(self.set_stream_frame_size)
        self.frame_changed.connect(self._update_slider_position)
        self.buttonDisplayAll.clicked.connect(self._display_all_tracks)
        self.frameVideo.mouse_clicked.connect(self._click_on_frame)

    def set_stream_frame_size(self, size):
        """ Метод для изменения размера считываемых кардов """
        self.frame_size = size
        if self.stream is not None:
            self.stream.set_frame_size(size)

    def set_video(self, file):
        self.file = file
        self.release()
        if not file.valid:
            self.clear()
            return

        cv_stream = cv2.VideoCapture(file.path)
        self.tracks_reader.set_file(file)
        if file.processed:
            self.stream = VideoStream(cv_stream,
                                      visualizer=self.visualizer,
                                      tracks_reader=self.tracks_reader)
            self.sup_stream = cv2.VideoCapture(file.path)
        else:
            self.stream = VideoStream(cv_stream)

        self._set_state(state.PAUSE)

        self.buttonPlay.setEnabled(True)
        self.fps = self.stream.get_fps()
        self.frame_num = 0
        self.totalFrameNum = int(self.file.frame_num)
        self.totalTime = self._get_QTime(self.totalFrameNum)
        self.timeFormat = 'hh:mm:ss' if self.totalFrameNum > 3600 else 'mm:ss'
        self.sliderTime.setRange(0, self.totalFrameNum)
        size = (self.frameVideo.height(), self.frameVideo.width())
        self.set_stream_frame_size(size)
        # ожидание прогрузки первых кадров, что бы отобразить первый кадр
        time.sleep(0.1)
        self._next_frame()

    def set_frame_num(self, frame_num):
        """ Метод для установки номера кадра с которого продолжить проищрывания видео """
        if self.stream is None:
            return
        cur_state = self.state
        self._set_state(state.PAUSE)
        if 0 <= frame_num < self.totalFrameNum:
            self.frame_num = frame_num
        else:
            self.frame_num = 0

        self._update_time_label()

        self.stream.set_frame_num(self.frame_num)
        time.sleep(0.1)
        self._next_frame()

        if 0 <= frame_num < self.totalFrameNum and cur_state == state.PLAYING:
            self._set_state(state.PLAYING)

    def _get_QTime(self, time):
        """ Получение аремени в формате QTime по номеру кадра"""
        time = int(time // self.fps)
        return QtCore.QTime((time // 3600) % 60, (time // 60) % 60, time % 60)

    # @profile()
    def _next_frame(self):
        """ Метод для вызова отрисовки следующего кадра"""
        if self.state == state.NOT_LOADED:
            return
        ret, self.cur_img = self.stream.get_next_frame()
        if not ret:
            if self.frame_num >= self.totalFrameNum - 1:
                self.set_frame_num(-1)
                print("конец")
            else:
                print("Ошибка во время чтения файла")
        else:
            self.frame_num += 1
            self.frameVideo.set_image(self.cur_img)
            self.frame_changed.emit(self.frame_num)
            self._update_time_label()

    def _set_state(self, state):
        """ Устанавливаем иконку в зависимлсти от текущего состояния """
        self.state = state
        if state == state.PAUSE:
            self.timer.stop()
            self.buttonPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.buttonDisplayAll.setEnabled(True)
        elif state == state.PLAYING:
            self.timer.start(int(1000 / self.fps))
            self.buttonPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self.buttonDisplayAll.setEnabled(False)
        elif state == state.NOT_LOADED:
            self.timer.stop()
            self.buttonPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.buttonPlay.setEnabled(False)
            self.buttonDisplayAll.setEnabled(False)
        else:
            print("А какую иконку сейчас ставить?")

    def _buttonPlay_clicked(self):
        """ Слот для работы кнопки проигрывания видео"""
        if self.state == state.PAUSE:
            self._set_state(state.PLAYING)
        elif self.state == state.PLAYING:
            self._set_state(state.PAUSE)
        else:
            print("BUG _buttonPlay_clicked")
            pass

    def _update_slider_position(self):
        """ Метод для отрисоки текущей позиции слайдера

        Текущая позиция хранмиться в self.frame_num
        """
        if not self.sliderUpdateFrozen:
            self.sliderTime.setValue(self.frame_num)

    def _update_time_label(self):
        """ Метод для отображения времени в котором находиться сейчас видео

        Текущая время(кадр) хранмиться в self.frame_num
        """
        # TODO возможно нужно убрать if
        if self.frame_num % self.fps == 1:
            current_time = self._get_QTime(self.frame_num)
            tStr = current_time.toString(self.timeFormat) + " / " + self.totalTime.toString(self.timeFormat)
            self.labelTime.setText(tStr)

        self.labelFrameCur.setText(str(self.frame_num))
        self.labelFrameAll.setText("/" + str(self.totalFrameNum))

    def _freeze_slider(self):
        """ Слот для остановки обновления слайдера"""
        self.sliderUpdateFrozen = True

    def _get_slider_handle_value(self):
        """ Слот для изменения текущего номера кадра """
        self.sliderUpdateFrozen = False
        if self.state == state.NOT_LOADED:
            self._update_slider_position()
            return
        value = self.sliderTime.value()
        self.set_frame_num(value)

    def _get_slider_groove_value(self, a):
        """ Слот для установки номера кадра когда нажали на слайдер вне ползунка"""
        if self.state == state.NOT_LOADED:
            self._update_slider_position()
            return

        if a in (3, 4):
            cursor_pos = self.mapFromGlobal(QCursor.pos()).x() - self.sliderTime.x()
            slider_width = self.sliderTime.width()
            frame = int((cursor_pos / slider_width) * self.totalFrameNum)
            self.set_frame_num(frame)

    def _display_all_tracks(self):
        if self.state != state.PAUSE:
            raise Exception("Вызов отрисовки всех треков не на паузе")
        bboxes, colors = self.tracks_reader.get_tracks_at(self.frame_num, all_tracks=True)
        self.cur_img = self.visualizer.draw_tracks(self.cur_img, bboxes, colors, self.cur_img.shape[0] // self.frame_size[0])
        self.frameVideo.set_image(self.cur_img)
        self._show_all_tracks = True

    def redraw(self):
        if self.state == state.PAUSE:
            self.set_frame_num(self.frame_num - 1)

    def _get_included(self, bboxes, pos):
        included = []
        for bbox in bboxes:
            if bbox[1] <= pos[0] <= (bbox[1] + bbox[3]) and bbox[2] <= pos[1] <= (bbox[2] + bbox[4]):
                included.append((int(bbox[0]), bbox[1:]))
        return included

    def _click_on_frame(self, pos):
        if self.state == state.PAUSE and self.tracks_reader:
            bboxes, _ = self.tracks_reader.get_tracks_at(self.frame_num, all_tracks=True)
            included = self._get_included(bboxes, pos)

            for i in range(len(included)):
                id_, bbox = included[i]
                included[i] = (self.controller.get_track_by_id(id_), bbox)
            if len(included) == 1:
                self._track_clicked(included[0], pos)
            elif len(included) > 1:
                menu = QtWidgets.QMenu(self)
                actions = []
                for track, bbox in included:
                    label = QtWidgets.QLabel("    ")
                    label.setFixedSize(60, 20)

                    label.setStyleSheet(f" background-color : rgb{tuple(track.color)};")
                    action = QtWidgets.QWidgetAction(self)
                    action.setDefaultWidget(label)
                    action.triggered.connect(partial(self._track_clicked, (track, bbox), pos))
                    actions.append(action)

                menu.addActions(actions)
                menu.exec_(self.mapToGlobal(QtCore.QPoint(pos[2] + 10, pos[3] + 10)))

        self._show_all_tracks = False
        self.redraw()

    def _track_clicked(self, track_info, pos):
        menu = QtWidgets.QMenu(self)
        track, bbox = track_info

        if not track.checked:
            show_action = QtWidgets.QAction("Показать", self)
            show_action.triggered.connect(partial(self._check_track, track, True))
        else:
            show_action = QtWidgets.QAction("Скрыть", self)
            show_action.triggered.connect(partial(self._check_track, track, False))

        search_action = QtWidgets.QAction("Поиск", self)
        search_action.triggered.connect(partial(self._place_to_find, track_info))

        menu.addActions((show_action, search_action))
        menu.exec_(self.mapToGlobal(QtCore.QPoint(pos[2] + 10, pos[3] + 10)))

    def _check_track(self, track, flag):
        print("Отобразить/скрыть", track.track_id)
        self.track_checked.emit(track, flag)

    def _place_to_find(self, track_info):
        track, bbox = track_info
        print("В поиск", track.track_id)
        height, width, _ = self.cur_img.shape
        img = self.get_full_frame_from(self.frame_num)
        if img is None:
            img = self.cur_img
        img = crop(img, xywh_to_xyxy(bbox), bbox_type='r')
        self.track_to_find.emit(track, img)

    def get_full_frame_from(self, frame_num):
        if frame_num < 0 or frame_num > self.totalFrameNum:
            print("Попытка взять кадр которого нет в видео")
            return None

        self.sup_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, img = self.sup_stream.read()
        img = np.asarray(img[:, :, ::-1])
        if ret:
            return img

        return None

    def get_cur_frame_num(self):
        return self.frame_num

    def clear(self):
        self.release()
        frame = np.ones((self.frame_size[0], self.frame_size[1], 3)).astype(np.uint8)*255
        self.frameVideo.set_image(frame)

    def release(self):
        """ Метод завершающий процесс считывания видео для освобождения ресурсов

        Вызвать при закрытие приложения
        """
        if self.state != state.NOT_LOADED:
            self.stream.stop()
            self.stream = None

        self._set_state(state.NOT_LOADED)


