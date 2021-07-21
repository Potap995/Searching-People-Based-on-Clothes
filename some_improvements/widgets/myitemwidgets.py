from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPalette

import os
MYDESIGNER = os.getenv("MYDESIGNER")
if MYDESIGNER is None:
    from processing import FileItem, TrackItem


class MyFileItemWidget(QtWidgets.QWidget):

    if MYDESIGNER is None:
        openFile = QtCore.pyqtSignal(FileItem)
        processFile = QtCore.pyqtSignal(QtWidgets.QWidget)

    def __init__(self, parent=None, file=None):
        super(MyFileItemWidget, self).__init__(parent)
        uic.loadUi('MyWidgets/WidgetsUI/myfileitemwidget.ui', self)
        self.labelDist.setText("")
        if file:
            self.file = file
            if file.valid:
                name = file.name
            else:
                name = "(Плохо) " + file.name
            self.setObjectName(file.name)
            self.file_name.setText(name)
            if file.processed:
                self.buttonProcess.setEnabled(False)
                self.buttonProcess.setText("Обработан")
            else:
                self.buttonProcess.setEnabled(True)

        self.buttonOpen.clicked.connect(self._buttonOpen_clicked)
        self.buttonProcess.clicked.connect(self._buttonProcess_clicked)
        self.base_color = self.palette().color(QPalette.Background)
        self.set_highlighted(False)

    def contextMenuEvent(self, event):
        color = self.palette().color(QPalette.Background).name()
        self._set_background_color("#C8C8C8")
        menu = QtWidgets.QMenu(self)
        open_action = QtWidgets.QAction("Открыть", self)
        open_action.triggered.connect(self._buttonOpen_clicked)

        process_action = QtWidgets.QAction("Обработать", self)
        process_action.triggered.connect(self._buttonProcess_clicked)
        if self.file.processed:
            process_action.setEnabled(False)

        menu.addActions((open_action, process_action))
        menu.aboutToHide.connect(lambda: self._set_background_color(color))
        menu.exec_(self.mapToGlobal(event.pos()))

    def _buttonProcess_clicked(self):
        self.buttonProcess.setEnabled(False)
        self.processFile.emit(self)

    def _buttonOpen_clicked(self):
        self.openFile.emit(self.file)

    def set_highlighted(self, highlighted, min_dist=None):
        if highlighted == 1:
            color = "#10BC11"
        elif highlighted == 2:
            color = "#EBE300"
        else:
            color = self.base_color
        self._set_background_color(color)

        if min_dist is None:
            self.labelDist.setText("")
        else:
            self.labelDist.setText(f"({str(round(min_dist, 3))})")

    def _set_background_color(self, color):
        style = f"QWidget#{self.objectName()} {{background-color: {color};}}"
        self.setStyleSheet(style)

    def paintEvent(self, event):

        # Дальше код нужен, что бы работал setStyleSheet
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

    def set_processing_state(self, state_text: str):
        self.buttonProcess.setText(state_text)


class MyTrackItemWidget(QtWidgets.QWidget):

    if MYDESIGNER is None:
        state_changed = QtCore.pyqtSignal(TrackItem, bool)
        clothes_clicked = QtCore.pyqtSignal(TrackItem)

    def __init__(self, parent=None, track=None):
        super(MyTrackItemWidget, self).__init__(parent)
        uic.loadUi('MyWidgets/WidgetsUI/mytrackitemwidget.ui', self)
        if track:
            self.track = track
            self.track_id.setText(str(track.track_id))
            self.setObjectName("track_" + str(track.track_id))

            track.widget = self

        self.track_id.stateChanged.connect(self._state_changed)
        self.buttonClothes.clicked.connect(self._clothes_clicked)
        self._update_outside = False

    def set_state(self, state):
        if self.checked() != state:
            self.track.set_checked(bool(state))
            self._update_outside = True
            self.track_id.setCheckState(2 if state else 0)

    def checked(self):
        return bool(self.track_id.checkState())

    def _state_changed(self, state):
        self.track.set_checked(bool(state))
        if not self._update_outside:
            self._update_outside = False
            self.state_changed.emit(self.track, bool(state))
        else:
            self._update_outside = False

    def _clothes_clicked(self):
        self.clothes_clicked.emit(self.track)


class MyTrackTimeWidget(QtWidgets.QWidget):

    if MYDESIGNER is None:
        hided = QtCore.pyqtSignal(TrackItem)
        mistake_dist = QtCore.pyqtSignal(TrackItem, float, int)
        clicked = QtCore.pyqtSignal(TrackItem, bool) # , show(true) or hide(false)

    def __init__(self, parent=None, track=None, dist=None):
        super(MyTrackTimeWidget, self).__init__(parent)
        uic.loadUi('MyWidgets/WidgetsUI/mytracktimewidget.ui', self)
        self._color = QColor(255, 255, 255)
        self._closed = False
        if track:
            self.track = track
            self.labelID.setText(str(track.track_id))
            self.setObjectName("time_" + str(track.track_id))
            self.dist = dist
            if dist is not None:
                self.labelDistance.setText(f"{dist:.4f}")

    def _draw_time_line(self):
        height = self.labelTime.height()
        width = self.track.file.frame_num

        img = QPixmap(width, height)
        img.fill(self._color)
        painter = QPainter(img)
        pen = QPen(QtCore.Qt.black, 1)
        painter.setPen(pen)

        mid_height = height // 2 + 1
        painter.drawLine(0, mid_height, width, mid_height)

        pen.setWidth(11)
        for segment in self.track.segments:
            pen.setColor(QColor(self.track.color[0], self.track.color[1], self.track.color[2]))
            painter.setPen(pen)
            painter.drawLine(segment[0], mid_height, segment[1], mid_height)
        painter.end()

        label_width = self.labelTime.width()
        img = img.scaled(label_width, height)
        self.labelTime.setPixmap(img)

    # def resizeEvent(self, event):
    #     self._draw_time_line()

    def paintEvent(self, event):
        self._draw_time_line()

    def contextMenuEvent(self, event):
        self._set_background_color(QColor(200, 200, 200))
        menu = QtWidgets.QMenu(self)
        menu.aboutToHide.connect(self._aboutToHide_action)
        hide_action = QtWidgets.QAction("Скрыть", self)
        hide_action.triggered.connect(self._hide_signal)
        closer_action = QtWidgets.QAction("Ближе", self)
        closer_action.triggered.connect(self._closer_signal)
        farther_action = QtWidgets.QAction("Дальше", self)
        farther_action.triggered.connect(self._farther_signal)
        if self.dist is None:
            closer_action.setEnabled(False)
            farther_action.setEnabled(False)
        # else:
        #     self.clicked.emit(self.track, True)
        self.clicked.emit(self.track, True)


        menu.addActions((hide_action, closer_action, farther_action))
        menu.exec_(self.mapToGlobal(event.pos()))

    def _aboutToHide_action(self):
        self._set_background_color(QColor(255, 255, 255))
        self.clicked.emit(self.track, False)

    def _set_background_color(self, color):
        if not self._closed:
            self._color = color
            self.repaint()

    def _hide_signal(self):
        if not self._closed:
            self.hide_track_time()
            self.hided.emit(self.track)

    def _closer_signal(self):
        self.mistake_dist.emit(self.track, self.dist, -1)

    def _farther_signal(self):
        self.mistake_dist.emit(self.track, self.dist, +1)

    def hide_track_time(self):
        self._closed = True
        self.track.widget_time.deleteLater()
        self.track.set_widget_time(None)


class MyDescriptionItemWidget(QtWidgets.QWidget):

    delete = QtCore.pyqtSignal(int)
    changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, ident_=-1, description=None):
        super(MyDescriptionItemWidget, self).__init__(parent)
        uic.loadUi('MyWidgets/WidgetsUI/mydescriptionitemwidget.ui', self)

        self.description_id = ident_
        if description:
            self.description = description
            self.labelClothes.setText(description.name)
            self.labelScore.setText(str(round(description.score, 2)))
            if description.color:
                self._set_color(description.color)

        self.buttonDelete.clicked.connect(self._delete_button)
        self.buttonColor.clicked.connect(self._color_clicked)

    def _delete_button(self):
        self.deleteLater()
        self.delete.emit(self.description_id)

    def _set_color(self, color):
        self.buttonColor.setText("")
        self.buttonColor.setStyleSheet(f" background-color : {color};")

    def _color_clicked(self):
        cur_color = QColor()
        if self.description.color:
            cur_color = QColor(*self.description.rgb_color)
        ret = QtWidgets.QColorDialog.getColor(cur_color)
        if ret.isValid():
            self.description.color = ret.name()
            self._set_color(ret.name())
            self.changed.emit(self.description_id)

