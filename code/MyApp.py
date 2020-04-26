from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtWidgets import QStyle
from VideoPlayerWidget import VideoPlayerQWidget
from FilesProcessing import FilesGetter
from TrackList import TrackList
from FilesList import FilesList
from ClothesList import ClothesList
import MainWindow

from PyQt5.QtCore import QObject, QThread, pyqtSignal
import time

from functools import partial


class MainApp(QtWidgets.QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.VideoWidget = VideoPlayerQWidget(self)
        self.TrackList = TrackList(self)
        self.FilesList = FilesList(self)
        self.ClothesList = ClothesList(self)

        self.addWidgets()
        self.makeConnects()

        self.clicked = False
        self.duration = 0
        self.UpdateRate = 24
        self.SliderFrozen = False
        self.Fined = False
        self.checked_clothes = []
        self.setEditBtns(False)

    '''
    Buttons
    '''
    # Files

    def _getCurRowName(self):
        if self.listWidgetFiles.currentRow() < 0:
            self.statusBar.showMessage("Select file")
            return
        item = self.listWidgetFiles.item(self.listWidgetFiles.currentRow())
        if item.flags() & Qt.ItemIsEnabled:
            return item.text()
        else:
            return ""

    def _preFilesClick(self):
        name = self._getCurRowName()
        if self.clicked:
            self.VideoWidget.release()
        self.btnPlay.setEnabled(False)
        self.sliderTime.setEnabled(False)
        self.TrackList.clearTracks()
        return name

    def btnOpen_clicked(self):
        name = self._preFilesClick()
        if name == "":
            return
        self.clicked = True

        tracks, tracks_path, clothes, clothes_path = self.FilesList.openFile(name)

        self.btnPlay.setEnabled(True)
        self.sliderTime.setEnabled(True)
        self.TrackList.clearTracks()
        if tracks_path != "" and clothes_path != "":
            self.TrackList.setInformation(tracks, clothes, tracks_path, clothes_path)
            return
        if tracks_path != "":
            self.setEditBtns(True)
            self.TrackList.setTracks(tracks, tracks_path)

    def btnTrack_clicked(self):
        name = self._preFilesClick()
        self.FilesList.trackFile(name)

    def btnClothes_clicked(self):
        name = self._preFilesClick()
        self.FilesList.clothesFile(name)

    def btnAdd_clicked(self):
        currentFileName = \
        QtWidgets.QFileDialog.getOpenFileName(None, "Select Video File", "D:/Programming/CourseWork_3_dev/output/video",
                                              "*.mp4")[0]
        if currentFileName == "":
            return
        self.FilesList.addNewFile(currentFileName)

    def btnDeleteFiles_clicked(self):
        name = self._preFilesClick()
        self.VideoWidget.stopPlaying()
        self.TrackList.clearTracks()
        self.TrackList.applyTracks()
        self.FilesList.deleteFile(name)

    # Tracks

    def btnApply_clicked(self):
        self.TrackList.applyTracks()

    # Editing

    def btnConcat_clicked(self):
        if self.VideoWidget.status != self.VideoWidget.STATUS_PAUSE:
            self.DisplayMsg("You can split only on PAUSE")
            return
        self.TrackList.makeConcat()

    def btnSplit_clicked(self):
        if self.VideoWidget.status != self.VideoWidget.STATUS_PAUSE:
            self.DisplayMsg("You can split only on PAUSE")
            return
        self.TrackList.makeSplit(self.VideoWidget.getCurrentTime_frame())

    def btnReset_clicked(self):
        self.TrackList.resetChanges()

    def btnSave_clicked(self):
        self.TrackList.saveChanges()

    def btnDeleteEdit_clicked(self):
        self.TrackList.deleteTracks()

    # Clothes

    def btnFined_clicked(self):
        if self.clicked:
            self.VideoWidget.release()
        self.btnPlay.setEnabled(False)
        self.sliderTime.setEnabled(False)
        self.TrackList.clearTracks()
        self.checked_clothes = self.ClothesList.getCheckedClothes()
        if len(self.checked_clothes) > 0:
            self.Fined = True
            self.FilesList.displayWithClothes(self.checked_clothes)

    def btnWithout_clicked(self):
        self.FilesList.hideAll()
        self.FilesList.matchAll()
        self.FilesList.showMatched()

    # Other

    def btnPlay_clicked(self):
        if self.VideoWidget.status in (self.VideoWidget.STATUS_PAUSE, self.VideoWidget.STATUS_INIT):
            self.VideoWidget.play()
            self.btnPlay.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.VideoWidget.status == self.VideoWidget.STATUS_PLAYING:
            self.VideoWidget.pause()
            self.btnPlay.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    '''
    Buttons Set
    '''

    def setClothesBtns(self, flag):
        self.btnFined.setEnabled(flag)
        self.btnWithout.setEnabled(flag)

    def setEditBtns(self, flag):
        self.btnSplit.setEnabled(flag)
        self.btnConcat.setEnabled(flag)
        self.btnSave.setEnabled(flag)
        self.btnReset.setEnabled(flag)
        self.btnDeleteEdit.setEnabled(flag)

    def setTracksBtns(self, state):
        self.btnApply.setEnabled(state)
        self.btnCheckAll.setEnabled(state)
        self.btnUncheckAll.setEnabled(state)

    def set_btnPlay(self):
        self.btnPlay.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def freezeSlider(self):
        self.SliderFrozen = True

    def makeConnects(self):
        self.btnOpen.clicked.connect(self.btnOpen_clicked)
        self.sliderTime.sliderReleased.connect(self.set_time)
        self.sliderTime.sliderPressed.connect(self.freezeSlider)
        self.btnPlay.clicked.connect(self.btnPlay_clicked)
        self.btnApply.clicked.connect(self.btnApply_clicked)
        self.cbPlaySpeed.activated[str].connect(self.cbPlaySpeed_changed)
        self.cbRewindRate.activated[str].connect(self.cbRewindRate_changed)
        self.btnCheckAll.clicked.connect(self.TrackList.checkAll)
        self.btnUncheckAll.clicked.connect(self.TrackList.uncheckAll)
        self.btnAdd.clicked.connect(self.btnAdd_clicked)
        self.btnSplit.clicked.connect(self.btnSplit_clicked)
        self.btnClothes.clicked.connect(self.btnClothes_clicked)
        self.btnTrack.clicked.connect(self.btnTrack_clicked)
        self.btnReset.clicked.connect(self.btnReset_clicked)
        self.btnSave.clicked.connect(self.btnSave_clicked)
        self.btnFined.clicked.connect(self.btnFined_clicked)
        self.btnConcat.clicked.connect(self.btnConcat_clicked)
        self.btnWithout.clicked.connect(self.btnWithout_clicked)
        self.btnDeleteEdit.clicked.connect(self.btnDeleteEdit_clicked)
        self.btnDeleteFiles.clicked.connect(self.btnDeleteFiles_clicked)

    '''
    Video methods
    '''

    def set_time(self):
        self.SliderFrozen = False
        value = self.sliderTime.value()
        if self.cbRewindRate.currentText() == "Sec":
            value *= 24
        self.VideoWidget.setFrame(value)

    def set_slider_time(self, value):
        if not self.SliderFrozen:
            self.sliderTime.setValue(value)
        if  self.cbRewindRate.currentText() == "Frame":
            value = value // 24
        self.updateDurationInfo(value)

    def updateDurationInfo(self, currentInfo):
        duration = self.duration // 24
        if currentInfo or duration:
            currentTime = QTime((currentInfo / 3600) % 60, (currentInfo / 60) % 60,
                                currentInfo % 60)
            totalTime = QTime((duration / 3600) % 60, (duration / 60) % 60,
                              duration % 60);

            format = 'hh:mm:ss' if duration > 3600 else 'mm:ss'
            tStr = currentTime.toString(format) + " / " + totalTime.toString(format)
        else:
            tStr = ""

        self.labelDuration.setText(tStr)

    def openVideo(self, path):
        self.VideoWidget.setVideo(path)
        self.duration = self.VideoWidget.getFrameCount()
        self.sliderTime.setRange(0, self.duration // self.UpdateRate)
        self.sliderTime.setValue(0)
        self.updateDurationInfo(0)

    def cbPlaySpeed_changed(self, text):
        self.VideoWidget.setPlaySpeed(float(text))

    def cbRewindRate_changed(self, text):
        if text == "Sec":
            self.UpdateRate = 24
        elif text == "Frame":
            self.UpdateRate = 1

        self.VideoWidget.setUpdateRate(self.UpdateRate)
        self.sliderTime.setRange(0, self.duration // self.UpdateRate)
        self.set_slider_time(self.VideoWidget.getCurrentTime())

    '''
    For Lists
    '''


    def GetTracksList(self):
        return self.verticalLayoutListTracks

    def GetFilesList(self):
        return self.listWidgetFiles

    def GetClothesList(self):
        return self.verticalLayoutClothes

    def addCHeckBoxTrack(self, name):
        wid = QtWidgets.QCheckBox(self.scrollAreaWidgetContentsTracks)
        wid.setText(f"{name}")
        wid.setChecked(True)
        self.verticalLayoutListTracks.addWidget(wid, 0, alignment=Qt.AlignTop)
        return wid

    def addCHeckBoxClothes(self, name):
        wid = QtWidgets.QCheckBox(self.scrollAreaWidgetContentsClothes)
        wid.setText(f"{name}")
        wid.setChecked(False)
        self.verticalLayoutClothes.addWidget(wid, 0, alignment=Qt.AlignTop)
        return wid

    '''
    Other
    '''

    def DisplayMsg(self, str):
        self.statusBar.showMessage(str)

    def addWidgets(self):
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.VideoWidget.setSizePolicy(sizePolicy)
        self.VideoWidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.verticalLayoutMain.insertWidget(0, self.VideoWidget)
        self.btnPlay.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def closeEvent(self, event):
        self.FilesList.endProcessing()
        self.VideoWidget.release()
