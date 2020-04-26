# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1121, 662)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutMain = QtWidgets.QHBoxLayout(self.centralwidget)
        self.layoutMain.setObjectName("layoutMain")
        self.groupBoxVideo = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxVideo.sizePolicy().hasHeightForWidth())
        self.groupBoxVideo.setSizePolicy(sizePolicy)
        self.groupBoxVideo.setObjectName("groupBoxVideo")
        self.verticalLayoutMain = QtWidgets.QVBoxLayout(self.groupBoxVideo)
        self.verticalLayoutMain.setContentsMargins(5, 5, 5, 5)
        self.verticalLayoutMain.setObjectName("verticalLayoutMain")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnPlay = QtWidgets.QPushButton(self.groupBoxVideo)
        self.btnPlay.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnPlay.sizePolicy().hasHeightForWidth())
        self.btnPlay.setSizePolicy(sizePolicy)
        self.btnPlay.setMinimumSize(QtCore.QSize(0, 0))
        self.btnPlay.setMaximumSize(QtCore.QSize(25, 16777215))
        self.btnPlay.setSizeIncrement(QtCore.QSize(0, 0))
        self.btnPlay.setBaseSize(QtCore.QSize(0, 0))
        self.btnPlay.setText("")
        self.btnPlay.setObjectName("btnPlay")
        self.horizontalLayout.addWidget(self.btnPlay)
        self.cbPlaySpeed = QtWidgets.QComboBox(self.groupBoxVideo)
        self.cbPlaySpeed.setMaximumSize(QtCore.QSize(40, 16777215))
        self.cbPlaySpeed.setObjectName("cbPlaySpeed")
        self.cbPlaySpeed.addItem("")
        self.cbPlaySpeed.addItem("")
        self.horizontalLayout.addWidget(self.cbPlaySpeed)
        self.sliderTime = QtWidgets.QSlider(self.groupBoxVideo)
        self.sliderTime.setMinimumSize(QtCore.QSize(100, 0))
        self.sliderTime.setOrientation(QtCore.Qt.Horizontal)
        self.sliderTime.setObjectName("sliderTime")
        self.horizontalLayout.addWidget(self.sliderTime)
        self.cbRewindRate = QtWidgets.QComboBox(self.groupBoxVideo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbRewindRate.sizePolicy().hasHeightForWidth())
        self.cbRewindRate.setSizePolicy(sizePolicy)
        self.cbRewindRate.setMaximumSize(QtCore.QSize(55, 16777215))
        self.cbRewindRate.setEditable(False)
        self.cbRewindRate.setObjectName("cbRewindRate")
        self.cbRewindRate.addItem("")
        self.cbRewindRate.addItem("")
        self.horizontalLayout.addWidget(self.cbRewindRate)
        self.labelDuration = QtWidgets.QLabel(self.groupBoxVideo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelDuration.sizePolicy().hasHeightForWidth())
        self.labelDuration.setSizePolicy(sizePolicy)
        self.labelDuration.setMaximumSize(QtCore.QSize(16777215, 21))
        self.labelDuration.setObjectName("labelDuration")
        self.horizontalLayout.addWidget(self.labelDuration)
        self.verticalLayoutMain.addLayout(self.horizontalLayout)
        self.layoutMain.addWidget(self.groupBoxVideo)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_2.setSpacing(3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setContentsMargins(4, 3, 4, 3)
        self.verticalLayout_3.setSpacing(3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btnCheckAll = QtWidgets.QPushButton(self.groupBox_2)
        self.btnCheckAll.setEnabled(False)
        self.btnCheckAll.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.btnCheckAll.setObjectName("btnCheckAll")
        self.verticalLayout_3.addWidget(self.btnCheckAll)
        self.btnUncheckAll = QtWidgets.QPushButton(self.groupBox_2)
        self.btnUncheckAll.setEnabled(False)
        self.btnUncheckAll.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btnUncheckAll.setObjectName("btnUncheckAll")
        self.verticalLayout_3.addWidget(self.btnUncheckAll)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_3.addItem(spacerItem)
        self.btnApply = QtWidgets.QPushButton(self.groupBox_2)
        self.btnApply.setEnabled(False)
        self.btnApply.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.btnApply.setObjectName("btnApply")
        self.verticalLayout_3.addWidget(self.btnApply)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setContentsMargins(4, 3, 4, 3)
        self.verticalLayout_4.setSpacing(3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btnAdd = QtWidgets.QPushButton(self.groupBox_3)
        self.btnAdd.setObjectName("btnAdd")
        self.verticalLayout_4.addWidget(self.btnAdd)
        self.btnTrack = QtWidgets.QPushButton(self.groupBox_3)
        self.btnTrack.setObjectName("btnTrack")
        self.verticalLayout_4.addWidget(self.btnTrack)
        self.btnClothes = QtWidgets.QPushButton(self.groupBox_3)
        self.btnClothes.setObjectName("btnClothes")
        self.verticalLayout_4.addWidget(self.btnClothes)
        self.btnDeleteFiles = QtWidgets.QPushButton(self.groupBox_3)
        self.btnDeleteFiles.setObjectName("btnDeleteFiles")
        self.verticalLayout_4.addWidget(self.btnDeleteFiles)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem1)
        self.btnOpen = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnOpen.sizePolicy().hasHeightForWidth())
        self.btnOpen.setSizePolicy(sizePolicy)
        self.btnOpen.setAutoFillBackground(False)
        self.btnOpen.setObjectName("btnOpen")
        self.verticalLayout_4.addWidget(self.btnOpen)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_5.setContentsMargins(4, 3, 4, 3)
        self.verticalLayout_5.setSpacing(3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.btnSplit = QtWidgets.QPushButton(self.groupBox_4)
        self.btnSplit.setObjectName("btnSplit")
        self.verticalLayout_5.addWidget(self.btnSplit)
        self.btnConcat = QtWidgets.QPushButton(self.groupBox_4)
        self.btnConcat.setEnabled(True)
        self.btnConcat.setObjectName("btnConcat")
        self.verticalLayout_5.addWidget(self.btnConcat)
        self.btnDeleteEdit = QtWidgets.QPushButton(self.groupBox_4)
        self.btnDeleteEdit.setObjectName("btnDeleteEdit")
        self.verticalLayout_5.addWidget(self.btnDeleteEdit)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem2)
        self.btnSave = QtWidgets.QPushButton(self.groupBox_4)
        self.btnSave.setObjectName("btnSave")
        self.verticalLayout_5.addWidget(self.btnSave)
        self.btnReset = QtWidgets.QPushButton(self.groupBox_4)
        self.btnReset.setObjectName("btnReset")
        self.verticalLayout_5.addWidget(self.btnReset)
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_9.setContentsMargins(4, 3, 4, 3)
        self.verticalLayout_9.setSpacing(3)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.btnWithout = QtWidgets.QPushButton(self.groupBox_5)
        self.btnWithout.setObjectName("btnWithout")
        self.verticalLayout_9.addWidget(self.btnWithout)
        self.btnFined = QtWidgets.QPushButton(self.groupBox_5)
        self.btnFined.setObjectName("btnFined")
        self.verticalLayout_9.addWidget(self.btnFined)
        self.verticalLayout_2.addWidget(self.groupBox_5)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.layoutMain.addWidget(self.groupBox)
        self.groupBoxTracks = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxTracks.sizePolicy().hasHeightForWidth())
        self.groupBoxTracks.setSizePolicy(sizePolicy)
        self.groupBoxTracks.setMaximumSize(QtCore.QSize(170, 16777215))
        self.groupBoxTracks.setObjectName("groupBoxTracks")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBoxTracks)
        self.verticalLayout_6.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_6.setSpacing(2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBoxTracks)
        self.lineEdit.setMaximumSize(QtCore.QSize(16777215, 0))
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_6.addWidget(self.lineEdit)
        self.scrollAreaTracks = QtWidgets.QScrollArea(self.groupBoxTracks)
        self.scrollAreaTracks.setAutoFillBackground(True)
        self.scrollAreaTracks.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.scrollAreaTracks.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scrollAreaTracks.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollAreaTracks.setWidgetResizable(True)
        self.scrollAreaTracks.setObjectName("scrollAreaTracks")
        self.scrollAreaWidgetContentsTracks = QtWidgets.QWidget()
        self.scrollAreaWidgetContentsTracks.setGeometry(QtCore.QRect(0, 0, 131, 601))
        self.scrollAreaWidgetContentsTracks.setAutoFillBackground(False)
        self.scrollAreaWidgetContentsTracks.setObjectName("scrollAreaWidgetContentsTracks")
        self.verticalLayoutListTracks = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContentsTracks)
        self.verticalLayoutListTracks.setObjectName("verticalLayoutListTracks")
        self.scrollAreaTracks.setWidget(self.scrollAreaWidgetContentsTracks)
        self.verticalLayout_6.addWidget(self.scrollAreaTracks)
        self.layoutMain.addWidget(self.groupBoxTracks)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBoxFiles = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.groupBoxFiles.sizePolicy().hasHeightForWidth())
        self.groupBoxFiles.setSizePolicy(sizePolicy)
        self.groupBoxFiles.setMaximumSize(QtCore.QSize(150, 16777215))
        self.groupBoxFiles.setObjectName("groupBoxFiles")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBoxFiles)
        self.verticalLayout_8.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.listWidgetFiles = QtWidgets.QListWidget(self.groupBoxFiles)
        self.listWidgetFiles.setObjectName("listWidgetFiles")
        self.verticalLayout_8.addWidget(self.listWidgetFiles)
        self.verticalLayout.addWidget(self.groupBoxFiles)
        self.groupBoxClothes = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.groupBoxClothes.sizePolicy().hasHeightForWidth())
        self.groupBoxClothes.setSizePolicy(sizePolicy)
        self.groupBoxClothes.setMaximumSize(QtCore.QSize(150, 16777215))
        self.groupBoxClothes.setObjectName("groupBoxClothes")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBoxClothes)
        self.verticalLayout_7.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.scrollAreaClothes = QtWidgets.QScrollArea(self.groupBoxClothes)
        self.scrollAreaClothes.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.scrollAreaClothes.setWidgetResizable(True)
        self.scrollAreaClothes.setObjectName("scrollAreaClothes")
        self.scrollAreaWidgetContentsClothes = QtWidgets.QWidget()
        self.scrollAreaWidgetContentsClothes.setGeometry(QtCore.QRect(0, 0, 142, 392))
        self.scrollAreaWidgetContentsClothes.setObjectName("scrollAreaWidgetContentsClothes")
        self.verticalLayoutClothes = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContentsClothes)
        self.verticalLayoutClothes.setObjectName("verticalLayoutClothes")
        self.scrollAreaClothes.setWidget(self.scrollAreaWidgetContentsClothes)
        self.verticalLayout_7.addWidget(self.scrollAreaClothes)
        self.verticalLayout.addWidget(self.groupBoxClothes)
        self.layoutMain.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBoxVideo.setTitle(_translate("MainWindow", "VideoPlayer"))
        self.cbPlaySpeed.setItemText(0, _translate("MainWindow", "1"))
        self.cbPlaySpeed.setItemText(1, _translate("MainWindow", "0.2"))
        self.cbRewindRate.setItemText(0, _translate("MainWindow", "Sec"))
        self.cbRewindRate.setItemText(1, _translate("MainWindow", "Frame"))
        self.labelDuration.setText(_translate("MainWindow", "00:00/00:00"))
        self.groupBox.setTitle(_translate("MainWindow", "Buttons"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Tracks"))
        self.btnCheckAll.setText(_translate("MainWindow", "Check"))
        self.btnUncheckAll.setText(_translate("MainWindow", "Uncheck"))
        self.btnApply.setText(_translate("MainWindow", "Apply"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Files"))
        self.btnAdd.setText(_translate("MainWindow", "Add"))
        self.btnTrack.setText(_translate("MainWindow", "Track"))
        self.btnClothes.setText(_translate("MainWindow", "Clothes"))
        self.btnDeleteFiles.setText(_translate("MainWindow", "Delete"))
        self.btnOpen.setText(_translate("MainWindow", "View"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Editing"))
        self.btnSplit.setText(_translate("MainWindow", "Split"))
        self.btnConcat.setText(_translate("MainWindow", "Concat"))
        self.btnDeleteEdit.setText(_translate("MainWindow", "Delete"))
        self.btnSave.setText(_translate("MainWindow", "Save"))
        self.btnReset.setText(_translate("MainWindow", "Reset"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Clothes"))
        self.btnWithout.setText(_translate("MainWindow", "Without"))
        self.btnFined.setText(_translate("MainWindow", "Finde"))
        self.groupBoxTracks.setTitle(_translate("MainWindow", "Tracks"))
        self.groupBoxFiles.setTitle(_translate("MainWindow", "Files"))
        self.groupBoxClothes.setTitle(_translate("MainWindow", "Clothes"))