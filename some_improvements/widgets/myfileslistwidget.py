from pathlib import Path

from PyQt5 import QtWidgets, QtCore

from processing import FileItem
from MyWidgets import MyFileItemWidget
from MyWidgets.processqueue import ProcessingQueue
import os
from processing import config


def get_name(path):
    path = Path(path)
    name = path.stem
    return name


class MyFilesListWidget(QtCore.QObject):
    openFile = QtCore.pyqtSignal(FileItem)

    def __init__(self, parent=None):
        super(MyFilesListWidget, self).__init__(parent)
        self.controller = parent
        self.scrollareaFiles = parent.scrollareacontentsFiles
        self.layoutFiles = parent.verticallayoutFiles
        self.buttonAddFile = parent.buttonAddFile
        self.db = parent.myDB
        self.files = {}

        self.files_process_queue = ProcessingQueue(self.db)

        self.buttonAddFile.clicked.connect(self._buttonAddFile_clicked)
        self.set_availability(False)
        self.read_saved_files()

    def read_saved_files(self):

        # TODO проверить,что сохраненые файлы не удалились
        self.set_availability(False)
        files = self.db.get_files_list()
        for file in files:
            id_, name, path, processed_flag = file
            path = os.path.normpath(os.path.join(config.global_config["application"].get("video_folder_path"),
                                                 path))
            file_item = FileItem(name, path, id_, processed_flag)

            res = Path(file_item.path)
            if not res.exists():
                file_item.set_valid(False)

            fileWidget = MyFileItemWidget(self.scrollareaFiles, file_item)
            file_item.set_widget(fileWidget)
            self.files[file_item.file_id] = file_item
            self.files[file_item.file_id].widget.openFile.connect(self._openFile_clicked)
            self.files[file_item.file_id].widget.processFile.connect(self._processFile_cliked)

        self._display_files_list()
        self.set_availability(True)

    def _display_files_list(self):
        for file in self.files:
            if self.files[file].widget:
                self.layoutFiles.addWidget(self.files[file].widget, alignment=QtCore.Qt.AlignTop)

        self.layoutFiles.addStretch()

    def _openFile_clicked(self, file):
        self.openFile.emit(file)

    def _processFile_cliked(self, file_widget):
        print(file_widget.file.file_id)
        self.files_process_queue.add_to_queue(file_widget)

    def _buttonAddFile_clicked(self):
        current_file_names = \
            QtWidgets.QFileDialog.getOpenFileNames(None, "Select Video File",
                                                   "../Tracker/data/MOT/video",
                                                   "*.mp4 *.mkv")[0]
        for current_file_name in current_file_names:
            if current_file_name == "":
                return
            else:
                print(current_file_name, get_name(current_file_name))
                fileItem = FileItem(get_name(current_file_name), current_file_name)
                try:
                    id_ = self.db.add_file(fileItem)
                except FileExistsError as e:
                    QtWidgets.QMessageBox.warning(self.scrollareaFiles, "Внимание", str(e))
                else:
                    fileItem.set_id(id_)
                    fileWidget = MyFileItemWidget(self.scrollareaFiles, fileItem)
                    fileItem.set_widget(fileWidget)
                    self.files[fileItem.file_id] = fileItem
                    self.files[fileItem.file_id].widget.openFile.connect(self._openFile_clicked)
                    self.files[fileItem.file_id].widget.processFile.connect(self._processFile_cliked)
                    files_count = len(self.files)
                    self.layoutFiles.insertWidget(files_count - 1, fileWidget, alignment=QtCore.Qt.AlignTop)

    def set_matches(self, matches):
        for file_id in self.files:
            highlighted = 0
            self.files[file_id].set_checked([])
            min_dist = None
            if file_id in matches:
                ids = []
                matches_type = matches[file_id]["match_type"]
                min_dist = matches[file_id]["min_dist"]
                if len(matches_type[1]):
                    highlighted = 2
                    ids.extend(matches_type[1])
                if len(matches_type[0]):
                    highlighted = 1
                    ids.extend(matches_type[0])
                self.files[file_id].set_checked(ids, min_dist)
                min_dist = min_dist
            self.files[file_id].widget.set_highlighted(highlighted, min_dist)

    def reset(self):
        print("lol")
        for file_id in self.files:
            self.files[file_id].set_checked([])
            self.files[file_id].widget.set_highlighted(0)

    def clear(self):
        self.files_process_queue.stop()
        self.files_process_queue = ProcessingQueue(self.db)
        self.files = {}

        while self.layoutFiles.count():
            item = self.layoutFiles.takeAt(0)
            if item.widget() is not None:
                item.widget().file.set_widget(None)
                item.widget().deleteLater()

            self.layoutFiles.removeItem(item)

    def set_availability(self, state):
        if state:
            self.buttonAddFile.setEnabled(True)
            self.buttonAddFile.setText("Добавить")
        else:
            self.buttonAddFile.setEnabled(False)
            self.buttonAddFile.setText("Загрузка ...")

    def release(self):
        self.files_process_queue.stop()
