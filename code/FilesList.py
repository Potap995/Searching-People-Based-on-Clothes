from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread
from TrackerProcessor import SingleTrackerProcessor
from ClothesProcessor import SingleClothesProcessor
from FilesProcessing import FilesGetter

from functools import partial
from pathlib import Path
import os
import numpy as np
import shutil
import threading
import json

class FilesList:
    LOADED = 0
    TRACKED = 1
    FOUND = 2
    types = ["❌", "⚠", "✅"]

    def __init__(self, parent):
        self.video_folder = "data\\video\\"
        self.tracks_folder = "data\\tracks\\"
        self.clothes_folder = "data\\clothes\\"
        self.files = dict()
        self.parent = parent
        self.filesListWidget = self.parent.GetFilesList()
        self.readProcessedFiles()
        self.counter = 0
        self.threads = dict()

    def readProcessedFiles(self):
        videos = os.listdir(self.video_folder)
        for video in videos:
            file_name, file_type = video.rsplit('.', 1)
            if file_type != "mp4":
                continue

            paths = FilesGetter.finedAllFiles(self.video_folder + file_name)
            if paths[1] == "":
                self.addToList(file_name, self.LOADED, paths)
                continue
            tracks_file = open(paths[1], "r")
            tracks = self.parsTracks(tracks_file.readlines())
            tracks_file.close()
            if tracks.shape[0] == 0:
                self.addToList(file_name, self.LOADED, paths)
                continue

            if paths[2] == "":
                self.addToList(file_name, self.TRACKED, paths)
                continue
            # clothes_file = open(paths[2], "r")
            # clothes = self.parsTracks(clothes_file.readlines())
            # clothes_file.close()
            self.addToList(file_name, self.FOUND, paths)

    def addToList(self, file_name, type, paths):
        item = QtWidgets.QListWidgetItem()
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        item.setText(f"{self.types[type]}{file_name}")
        self.filesListWidget.addItem(item)
        self.files[file_name] = [item, type, paths, True]

    def parsTracks(self, tracks):
        tracks_list = []
        for line in tracks:
            line = line.strip().split(",")
            if len(line) != 10:
                self.parent.DisplayMsg("⚠ Can't parse tracks file")
                tracks_list = []
                break
            try:
                line = list(map(int, map(float, line)))
            except Exception as ex:
                self.parent.DisplayMsg("⚠ Can't parse tracks file")
                tracks_list = []
                break
            tracks_list.append(line)

        return np.array(tracks_list)

    def addNewFile(self, path_from):
        my_thread = threading.Thread(target=self.copy, args=(path_from,))
        my_thread.start()

    def trackFile(self, file_name):
        file_name = file_name.strip("❌⚠✅")
        if file_name not in self.files:
            print("bug")
            return
        if self.files[file_name][1] != self.LOADED:
            self.parent.DisplayMsg(f"⚠ File '{file_name}' already tracked!")
            return
        video_path = self.video_folder + file_name + ".mp4"
        tracks_path = self.tracks_folder + file_name + ".txt"
        processor = SingleTrackerProcessor(video_path, tracks_path)
        self.runTread(processor, file_name, self.TRACKED)

    def clothesFile(self, file_name):
        file_name = file_name.strip("❌⚠✅")
        if file_name not in self.files:
            print("bug")
            return
        if self.files[file_name][1] == self.LOADED:
            self.parent.DisplayMsg(f"⚠ File '{file_name}' not tracked yet!")
            return
        if self.files[file_name][1] == self.FOUND:
            self.parent.DisplayMsg(f"⚠ File '{file_name}' already founded(clothes) !")
            return
        video_path = self.video_folder + file_name + ".mp4"
        tracks_path = self.tracks_folder + file_name + ".txt"
        clothes_path = self.clothes_folder + file_name + ".txt"
        processor = SingleClothesProcessor(video_path, tracks_path, clothes_path)
        self.runTread(processor, file_name, self.FOUND)

    def runTread(self, obj, name, type):
        self.counter += 1
        cur_num = self.counter % 20
        item = self.files[name][0]
        item.setFlags(Qt.NoItemFlags)
        self.threads[cur_num] = (obj, QThread())
        self.threads[cur_num][0].moveToThread(self.threads[cur_num][1])
        self.threads[cur_num][0].finished.connect(partial(self.end_thread, cur_num))
        self.threads[cur_num][0].percent.connect(partial(self.percentChange, item))
        self.threads[cur_num][1].started.connect(self.threads[cur_num][0].process)
        self.threads[cur_num][1].finished.connect(partial(self.release_thread, cur_num, item, name, type))
        self.threads[cur_num][1].start()


    def percentChange(self, item, percent):
       name = item.text().split("|", 1)[-1]
       item.setText(f"{percent}%|{name}")

    def end_thread(self, name):
        self.threads[name][1].quit()

    def release_thread(self, num, item, name, type):
        del self.threads[num]
        item.setText(f"{self.types[type]}{name}")
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        paths = FilesGetter.finedAllFiles(self.video_folder + name + ".mp4")
        self.files[name][1] = type
        self.files[name][2] = paths

    def endProcessing(self):
        for name, process in self.threads.items():
            process[0].setStopped()
            process[1].quit()
            process[1].wait()

    def copy(self, path_from):
        parts = list(Path(path_from).parts)
        file_name = parts[-1].rsplit('.', 1)[0]
        if file_name in self.files:
            self.parent.DisplayMsg(f"⚠ File '{file_name}' already added!")
            return
        path_to = os.path.join(self.video_folder, parts[-1])
        shutil.copy(path_from, path_to)
        paths = FilesGetter.finedAllFiles(path_to)
        self.addToList(file_name, self.LOADED, paths)

    def openFile(self, name):
        name = name.strip("❌⚠✅")
        ret = [np.array([]), "", dict(), ""]
        if name not in self.files:
            print("bug")
            return ret
        self.parent.openVideo(self.video_folder + name + ".mp4")
        if self.files[name][1] in (self.TRACKED, self.FOUND):
            self.parent.setTracksBtns(True)
            tracks_file = open(self.files[name][2][1], "r")
            tracks = self.parsTracks(tracks_file.readlines())
            tracks_file.close()
            ret[0] = tracks
            ret[1] = self.files[name][2][1]
            if self.files[name][1] == self.FOUND:
                clothes_file = open(self.files[name][2][2], "r")
                clothes = json.load(clothes_file)
                clothes_file.close()
                self.parent.setClothesBtns(True)
                ret[2] = clothes
                ret[3] = self.files[name][2][2]

        return ret

    def deleteFile(self, name):
        #TODO fix a lot of bugs
        name = name.strip("❌⚠✅")
        if name not in self.files:
            print("bug")
            return
        self.files[name][0] = self.filesListWidget.takeItem(self.filesListWidget.row(self.files[name][0]))
        if self.files[name][2][0] != "":
            self.files[name][2][0] += ".mp4"
        for path in self.files[name][2][:-1]:
            if path != "":
                try:
                    os.remove(path)
                except:
                    print("bug")
        del self.files[name]

    def hideAll(self):
        for file in self.files:
            if self.files[file][3]:
                self.files[file][0] = self.filesListWidget.takeItem(self.filesListWidget.row(self.files[file][0]))

    def displayWithClothes(self, checked_clothes):
        self.hideAll()
        for file in self.files:
            self.files[file][3] = False
            if self.files[file][1] == self.FOUND:
                clothes_file = open(self.files[file][2][2], "r")
                clothes = json.load(clothes_file)
                clothes_file.close()
                for track in clothes:
                    flagIn = any(map(lambda v: v in clothes[track], checked_clothes))
                    if flagIn:
                        self.files[file][3] = True
                        break
        self.showMatched()

    def showMatched(self):
        for file in self.files:
            if self.files[file][3]:
                self.filesListWidget.addItem(self.files[file][0])

    def matchAll(self):
        for file in self.files:
            self.files[file][3] = True





