# пусть это будет контроллером


import cv2
import numpy as np
from enum import Enum
import time
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QCursor

from MyWidgets import MyFilesListWidget, MyTracksListWidget, MyTimesListWidget, MyPersonWidget, MyDescriptionsListWidget, PersonDescriptionController
from processing import MyDB, create_db, crop, xywh_to_xyxy, TrackItem, config

from configWidget import MyConfigWindow

images_path = Path(config.global_config["application"]["images_folder_path"])
MAIN_DB = "./data/main.db"

class MyApp(QtWidgets.QMainWindow):

    def __init__(self):
        time1 = time.perf_counter()

        super().__init__()
        uic.loadUi('myapp.ui', self)

        if not os.path.exists(MAIN_DB):
            self._creat_db(MAIN_DB)
        self.myDB = MyDB(MAIN_DB)

        self.widgetVideo.post_init(self)
        self.times_list = MyTimesListWidget(self)
        self.files_list = MyFilesListWidget(self)
        self.tacks_list = MyTracksListWidget(self)  # Нужно объявлять после times_list
        self.person_controller = PersonDescriptionController(self)

        self.files_list.openFile.connect(self._buttonOpenFile_clicked)
        self.widgetVideo.frame_changed.connect(self.times_list.set_frame_num)
        self.times_list.click_time_change.connect(self.widgetVideo.set_frame_num)
        self.times_list.time_deleted.connect(self._set_track_state)
        self.tacks_list.track_state_changed.connect(self._set_track_state)
        self.widgetVideo.track_checked.connect(self._set_track_state)
        self.widgetVideo.track_to_find.connect(self._display_track_info)
        self.tacks_list.track_clothes_cliked.connect(self._display_track_info)
        self.db_open.triggered.connect(self._open_db_dialog)
        self.db_create.triggered.connect(self._creat_db_dialog)
        self.times_list.mistake_to_save.connect(self._save_mistake)
        self.times_list.time_item_cliked.connect(self._show_time_item_info)
        self.open_config.triggered.connect(self._open_config)

        self.config_window = MyConfigWindow()

        time2 = time.perf_counter()
        print("Let's start", time2-time1)

    def _buttonOpenFile_clicked(self, file):
        if file.path == "":
            return
        else:
            valid = file.check_self_valid()
            if not valid:
                # TODO написать ошибку для пользователя
                print(f"Файл '{file.path}' не может быть открыт")
            else:
                self.times_list.set_total_frame_num(file.frame_num)

            self.tacks_list.set_file(file)
            self.tacks_list.display()

            self.widgetVideo.set_video(file)  # Вызывать после того как

    def _set_track_state(self, track, state, redraw=True):
        if state:  # Добавление
            if self.sender() != self.tacks_list:
                self.tacks_list.set_track_state(track, state)
            if self.sender() != self.times_list:
                dist = self.person_controller.get_dist_to_track(track)
                self.times_list.add_track(track, dist, last=not redraw)
            if self.sender() != self.widgetVideo and redraw:
                self.widgetVideo.redraw()
        else:  # Удаление
            if self.sender() != self.tacks_list:
                self.tacks_list.set_track_state(track, state)
            if self.sender() != self.times_list:
                self.times_list.del_track(track)
            if self.sender() != self.widgetVideo and redraw:
                self.widgetVideo.redraw()

    def _display_track_info(self, track, img=None):
        if img is None:
            if track.midl_frame != -1:
                img = self._get_track_from_frame(track, track.midl_frame - 1)
                if img is None:
                    print(f"Не можем найти кадр {track.midl_frame}")
        self.person_controller.display_track_info(track, img)

    def _get_track_from_frame(self, track, frame_num):
        img = self.widgetVideo.get_full_frame_from(frame_num)
        if img is not None:
            bbox = self.myDB.get_track_bbox_from_frame(track, frame_num + 1)
            if len(bbox) < 1:
                return None
            else:
                assert len(bbox) == 1
                bbox = list(bbox[0])
                img = crop(img, xywh_to_xyxy(bbox), bbox_type='r')
                return img
        else:
            return None

    def get_track_by_id(self, track_id):
        return self.tacks_list.tracks[track_id]

    def set_found_person(self, matches):
        self.widgetVideo.clear()
        self.tacks_list.clear()
        self.files_list.set_matches(matches)

    def _set_db(self, path):
        self.myDB.set_db_file(path)
        self.files_list.read_saved_files()
        db_name = self.myDB.get_name()
        self.cur_images_path = images_path / db_name

        self.widgetVideo.clear()
        self.tacks_list.clear()
        self.person_controller.clear()
        self.files_list.clear()
        print("Файл бд загружен")

    def _open_db_dialog(self):
        current_file_name = \
            QtWidgets.QFileDialog.getOpenFileName(None, "Выберете файл базы данных",
                                                  "./data",
                                                  "*.db")[0]
        if current_file_name == "":
            return

        self._set_db(current_file_name)

    def _creat_db_dialog(self):
        current_file_name, cur_type = \
            QtWidgets.QFileDialog.getSaveFileName(None, "Создать базу",
                                                  "./data/new.db",
                                                  "*.db")

        if current_file_name == "":
            return
        if current_file_name[-3:] != ".db":
            current_file_name += ".db"

        self.files_list.set_availability(False)
        self._creat_db(current_file_name)
        self._set_db(current_file_name)
        self.files_list.set_availability(True)

    def _creat_db(self, db_path):
        create_db(db_path)

        # создание дополнительных папок
        db_name = Path(db_path).stem
        cur_path = images_path / db_name
        cur_path.mkdir(parents=True, exist_ok=True)

        query_dir = cur_path / "query"
        found_dir = cur_path / "found"
        query_dir.mkdir(exist_ok=True)
        found_dir.mkdir(exist_ok=True)

        print("База создана")

    def _save_mistake(self, track, dist, mistake_type):
        # Сбор текущей информации
        cur_frame_num = self.widgetVideo.get_cur_frame_num()
        frame_to_read = cur_frame_num  # или другой фрейм ТУТ МЕНЯТЬ ФРЕЙМ
        # проверить что на cur_frame есть этот трек
        track_img = self._get_track_from_frame(track, frame_num=frame_to_read)

        if track_img is None:
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setText('На текущем кадре должен присутсвовать трек!')
            error_dialog.exec()
            return
        track_info = {"out": False, "file_id": track.file.file_id,
                      "track_id": track.track_id, "frame_from": frame_to_read}

        # Сохранение информации
        # - поиск среди сохраненных
        search_img_id = self.person_controller.get_cur_person_img_id()
        track_img_id = self.myDB.search_saved_img_info(track_info)
        last_img_id = self.myDB.get_last_img_id()

        # - сохранение в базе
        assert search_img_id is not None
        if track_img_id is None:
            last_img_id += 1
            track_img_id = last_img_id
            self.myDB.save_img_info(track_info, track_img_id)
            img_folder = self.get_images_folder() / "found"
            cv2.imwrite(str(img_folder / (str(track_img_id) + ".png")), cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB))

        self.myDB.save_last_img_id(last_img_id)

        self.myDB.save_mistake(search_img_id, track_img_id, dist, mistake_type)

    def _open_config(self):
        self.config_window.show()

    def get_images_folder(self):
        return self.cur_images_path

    def _show_time_item_info(self, track, show):
        if show:
            search_img_id = self.person_controller.get_cur_person_img_id()
            match = self.myDB.get_track_mistakes(track, search_img_id)
            if match is not None:
                msg = "Было отмечено: "
                if match == 1:
                    msg += "ДАЛЬШЕ"
                else:
                    msg += "БЛИЖЕ"
                self.label_status_bar.setText(msg)
        else:
            self.label_status_bar.setText("")

    def reset_on_img_changed(self):
        self.files_list.reset()
        self.tacks_list.reset()

    def closeEvent(self, event):
        self.widgetVideo.release()
        self.files_list.release()
        print("Released")
