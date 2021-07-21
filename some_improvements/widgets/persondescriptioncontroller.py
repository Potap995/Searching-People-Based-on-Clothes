from .mypersonwidget import MyPersonWidget
from .mydescriptionslistwidget import MyDescriptionsListWidget

from processing import PersonFinder, ClothesItem, rgb_to_hex, draw_on_img, TrackItem

from collections import defaultdict
import cv2
# from detectron2.utils.visualizer import Visualizer
from functools import partial


class PersonDescriptionController:

    def __init__(self, parent):
        self.controller = parent
        self.db = parent.myDB
        self.clothes_name = self.db.get_clothes_names()

        self.descriprions_widget = MyDescriptionsListWidget(parent, self)
        self.person_widget = MyPersonWidget(parent, self)

        self.person_finder = PersonFinder(self.db)
        future = self.person_finder.load_models()
        future.add_done_callback(self._models_loaded_callback)

        self.matches = defaultdict(list)
        self.found_dists = dict()
        self.reid_found_flag = False

    def set_track(self, img, track):
        self.clear()
        if img is not None:
            self.person_widget.set_person_img(img)

        self.controller.reset_on_img_changed()
        self.reid_found_flag = False
        self.descriprions_widget.set_person(track)

        person_info = self._get_person_info_to_save(track)
        id_ = self.db.search_saved_img_info(person_info)
        self.person_widget.set_img_id(id_)

    @staticmethod
    def _get_person_info_to_save(person_info):
        if isinstance(person_info, TrackItem):
            person_info = {"out": False, "file_id": person_info.file.file_id,
                           "track_id": person_info.track_id, "frame_from": person_info.midl_frame}
        elif isinstance(person_info, str):
            person_info = {"out": True, "path": person_info}
        else:
            person_info = None
            # raise ValueError("Пустое значение человека для поиска, такого не должно быть")

        return person_info

    def _save_cur_person_img(self):
        track = self.get_cur_person_info()
        person_info = self._get_person_info_to_save(track)
        assert person_info is not None
        last_img_id = self.db.get_last_img_id()
        img_folder = self.controller.get_images_folder() / "query"
        last_img_id += 1
        search_img_id = last_img_id
        search_img = self.get_cur_person_img()

        cv2.imwrite(str(img_folder / (str(search_img_id) + ".png")), cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB))
        self.db.save_img_info(person_info, search_img_id)
        self.db.save_last_img_id(last_img_id)
        self.person_widget.set_img_id(last_img_id)

    def find_by_reid(self, img):
        self.matches = dict()
        track = self.descriprions_widget.get_person()
        if isinstance(track, str):
            track = None

        if self.reid_found_flag:
            return None

        search_img_id = self.person_widget.get_img_id()
        if search_img_id is None:
            self._save_cur_person_img()

        matches, img = self.person_finder.find_by_reid(img, track)
        for match in matches:
            if match["type"] == 1:
                self.found_dists[(match["file"], match["id"])] = match["dist"]
            if match["file"] not in self.matches:
                self.matches[match["file"]] = {"match_type": ([], []), "min_dist": match["dist"]}
            self.matches[match["file"]]["match_type"][match["type"] - 1].append(match["id"])
            if self.matches[match["file"]]["min_dist"] > match["dist"]:
                self.matches[match["file"]]["min_dist"] = match["dist"]

        print(self.matches)
        self.controller.set_found_person(self.matches)
        self.reid_found_flag = True
        return img

    def find_and_display_clothes(self, img):
        clothes, img = self.person_finder.get_clothes_on_person(img)
        boxes, masks = clothes["bboxes"], clothes["masks"]
        labels = [self._get_clothes_name_by_id(id_) for id_ in clothes["classes"]]
        vis_img = draw_on_img(img, masks, boxes, labels, alpha=0.2)
        description = []
        clothes = (clothes["classes"], clothes["rgb_colors"], clothes["scores"])
        for ell in zip(*clothes):
            id_ = ell[0]
            color = rgb_to_hex(ell[1])
            score = ell[2]
            clothes = self._get_clothes_by_id(id_, color=color, score=score)
            description.append(clothes)

        self.descriprions_widget.set_person_description(description)
        return vis_img

    def find_by_clothes(self, description, clothes_num=3):
        self.matches = dict()
        matches = self.person_finder.find_by_clothes(description, clothes_num)
        for match in matches:
            if match["file"] not in self.matches:
                self.matches[match["file"]] = {"match_type": ([], []), "min_dist": match["dist"]}
            self.matches[match["file"]]["match_type"][match["type"] - 1].append(match["id"])

        self.controller.set_found_person(self.matches)

    def get_clothes_obj(self, name):
        clothes = self._get_clothes_by_name(name, score=1)
        return clothes

    def _get_clothes_name_by_id(self, id_):
        name = None
        for id_name in self.clothes_name:
            if id_ == id_name["clothes_id"]:
                name = id_name["name"]
        return name

    def _get_clothes_by_name(self, name, color=None, score=-1, hist=None, row_id=None):
        id_ = -1
        for id_name in self.clothes_name:
            if name == id_name["name"]:
                id_ = id_name["clothes_id"]
        return ClothesItem(id_, name, color, score, hist, db_id=row_id)

    def _get_clothes_by_id(self, id_, color=None, score=-1, hist=None, row_id=None):
        name = self._get_clothes_name_by_id(id_)
        return ClothesItem(id_, name if name is not None else "not_find", color, score, hist, db_id=row_id)

    def get_clothes_classes(self):
        return [clothes["name"] for clothes in self.clothes_name]

    def display_track_info(self, track, img=None):
        self.set_track(img, track)

        descriptions = []
        descriptions_db = self.db.get_track_clothes(track)
        for one_description in descriptions_db:
            id_ = one_description["clothes"]
            color = one_description["color"]
            score = one_description["score"]
            row_id = one_description["rowid"]
            clothes = self._get_clothes_by_id(id_, color, score, row_id=row_id)
            descriptions.append(clothes)
        self.descriprions_widget.set_person_description(descriptions)

    def _models_loaded_callback(self, future_):
        error = future_.exception()
        if error is None:
            print("Models should be available")
            self.person_widget.set_model_available(True)
        else:
            raise error

    def get_dist_to_track(self, track):
        id_ = track.track_id
        file_ = track.file.file_id
        if (file_, id_) in self.found_dists:
            return self.found_dists[(file_, id_)]
        return None

    def save_description_changes(self, changes, person):
        self.db.del_clothes(changes["del"])

        self.db.del_clothes(changes["change"])
        new_rowids = self.db.add_clothes(changes["change"], person)
        for desc, rowid in zip(changes["change"], new_rowids):
            desc.db_id = rowid

        new_rowids = self.db.add_clothes(changes["add"], person)
        for desc, rowid in zip(changes["add"], new_rowids):
            desc.db_id = rowid

    def get_cur_person_img(self):
        return self.person_widget.get_cur_img()

    def get_cur_person_info(self):
        return self.descriprions_widget.get_person()

    def get_cur_person_img_id(self):
        id_ = self.person_widget.get_img_id()
        return id_

    def clear(self):
        track = self.descriprions_widget.get_person()
        if track:
            self.person_widget.clear()
            self.descriprions_widget.clear_description()
            self.descriprions_widget.clear_person()
