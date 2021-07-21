from .models import get_extractor_fastreid, get_detectron2_model, ClothesDetectron
from tqdm import tqdm
import numpy as np
import torch
import json
import cv2
import config
from collections import defaultdict
import concurrent.futures

from .support_functions import normalize, crop


class PersonFinder:

    def __init__(self, db):
        self.db = db
        self.model_reid = None
        self.model_person = None
        self.clothes_detector = None
        # self._load_models()

    def load_models(self):
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(self._load_models)
        return future

    def _load_models(self):
        # Может подвесить работу, нужно переписать в поток
        conf = {"detectron2": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                "detectron2_min_confidence": 0.7,
                "detectron2_nms_thresh": 0.6}
        self.model_reid = get_extractor_fastreid(conf=conf)
        # self.model_person = get_detectron2_model(conf=conf)
        self.clothes_detector = ClothesDetectron()
        print("Models Loaded")

    def unload_models(self):
        del self.model_reid
        #del self.model_person
        del self.clothes_detector
        # pass

    def find_by_clothes(self, description, matches_num):

        cur_clothes_ids = [description[i].get_clothes_id() for i in range(len(description))]

        finded_persons = []

        cur_id = -1
        cur_file = -1
        cur_description = []

        for db_clothes in tqdm(self.db.get_all_clothes()):
            if db_clothes["person_id"] == cur_id and db_clothes["file_id"] == cur_file:
                cur_description.append(db_clothes)
            else:
                finded_ids_files = self.check_clothes_similarity(cur_clothes_ids, cur_description, matches_num)
                if finded_ids_files:
                    finded_persons.append(finded_ids_files)

                cur_id = db_clothes["person_id"]
                cur_file = db_clothes["file_id"]
                cur_description = [db_clothes]

        # finded_persons = list(filter(None, finded_persons))
        print(finded_persons)
        return finded_persons

    @staticmethod
    def check_clothes_similarity(cur_clothes_ids, candidates_description, matches_num):
        counter = 0
        candidates_ids_files = None

        for one_description in candidates_description:
            if one_description["clothes"] in cur_clothes_ids:
                counter += 1
                candidates_ids_files = {'file': one_description["file_id"], 'id': one_description["person_id"], 'type': 1, "dist": None}
        if counter >= matches_num:
            return candidates_ids_files
        else:
            return None

    def get_main_person(self, img):
        def _get_area(_bbox):
            return (_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])

        # image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        # print(image.shape)
        # inputs = [{"image": image, "height": img.shape[0], "width": img.shape[1]}]  # inputs is ready

        # with torch.no_grad():
        #     outputs = self.model_person(inputs)
        #
        # res = outputs[0]["instances"].to("cpu").get_fields()
        # pred_classes = res["pred_classes"].numpy()
        # pred_boxes = res["pred_boxes"].tensor.numpy()[pred_classes == 0]
        # max_box = np.zeros((4,))
        # for bbox in pred_boxes:
        #     if _get_area(bbox) > _get_area(max_box):
        #         max_box = bbox
        #     # cv2.rectangle(img, tuple((int(bbox[0]), int(bbox[1]))),
        #     #               tuple((int((bbox[2])), int(bbox[3]))),
        #     #               (255, 0, 0), 3)
        # img = crop(img, max_box.astype(int), 10)
        return np.copy(img)

    def get_clothes_on_person(self, img):
        ret_img = np.array(img)
        # ret_img = self.get_main_person(ret_img)
        clothes = self.clothes_detector(ret_img)
        return clothes, ret_img

    def find_by_reid(self, img, track):
        finded_persons = []
        finded_ids = set()
        finded_fileid = set()
        files_ids = defaultdict(set)

        ret_img = np.array(img)
        # ret_img = self.get_main_person(ret_img)
        vector_find = normalize(self.model_reid([img[:, :, ::-1]])[0])  # RGB->BGR

        # данный участок не нужен
        if track is not None:
            info_from_db = self.db.get_track_vector_on_file(track)
            if len(info_from_db) > 1:
                print("!!!!!Много векторов нашли вместо 1!!!!!")
            else:
                vector, radius = info_from_db[0]
                vector_from_db = np.frombuffer(vector, dtype=np.float32)
                print("Vector from db", vector_from_db)
        else:
            print("Image not from db")
        # if track is not None:
        #     vector_find = vector_from_db
        # конец участка

        for db_person in tqdm(self.db.get_all_vectors()):
            file, id_, vector, radius = db_person

            files_ids[file].add(id_)

            vector = np.frombuffer(vector, dtype=np.float32)
            assert np.allclose(np.linalg.norm(vector), 1.), "Normalization failure"
            dist = np.linalg.norm(vector_find - vector)
            if dist < 0.93:  # < 0.975
                finded_ids.add(id_)
                if (file, id_) not in finded_fileid:
                    finded_persons.append({'file': file,
                                           'id': id_,
                                           'type': 1,
                                           "dist" : float(dist)
                                           })
                    finded_fileid.add((file, id_))
        finded_persons.sort(key=lambda x: x["dist"])
        # finded_persons = finded_persons[:5]
        for match in finded_persons:
            files_ids[match["file"]].discard(match["id"])

        # for file, ids in files_ids.items():
        #     ids = ids.intersection(finded_ids)
        #     for id_ in ids:
        #         finded_persons.append({'file': file, 'id': id_, 'type': 2})
        print(json.dumps(finded_persons[:5], indent=4))
        return finded_persons, ret_img
