from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PIL import Image
from collections import Counter
import torchvision

import torch
import numpy as np
import cv2
import json
import os


from pprint import pprint



class SingleClothesProcessor(QObject):
    finished = pyqtSignal()
    percent = pyqtSignal(int)


    def __init__(self, path_video, path_tracks, clothes_path):
        super().__init__()
        self.path_video = path_video
        self.path_tracks = path_tracks
        self.clothes_path = clothes_path
        # cur_path = os.getcwd()
        # self.model_path = os.path.join(cur_path, '\data\model_new_all.pt')
        self.model_path = 'D:\Programming\CourseWork_3\code\data\model_new_all.pt'
        self.cats_path = "D:\Programming\CourseWork_3\code\data\categories.txt"
        self.types = {
            "1": "top",
            "2": "bottom",
            "3": "all",
            "4": "res"
        }
        self.stopped = False

    @pyqtSlot()
    def process(self):
        self.percent.emit(0)

        model = torchvision.models.resnext101_32x8d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=1024, bias=True)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(in_features=1024, out_features=50, bias=True)
        ).cuda()
        model.load_state_dict(torch.load(self.model_path))
        model.cuda()
        model.eval()

        self.percent.emit(5)

        video = cv2.VideoCapture(self.path_video)
        sm = torch.nn.Softmax(dim=1)

        clothes_dict = dict()
        categories = self.getCategories()

        counter = 0
        frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        current_percent = 0

        tracks_file = open(self.path_tracks, "r")
        tracks = self.parsTracks(tracks_file.readlines())
        tracks_file.close()

        success, frame = video.read()
        while success and not self.stopped:
            cur_tracks = tracks[tracks[:, 0] == counter]

            for track in cur_tracks:
                x1, x2 = max(0, track[2]), max(track[2] + track[4], frame.shape[1])
                y1, y2 = max(0, track[3]), max(track[3] + track[5], frame.shape[0])
                # img_top = frame[y1:(y1 + int((y2 - y1) * 2 / 3)), x1:x2]
                # img_bottom = frame[(y1 + int((y2 - y1) * 1 / 3)):y2, x1:x2]
                # img_all = frame[y1:y2, x1:x2]
                # img_top = self.getProperImg(img_top)
                # img_bottom = self.getProperImg(img_bottom)
                # img_all = self.getProperImg(img_all)
                #
                # img_top = torch.tensor(img_top, dtype=torch.float32, device='cuda:0').reshape(1, 224, 224, 3).permute(0, 3, 1, 2).contiguous()
                # img_top = img_top / 256.0 - 0.5
                # img_bottom = torch.tensor(img_bottom, dtype=torch.float32, device='cuda:0').reshape(1, 224, 224, 3).permute(0, 3, 1, 2).contiguous()
                # img_bottom = img_bottom / 256.0 - 0.5
                # img = torch.tensor(img, dtype=torch.float32, device='cuda:0').reshape(1, 224, 224, 3).permute(0, 3, 1, 2).contiguous()
                # img_all = img_all / 256.0 - 0.5

                img = frame[y1:y2, x1:x2]
                img = self.getProperImg(img)

                img = torch.tensor(img, dtype=torch.float32, device='cuda:0').reshape(1, 224, 224, 3).permute(0, 3, 1, 2).contiguous()
                img = img / 256.0 - 0.5


                probs = sm(model(img))
                probs, idxs = probs.sort(descending=True)
                probs = probs.detach().cpu().numpy()[0]
                idxs = idxs.detach().cpu().numpy()[0]

                track_id = str(track[1])
                if track_id not in clothes_dict:
                    clothes_dict[track_id] = dict()
                    clothes_dict[track_id][str(-1)] = 0
                    for _, type in self.types.items():
                        clothes_dict[track_id][type] = Counter()

                for i in range(6):
                    cat = categories[idxs[i]][0]
                    type = categories[idxs[i]][1]
                    clothes_dict[track_id][type][cat] += probs[i]
                clothes_dict[track_id][str(-1)] += 1

            counter += 1
            if current_percent != int((counter / frames_count) * 95):
                current_percent = int((counter / frames_count) * 95)
            self.percent.emit(current_percent + 5)

            success, frame = video.read()

        video.release()
        if not self.stopped:
            dict_out = dict()
            res = ["top", "bottom", "all"]
            for track, types in clothes_dict.items():
                dict_out[track] = []
                cur_types = dict(zip(res, list(zip([0] * len(res), [""] * len(res)))))

                for type in res:
                    for clothes, percent in types[type].items():
                        if percent > cur_types[type][0]:
                            cur_types[type] = [percent, clothes]
                if cur_types["all"][0] * 2 > cur_types["bottom"][0] + cur_types["top"][0]:
                    dict_out[track] = [cur_types["all"][1]]
                else:
                    dict_out[track] = [cur_types["top"][1], cur_types["bottom"][1]]

            file_out = open(self.clothes_path, "w")
            file_out.write(json.dumps(dict_out))
            file_out.close()
            self.finished.emit()


    @staticmethod
    def getProperImg(img_arr):
        TARGET_SIZE = (224, 224)

        result = np.ones((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8) * 25
        try:
            img = Image.fromarray(img_arr)
        except:
            print(img.shape)
            print(img)
        img.thumbnail(TARGET_SIZE)
        offset = (np.array(TARGET_SIZE) - np.array(img.size)) // 2
        result[offset[1]: offset[1] + img.size[1], offset[0]: offset[0] + img.size[0]] = np.array(img)
        return result

    @staticmethod
    def parsTracks(tracks):
        tracks_list = []
        for line in tracks:
            line = line.strip().split(",")
            line = list(map(int, map(float, line)))
            tracks_list.append(line)

        return np.array(tracks_list)

    def getCategories(self):
        cats_file = open(self.cats_path, 'r').readlines()
        categories = []

        for ln in cats_file:
            cur = list(filter(None, ln[:-1].split(' ')))
            categories.append([cur[0], self.types[cur[1]]])

        return categories

    def setStopped(self):
        self.stopped = True

