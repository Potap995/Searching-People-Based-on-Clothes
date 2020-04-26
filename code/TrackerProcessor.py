from deep_sort_2.deep.feature_extractor import Extractor
from deep_sort_2.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_2.sort.preprocessing import non_max_suppression
from deep_sort_2.sort.detection import Detection
from deep_sort_2.sort.tracker import Tracker

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import time


class SingleTrackerProcessor(QObject):
    finished = pyqtSignal()
    percent = pyqtSignal(int)

    min_confidence = 0.3
    nn_budget = 100
    max_cosine_distance = 0.2
    nms_max_overlap = 0.1
    n_init = 3
    max_iou_distance = 0.7
    max_age = 70

    def __init__(self, video_path, tracks_path):
        super().__init__()
        self.video_path = video_path
        self.tracks_path = tracks_path
        self.detector_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        # cur_path = os.getcwd()
        # self.tracker_path = os.path.join(cur_path, "\deep_sort_2\deep\checkpoint\ckpt.t7")
        self.tracker_path = \
            "D:\Programming\CourseWork_3\code\deep_sort_2\deep\checkpoint\ckpt.t7"
        self.video = None
        self.stopped = False

    @pyqtSlot()
    def process(self):
        self.percent.emit(0)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.detector_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.detector_name)
        predictor = DefaultPredictor(cfg)
        metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric, max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)
        extractor = Extractor(self.tracker_path, use_cuda=True)

        self.percent.emit(5)

        out_file = open(self.tracks_path, 'w')
        self.video = cv2.VideoCapture(self.video_path)
        counter = 0
        frames_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        current_percent = 0

        det_time = 0
        trac_time = 0
        timeAll1 = time.time()

        success, frame = self.video.read()
        while success and not self.stopped:
            counter += 1
            if current_percent != int((counter / frames_count) * 95):
                current_percent = int((counter / frames_count) * 95)
            self.percent.emit(current_percent + 5)

            time1 = time.time()
            outputs = predictor(frame)
            time2 = time.time()
            preds = self.getBboxs(outputs["instances"].to("cpu"))

            features = self.get_features(preds[:, :4].astype(np.int32), frame, extractor)
            bbox_tlwh = self.xyxy_to_xywh(preds[:, :4])
            detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(preds[:, 4]) if
                          conf > self.min_confidence]

            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)
            time3 = time.time()

            det_time += (time2 - time1)
            trac_time += (time3 - time2)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr().astype(np.int32)

                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    counter, track.track_id, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                      file=out_file)

            success, frame = self.video.read()


        timeAll2 = time.time()

        print(det_time, trac_time, det_time + trac_time)
        print(timeAll2 - timeAll1)
        out_file.close()
        self.video.release()
        if self.stopped:
            os.remove(self.tracks_path)
        else:
            self.finished.emit()

    @staticmethod
    def getBboxs(output):
        bboxs = output.pred_boxes[output.pred_classes == 0].tensor.numpy()
        scores = output.scores[output.pred_classes == 0].numpy()
        return np.concatenate((bboxs, scores.reshape(-1, 1)), axis=1)

    @staticmethod
    def get_features(bbox_xyxy, ori_img, extractor):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = extractor(im_crops)
        else:
            features = np.array([])
        return features

    @staticmethod
    def xyxy_to_xywh(bbox_xyxy):
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[:, 2:] -= bbox_xywh[:, :2]
        return bbox_xywh

    def setStopped(self):
        self.stopped = True

