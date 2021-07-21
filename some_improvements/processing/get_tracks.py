import sys
import os
import configparser

CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))

# for path in sys.path:
#     print(path)

if __name__ == "__main__":
    raise Exception("Не запускается самостоятельно")
    # sys.path.append(os.path.join(CUR_FOLDER, '../../'))  # нужно для processing

import time
from threading import Thread, Event
from queue import Queue as t_Queue
import logging
import argparse
import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm

from .models import get_yolo_model, get_detectron2_model, get_detectron_clothes, \
    get_tracker, get_extractor_fastreid, get_extractor_base, get_extractor_torchreid, \
    Detection

from support_functions import normalize, xyxy_to_xywh, rgb_to_hex

from .config import global_config

# from motmetrics.distances import iou_matrix


conf = {"detectron2": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "detectron2_min_confidence": 0.7,
        "detectron2_nms_thresh": 0.6,
        "detectorn2_path_to_model": "",
        "tracker_path": os.path.join(CUR_FOLDER, 'deep_sort/deep/checkpoint/ckpt.t7'),
        "max_dist": 0.3,
        "yolo_min_confidence": 0.6,  # .4
        "yolo_nms_max_overlap": 0.6,  # .6
        "max_iou_distance": 0.7,
        "max_age": 80,
        "n_init": 3,
        "nn_budget": 200,
        "class": 0,
        "yolo_width": 512,
        "yolo_height": 320,
        "detectron_width": 704,
        "detectron_height": 512,
        "byteslen": 8192,
        "min_side": 5}

config = global_config
config = config["processing"]


def parse_args():
    def size(s):
        try:
            width, height = map(int, s.split("x"))
            return width, height
        except:
            raise argparse.ArgumentTypeError("Coordinates must be like 200x100")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video", help="Path to input video file",
        default=None)
    parser.add_argument(
        "--size", help="Width and height to resize video. Example: --size 200x100",
        type=size
    )
    parser.add_argument(
        "--out_tracks", help="Path to tracks output .txt file",
        default=None)
    parser.add_argument(
        "--out_vectors", help="Path to vectors output .txt file",
        default=None)
    return parser.parse_args()


def get_video_cap(file_name):
    video_cap = cv2.VideoCapture(file_name)
    return video_cap


def get_name(path):
    res = path.split("/")
    res = res[-1].split(".")
    return res[0]


class StopIterFlag:
    def __init__(self):
        pass


class AsyncIterator_T(Thread):
    def __init__(self, iterator, max_size=5, name=""):
        super().__init__(daemon=True)
        self._name = name
        self._queue = t_Queue(maxsize=max_size)
        self._iter = iterator
        self._running = True

    def run(self) -> None:
        self._running = True
        for item in self._iter:
            # logging.debug(f"put {self._name}")
            self._queue.put(item, block=True)
        self._queue.put(StopIterFlag())
        # logging.info("Stop running")
        self._running = False
        self._queue.join()

    def __iter__(self):
        return self

    def __next(self):
        # logging.debug(f"Try get nex iter {self._name}")
        running, not_empty = self._running, not self._queue.empty()
        if running or not_empty:
            to_return = self._queue.get(block=True)
            self._queue.task_done()
            if isinstance(to_return, StopIterFlag):
                raise StopIteration()
            return to_return
        else:
            raise StopIteration()

    def __len__(self):
        return len(self._iter)

    __next__ = __next


class VideoInputGenerator:
    def __init__(self, video_cap, size=(-1, -1)):
        self._video = video_cap
        self._size = size
        self._resize = False
        if size[0] > 1 or size[1] > 1:
            self._resize = True
        self._dur = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._time = 0

    def __len__(self):
        return int(self._dur)

    def __iter__(self):
        return self

    def __next__(self):
        logging.debug(f"reader")
        t1 = time.perf_counter()
        ok, frame = self._video.read()
        if not ok:
            raise StopIteration()

        if self._resize:
            frame = cv2.resize(frame, self._size)

        t2 = time.perf_counter()
        self._time += t2 - t1
        return frame

    def get_time(self):
        return self._time

    def get_size(self):
        return int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))


class BatchGenerator:
    def __init__(self, video_cap, batch_size=4, size=(-1, -1)):
        self._iter = video_cap
        self._batch_size = batch_size
        self._resize = False
        if size[0] > 1 and size[1] > 1:
            self._size = size
            self._resize = True

        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter) // self._batch_size + (1 if self._iter % self._batch_size else 0)

    def __next__(self):
        logging.debug(f"batch")
        t1 = time.perf_counter()
        batch = []
        batch_sized = []
        for i in range(self._batch_size):
            try:
                frame = next(self._iter)
            except StopIteration:
                if len(batch) > 0:
                    t2 = time.perf_counter()
                    self._time += t2 - t1
                    return np.array(batch), np.array(batch_sized)
                else:
                    raise StopIteration()
            else:
                batch.append(frame)
                if self._resize:
                    sized = cv2.resize(frame, self._size)
                else:
                    sized = frame
                batch_sized.append(sized)

        t2 = time.perf_counter()
        self._time += t2 - t1

        return np.array(batch), np.array(batch_sized)

    def get_time(self):
        return self._time


class UnbatchGenerator:
    def __init__(self, iter, length):
        self._iter = iter
        self._length = length
        self._batch = ([], [])
        self._cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        logging.debug(f"unbatch")
        self._cur += 1
        if len(self._batch[0]) <= self._cur:
            self._batch = next(self._iter)
            self._cur = 0
        return self._batch[0][self._cur], self._batch[1][self._cur]

    def __len__(self):
        return self._length


class DetectorGenerator:
    def __init__(self, detector, video_iterator, detect_f, nms_f=None):
        self._iter = video_iterator
        self._detector = detector
        self._do_detect = detect_f
        self._do_nms = nms_f
        self._time_detect = 0
        self._time_nms = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        logging.debug(f"detector")
        frames, sized = next(self._iter)
        t1 = time.perf_counter()
        boxes = self._do_detect(self._detector, sized)
        t2 = time.perf_counter()
        if self._do_nms is not None:
            boxes = self._do_nms(boxes)
        t3 = time.perf_counter()
        self._time_detect += t2 - t1
        self._time_nms += t3 - t2
        # print(boxes.shape)
        return frames, boxes  # batch, 2

    def get_time(self):
        return self._time_detect, self._time_nms


class NMSGenerator:
    def __init__(self, detector_iterator, nms_f):
        self._iter = detector_iterator
        self._do_nms = nms_f
        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        logging.debug(f"nms")
        frames, detections = next(self._iter)
        t1 = time.perf_counter()
        boxes = self._do_nms(detections)
        t2 = time.perf_counter()
        self._time += t2 - t1
        return frames, boxes

    def get_time(self):
        return self._time


def get_features(bbox_xyxy, ori_img, extractor):
    im_crops = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = box
        im = ori_img[y1:y2, x1:x2]
        im_crops.append(im)
    if im_crops:
        logging.debug(f"extractor func")
        features = extractor(im_crops)
        # print(len(features), len(features[0]), features)
    else:
        features = np.array([])
    return features


class TracksGenerator_Extractor:
    def __init__(self, detect_gen, size, extractor):
        self._iter = detect_gen
        self._size = size
        self._extractor = extractor
        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        logging.debug(f"tracker_extractor")
        frame, output = next(self._iter)

        t1 = time.perf_counter()

        to_size = np.array([self._size[0], self._size[1], self._size[0], self._size[1]], dtype=int)
        scores = output[1]
        bboxes = output[0]
        bboxes[bboxes < 0] = 0
        bboxes[bboxes > 1] = 1
        res1 = len(scores)

        mask_1 = (bboxes[:, 2] - bboxes[:, 0]) * self._size[0] > conf["min_side"]
        mask_2 = (bboxes[:, 3] - bboxes[:, 1]) * self._size[1] > conf["min_side"]
        mask = mask_1 & mask_2
        bboxes = bboxes[mask]
        scores = scores[mask]
        res2 = len(scores)

        if res2 < res1:
            print(f"Deleted some tracks because side < {conf['min_side']}")

        features = get_features((bboxes * to_size).astype(np.int64), frame, self._extractor)
        bbox_tlwh = xyxy_to_xywh(bboxes)

        detections = [Detection(bbox_tlwh[i], scores[i], features[i]) for i in range(len(scores))]

        t2 = time.perf_counter()
        self._time += t2 - t1

        return frame, detections

    def get_time(self):
        return self._time


class TracksGenerator_Tracker:
    def __init__(self, extractor_gen, tracker):
        self._iter = extractor_gen
        self._tracker = tracker
        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        logging.debug(f"tracker_Tracker")
        frame, detections = next(self._iter)

        t1 = time.perf_counter()

        self._tracker.predict()
        self._tracker.update(detections)

        tracks = self._tracker.tracks

        bboxes = []
        ids = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox_tlwh = track.to_tlwh()
            bbox_tlwh[bbox_tlwh < 0] = 0
            bbox_tlwh[bbox_tlwh > 1] = 1
            bbox_tlbr = bbox_tlwh
            bbox_tlbr[2:] = bbox_tlwh[:2] + bbox_tlwh[2:]

            bboxes.append(bbox_tlbr)
            ids.append(track.track_id)

        t2 = time.perf_counter()
        self._time += t2 - t1
        return ids, bboxes, detections, frame

    def get_time(self):
        return self._time


class ReIDGenerator:

    def __init__(self, tracker_gen, vectors_file, size=(1, 1), extractor=None):
        self._iter = tracker_gen
        self._vectors_file = vectors_file
        self._extractor = extractor
        self._to_size = np.array([size[0], size[1], size[0], size[1]], dtype=int)
        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        ids, bboxes, detections, frame = next(self._iter)
        t1 = time.perf_counter()
        reid_vectors = []

        if self._vectors_file and len(bboxes) > 0:
            trac_bboxes = np.array(bboxes)
            reid_vectors = get_features((trac_bboxes * self._to_size).astype(np.int64), frame, self._extractor)

        t2 = time.perf_counter()
        self._time += t2 - t1
        return ids, bboxes, reid_vectors, frame

    def get_time(self):
        return self._time


def get_clothes(bbox_xyxy, ori_img, detector):
    im_crops = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = box
        im = ori_img[y1:y2, x1:x2]
        im_crops.append(im)
    clothes_all = []
    if im_crops:
        for img in im_crops:
            classes, colors, scores, hists = [], [], [], []
            if config.getboolean("clothes_detection"):
                clothes = detector(img)
                classes = clothes["classes"]
                colors = clothes["rgb_colors"]
                scores = clothes["scores"]
                hists = clothes["hue_hists"]
            clothes_all.append(list(zip(classes, colors, scores, hists)))
    return clothes_all


class ClothesGenerator:

    def __init__(self, reid_gen, clothes_file, size=(1, 1), detector=None):
        self._iter = reid_gen
        self._detector = detector
        self._clothes_file = clothes_file
        self._to_size = np.array([size[0], size[1], size[0], size[1]], dtype=int)
        self._time = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        ids, bboxes, reid_vectors, frame = next(self._iter)
        t1 = time.perf_counter()
        clothes = []

        if self._clothes_file and self._detector and len(bboxes) > 0:
            trac_bboxes = np.array(bboxes)
            clothes = get_clothes((trac_bboxes * self._to_size).astype(np.int64), frame, self._detector)

        t2 = time.perf_counter()
        self._time += t2 - t1
        return ids, bboxes, reid_vectors, clothes

    def get_time(self):
        return self._time


class FileWriter:
    def __init__(self, tracker_gen, out_tracks_file, out_vectors_file, out_clothes_file, size):
        self._iter = tracker_gen
        self._tracks_file = out_tracks_file
        self._vectors_file = out_vectors_file
        self._clothes_file = out_clothes_file
        self._time = 0
        self._counter = 0
        self._frame_size = size
        self._byteslen = conf["byteslen"]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        logging.debug(f"print output")
        ids, bboxes, reid_vectors, clothes = next(self._iter)
        t1 = time.perf_counter()
        self._counter += 1
        if len(ids) != len(bboxes) or \
                self._clothes_file is not None and len(ids) != len(clothes) or \
                self._vectors_file is not None and len(ids) != len(reid_vectors):
            print(f"Ids: {len(ids)}, Bboxes: {len(bboxes)}, ReID: {len(reid_vectors)}, Clothes: {len(clothes)}")
            raise Exception("Колво треков и информации о них не совпадает")

        for i in range(len(ids)):
            track_id = ids[i]
            bbox = bboxes[i]

            top = bbox[0] * self._frame_size[0]
            left = bbox[1] * self._frame_size[1]
            width = (bbox[2] - bbox[0]) * self._frame_size[0]
            height = (bbox[3] - bbox[1]) * self._frame_size[1]
            print(
                f"{self._counter},{track_id},{top},{left},{width},{height},1,-1,-1,-1",
                file=self._tracks_file)

            if self._vectors_file:
                feature = normalize(reid_vectors[i])
                feature = feature.astype(np.float32)
                s = feature.tobytes()
                if len(s) != self._byteslen:
                    print("Непридвиденная длинна вектора в байтах")
                self._vectors_file.write(track_id.to_bytes(4, byteorder='big'))
                self._vectors_file.write(s)

            if self._clothes_file:
                track_clothes = clothes[i]
                line = str(track_id) + "|"
                for single_clothes in track_clothes:
                    clothes_id = single_clothes[0]
                    color = rgb_to_hex(single_clothes[1])
                    score = single_clothes[2]
                    hist = np.array(list(map(float, single_clothes[3])), dtype=float)
                    hist_ = list(np.around(hist, 4))
                    line += f"{clothes_id}:{color}:{score}:{hist_};"
                line = line.strip(';')
                print(line, file=self._clothes_file)
        # for ind, det in enumerate(features):
        #     bbox = det.tlwh
        #     vector = det.features
        #     det_bboxes.append(bbox)
        #
        # self._misses += len(trac_bboxes) - len(det_bboxes)
        # iou_matrix_ = iou_matrix(det_bboxes, trac_bboxes)
        # if iou_matrix_.shape[0] > 0:
        #     ids_ = iou_matrix_.argmin(axis=1)
        #     values_ = iou_matrix_.min(axis=1)

        t2 = time.perf_counter()
        self._time += t2 - t1

        return True

    def get_time(self):
        return self._time


def detect_func_yolo(detector, imgs):
    detections = make_detect_yolo(detector, imgs)
    bboxes = post_processing_yolo(detections)
    return bboxes


def make_detect_yolo(detector, imgs):
    if type(imgs) == np.ndarray and len(imgs.shape) == 3:  # cv2 image
        imgs = torch.from_numpy(imgs.transpose(2, 0, 1)).cuda().float().half().div(255.0).unsqueeze(0)
    elif type(imgs) == np.ndarray and len(imgs.shape) == 4:  # batch
        imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).cuda().float().half().div(255.0)
    else:
        print(imgs.shape)
        print("unknow image type")
        exit(-1)

    imgs = imgs.cuda()
    with torch.no_grad():
        logging.debug(f"detector func")
        output = detector(imgs)

    output[0] = output[0].detach().float()
    output[1] = output[1].detach().float()
    return output


def post_processing_yolo(output):
    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf, max_id = torch.max(confs, dim=2)
    logging.debug(f"NMS func")
    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = torch.logical_and(max_conf[i] > conf["yolo_min_confidence"], max_id[i] == conf["class"])
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]

        # torch.from_numpy(
        keep = torchvision.ops.nms(l_box_array, l_max_conf, conf["yolo_nms_max_overlap"])

        ll_box_array = l_box_array[keep, :]
        ll_max_conf = l_max_conf[keep]

        bboxes_batch.append([ll_box_array.cpu().numpy(), ll_max_conf.cpu().numpy()])

    return bboxes_batch


def make_detect_detectron(detector, imgs):
    assert imgs.shape[0] == 1, "Работаем только с 1 изображением"
    image = torch.as_tensor(imgs[0].astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": 1, "width": 1}]  # inputs is ready

    with torch.no_grad():
        outputs = detector(inputs)

    res = outputs[0]["instances"].to("cpu").get_fields()
    pred_classes = res["pred_classes"].numpy()
    pred_boxes = res["pred_boxes"].tensor.numpy()[pred_classes == 0]
    scores = res["scores"].numpy()[pred_classes == 0]
    output = [[]]
    output[0].append(pred_boxes)
    output[0].append(scores)
    return output


class TrackingProcess:

    def __init__(self, args, stop_event=Event()):
        self.stop_event = stop_event
        self.args = args
        self._percent = 0
        self._frequency = 100

    def run(self):
        args = self.args

        out_tracks_file = open(args["out_tracks"], 'w')
        if args["out_vectors"] is not None:
            out_vectors_file = open(args["out_vectors"], 'wb')
        else:
            out_vectors_file = None
        if args["out_clothes"] is not None:
            out_clothes_file = open(args["out_clothes"], 'w')
        else:
            out_clothes_file = None
        video_cap = get_video_cap(args["video"])

        start = time.perf_counter()

        video_gen = VideoInputGenerator(video_cap, size=(-1, -1))
        capacity = len(video_gen)
        video_size = video_gen.get_size()
        print(video_size)
        video_gen_a = AsyncIterator_T(video_gen, max_size=64)
        video_gen_a.start()

        if args["detector"] == "yolo":
            print("Using yolo")
            cur_size = (conf["yolo_width"], conf["yolo_height"])
            # cur_size = (1856, 1088)
            batch_gen = BatchGenerator(video_gen_a, batch_size=16, size=cur_size)
        else:
            batch_gen = BatchGenerator(video_gen_a, batch_size=1)
        batch_iterator = batch_gen
        # batch_gen_a = AsyncIterator_T(batch_gen, max_size=3, name="batch_gen")
        # batch_gen_a.start()
        # batch_iterator = batch_gen_a

        if args["detector"] == "yolo":
            model = get_yolo_model(conf)
            detections_gen = DetectorGenerator(model, batch_iterator, make_detect_yolo, post_processing_yolo)
        else:
            model = get_detectron2_model(conf)
            detections_gen = DetectorGenerator(model, batch_iterator, make_detect_detectron)
        unbatch_gen = UnbatchGenerator(detections_gen, capacity)
        detections_iterator = unbatch_gen
        detections_gen_a = AsyncIterator_T(unbatch_gen, max_size=3, name="detections_gen")
        detections_gen_a.start()
        detections_iterator = detections_gen_a

        if args["extractor"] == "fast_reid":
            # conf["fast_reid_conf"] = "./data/rud_1/config.yaml"
            # conf["fast_reid_weigths"] = "./data/rud_1/model_final.pth"
            extractor = get_extractor_fastreid(conf)
        elif args["extractor"] == "torch_reid":
            extractor = get_extractor_torchreid(conf)
        else:
            extractor = get_extractor_base(conf)
        tracker = get_tracker(conf)
        tracks_gen_ext = TracksGenerator_Extractor(detections_iterator, video_size, extractor)
        tracks_gen = TracksGenerator_Tracker(tracks_gen_ext, tracker)
        tracks_iterator = tracks_gen
        tracks_gen_a = AsyncIterator_T(tracks_gen, max_size=3, name="tracker_gen")
        tracks_gen_a.start()
        tracks_iterator = tracks_gen_a

        if config.getboolean("clothes_detection"):
            clothes_detector = get_detectron_clothes(conf)
        else:
            clothes_detector = None
        reid_gen = ReIDGenerator(tracks_iterator, out_vectors_file is not None, video_size, extractor)
        clothes_gen = ClothesGenerator(reid_gen, out_clothes_file is not None, video_size, clothes_detector)
        info_iterator = clothes_gen

        file_writer = FileWriter(info_iterator, out_tracks_file, out_vectors_file, out_clothes_file, video_size)
        file_writer_a = AsyncIterator_T(file_writer, max_size=2, name="file_writer")
        file_writer_a.start()

        for counter, ret in enumerate(tqdm(file_writer_a, position=0, leave=True)):
            if self.stop_event.is_set():
                return -1.0
            self._percent = int(counter/capacity * self._frequency)

        out_tracks_file.close()
        if args["out_vectors"] is not None:
            out_vectors_file.close()
        if args["out_clothes"] is not None:
            out_clothes_file.close()
        video_cap.release()

        end = time.perf_counter()

        video_gen_time = video_gen.get_time()
        detections_gen_time, nms_gen_time = detections_gen.get_time()
        tracks_gen_ext_time = tracks_gen_ext.get_time()
        tracks_gen_time = tracks_gen.get_time()
        write_time = file_writer.get_time()
        reid_time = reid_gen.get_time()
        clothes_time = clothes_gen.get_time()

        print("---------------------------------------")
        print(f"Processing time = {end - start}")
        print()
        print(f"Video time = {video_gen_time}")
        print(f"Detections time = {detections_gen_time}")
        print(f"NMS time = {nms_gen_time}")
        print(f"Extractor time = {tracks_gen_ext_time}")
        print(f"Tracking time = {tracks_gen_time}")
        print(f"ReID time = {reid_time}")
        print(f"Clothes time = {clothes_time}")
        print(f"Write time = {write_time}")
        print(
            f"All sum time = {video_gen_time + detections_gen_time + nms_gen_time + tracks_gen_time + tracks_gen_ext_time + write_time + reid_time + clothes_time}")
        return end - start

    @property
    def percent(self):
        return self._percent


def run_tracking(args):
    process = TrackingProcess(args)
    return process.run()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(relativeCreated)6d %(threadName)s %(thread)d %(message)s"
)
logging.disable(logging.DEBUG)


def main(args):
    # set_start_method('spawn')
    args = vars(args)
    # args["video"] = "/media/main-disk/Programming/Work/Tracker/data/MOT/video/KITTI-13.mp4"
    args["video"] = "data/video.mp4"
    args["size"] = None
    name = get_name(args["video"])
    args["out_vectors"] = f"output/{name}_out_vectors.txt"
    # args["out_vectors"] = None
    args["out_tracks"] = f"output/{name}_out_tracks.txt"
    # args["out_clothes"] = f"output/{name}_out_clothes.txt"
    args["out_clothes"] = None
    # args["extractor"] = "default"
    args["extractor"] = "fast_reid"
    args["detector"] = "yolo"
    # args["detector"] = "detectron"
    run_tracking(args)


if __name__ == "__main__":
    raise Exception("Не запускается самостоятельно, нужно правильно указать относительные пути до весов моделей")
    # parsed_args = parse_args()
    # main(parsed_args)
