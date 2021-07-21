import sys
import os

CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CUR_FOLDER, 'yolo4'))
sys.path.append(CUR_FOLDER)  # для deep_sort

import cv2
import torch
import torchvision # needed for correct loading of torchscript MaskRCNN model
import logging
import config

from tool.darknet2pytorch import Darknet
from tool.config import parse_cfg

from torch.nn import functional as F

from deep_sort.deep.feature_extractor import Extractor
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.sort.tracker import Tracker
from deep_sort.sort.detection import Detection
from collections import namedtuple


try:
    sys.path.append(os.path.join(CUR_FOLDER, 'fastreid'))
    from fastreid.engine import DefaultPredictor
    from fastreid.config import get_cfg as get_cfg_fastreid
except:
    print("Fast-ReID не установлен")

try:
    # raise NotImplementedError()
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2 import model_zoo
    from detectron2.config import get_cfg as get_cfg_detectron2
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.engine import DefaultPredictor as DefaultPredictorDetectron2
except:
    print("Detectron2 не установлен")
    logging.getLogger().info("Ошибка: неудалось загрузить detectron2. Проверьте, что у вас есть TS модель.")
    pass

import numpy as np



class ClothesDetectron:
    def __init__(self, conf=None):
        self.predictor = build_predictor_from_config()

    def __call__(self, image):
        # print(image.shape)
        outputs = self.predictor(image)

        classes = np.array(list(map(int, outputs["instances"].pred_classes.cpu())))
        boxes = np.around(outputs["instances"].pred_boxes.tensor.cpu().numpy())
        masks = outputs["instances"].pred_masks.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        class_to_mask = {}

        rgb_all_classes = []
        hue_hists = []

        mask_ = classes <= 27
        masks = masks[mask_, :]
        classes = classes[mask_]
        scores = scores[mask_]
        boxes = boxes[mask_]

        im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hue = im[:, :, 0].flatten()

        for i in range(len(classes)):
            # im = image[masks[i], :].astype(np.int)  # HSV - строить только по value (0 - 360)
            rgb_all_classes.append(np.mean(image[masks[i], :], axis=0).astype(np.int))
            hist = cv2.calcHist([hue[masks[i].flatten()]], [0], None, [60], [0, 360])  # 60 bin, возможно увеличить
            norm_hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hue_hists.append(norm_hist.flatten())
            # if classes[i] in class_to_mask:
            #     prev_classes = class_to_mask.get(classes[i])
            #     if type(prev_classes) is list:
            #         prev_classes.append(masks[i])
            #     else:
            #         prev_classes = [prev_classes, masks[i]]
            #     class_to_mask.update({classes[i]: prev_classes})
            # else:
            #     class_to_mask.update({classes[i]: [masks[i]]})

            # lab_all_classes.append(lab[masks[i], :])
            # rgb_hists.append(cv2.calcHist([im], [0, 0, 0], None, [16, 16, 16], [0, 256, 0, 256, 0, 256]))

        ret = {"classes": classes, "scores": scores, "bboxes": boxes, "rgb_colors": rgb_all_classes,
               "hue_hists": hue_hists, "masks": masks}

        return ret


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Function from detecton2
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


class MaskRCNNPredictorTS:
    Instances = namedtuple("Instances", field_names=["pred_classes", "pred_boxes", "pred_masks", "scores"])
    Box = namedtuple("Box", field_names=["tensor"])
    def __init__(self, model_file_path):
        self._model = torch.jit.load(model_file_path)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    def __call__(self, image):
        blob = torch.from_numpy(np.transpose(image, axes=[2, 0, 1])).to(self.device)
        with torch.no_grad():
            xyxy, otype, mask_small, confidence, padded_shape = self._model(blob)
            masks = _do_paste_mask(mask_small, xyxy, blob.shape[1], blob.shape[2], skip_empty=False)[0] > 0.5
        return {
            "instances" : self.Instances(otype, self.Box(tensor=xyxy), masks[:, :image.shape[0], :image.shape[1]], confidence)
        }


def build_predictor_detectron():
    # raise NotImplementedError()
    cfg = get_cfg_detectron2()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("./processing/data/imat_cfg_rcnn.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "./processing/data/model_final_rcnn.pth"
    return DefaultPredictorDetectron2(cfg)


def build_predictor_ts():
    path = config.global_config["processing"].get("cloth_segmentation_ts_model", "")
    assert os.path.exists(path), "Failed to find checkpoint file."
    return MaskRCNNPredictorTS(path)


def build_predictor_from_config():
    if config.global_config["processing"].getboolean("use_torch_script"):
        return build_predictor_ts()
    return build_predictor_detectron()


class Extractor_fastreid:
    def __init__(self, conf="fastreid/configs/Market1501/bagtricks_R50.yml", weights="data/model_final_comb.pth"):
        self.cfg = get_cfg_fastreid()
        self.cfg.merge_from_file(os.path.join(CUR_FOLDER, conf))
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg.MODEL.WEIGHTS = os.path.join(CUR_FOLDER, weights)
        print(self.cfg.MODEL.WEIGHTS)
        self.cfg.MODEL.DEVICE = "cuda"
        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)  # Кушает RGB

    def _img_preproc(self, original_image):
        original_image = original_image[:, :, ::-1]  # BGR -> RGB
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return image

    def __call__(self, im_crops):
        # Получаем изображения в формате BGR
        for i in range(len(im_crops)):
            im_crops[i] = self._img_preproc(im_crops[i])
        im_crops = torch.stack(im_crops)
        with torch.no_grad():
            ress = self.predictor(im_crops)
        return ress.numpy()


class ExtractorTorchScript(Extractor_fastreid):
    def __init__(self, saved_script):
        self.__model = torch.jit.load(saved_script).to(torch.device("cuda:0" if torch.cuda.is_available() else 'cpu'))

    def _img_preproc(self, original_image):
        original_image = original_image[:, :, ::-1]  # BGR -> RGB
        #TODO
        image = cv2.resize(original_image, (128, 256), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image.astype("float32")).cuda()
        return image

    def __call__(self, im_crops):
        im_crops = [self._img_preproc(c) for c in im_crops]
        batch = torch.stack(im_crops).permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.__model(batch).detach().float().cpu().numpy()


def get_yolo_model(conf):
    cfgfile = os.path.join(CUR_FOLDER, "yolo4/cfg/yolov4.cfg")
    weightfile = os.path.join(CUR_FOLDER, "data/yolov4.weights")

    model = Darknet(cfgfile)
    model.width = conf["yolo_width"]
    model.height = conf["yolo_height"]
    model.load_weights(weightfile)
    model.cuda()
    model.half()
    model.eval()

    return model


def get_detectron2_model(conf):
    cfg = get_cfg_detectron2()
    cfg.merge_from_file(model_zoo.get_config_file(conf["detectron2"]))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf["detectron2_min_confidence"]
    cfg.MODEL.RPN.NMS_THRESH = conf["detectron2_nms_thresh"]

    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(
        model_zoo.get_checkpoint_url(
            conf["detectron2"]))  # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
    model.eval()
    return model


def get_tracker(conf):
    metric = NearestNeighborDistanceMetric("cosine", conf["max_dist"], conf["nn_budget"])
    tracker = Tracker(metric, max_iou_distance=conf["max_iou_distance"], max_age=conf["max_age"], n_init=conf["n_init"])
    return tracker


def get_extractor_fastreid(conf):
    # переписать
    if config.global_config["processing"].getboolean("use_torch_script"):
        return ExtractorTorchScript(config.global_config["processing"].get("torch_script_reid_model_path"))
    if "fast_reid_conf" in conf and "fast_reid_weigths" in conf:
        print(conf["fast_reid_conf"], conf["fast_reid_weigths"])
        return Extractor_fastreid(conf["fast_reid_conf"], conf["fast_reid_weigths"])
    else:
        print("fast_reid_base")
        return Extractor_fastreid()


def get_extractor_base(conf):
    return Extractor(conf["tracker_path"], use_cuda=True)


def get_extractor_torchreid(conf):
    return Extractor_torchreid(
        model_name='osnet_x1_0',
        model_path='/content/drive/MyDrive/Colab Notebooks/deep-person-reid/log/dukemtmc/osnet_x1_0/model/model.pth.tar-60',
        device='cuda')


def get_detectron_clothes(conf):
    return ClothesDetectron()
