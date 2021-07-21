import json
import os
import configparser

global_config = configparser.ConfigParser()
global_config.read("config.ini")


class Config(object):

    def __new__(cls, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self, filepath='config_default.json'):
        self.filepath = filepath

        with open(self.filepath, 'r', encoding='utf-8') as fh:
            self._dict = json.load(fh)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val

    def update(self, conf):
        self._dict.update(conf)

    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as fp:
            json.dump(self._dict, fp)


# app_config = Config()


class Config_old(object):
    def __new__(cls, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config_old, cls).__new__(cls)
        return cls.instance

    def __init__(self, filepath=None):
        if filepath is None:  # config по дефолту
            with open('../config_default.json', 'r', encoding='utf-8') as fh:
                tmp_dict = json.load(fh)
        else:  # config из выбранного json файла
            with open(filepath, 'r', encoding='utf-8') as fh:
                tmp_dict = json.load(fh)

        self.detectron2 = tmp_dict["detectron2"]
        self.detectron2_min_confidence = tmp_dict["detectron2_min_confidence"]
        self.detectron2_nms_thresh = tmp_dict["detectron2_nms_thresh"]
        self.tracker_path = tmp_dict["tracker_path"]
        self.max_dist = tmp_dict["max_dist"]
        self.yolo_min_confidence = tmp_dict["yolo_min_confidence"]  # .3
        self.yolo_nms_max_overlap = tmp_dict["yolo_nms_max_overlap"]  # .5
        self.max_iou_distance = tmp_dict["max_iou_distance"]
        self.max_age = tmp_dict["max_age"]
        self.n_init = tmp_dict["n_init"]
        self.nn_budget = tmp_dict["nn_budget"]
        self.class_ = tmp_dict["class_"]
        self.yolo_width = tmp_dict["yolo_width"]
        self.yolo_height = tmp_dict["yolo_height"]
        self.detectron_width = tmp_dict["detectron_width"]
        self.detectron_height = tmp_dict["detectron_height"]
        self.byteslen = tmp_dict["byteslen"]
        self.min_side = tmp_dict["min_side"]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def update(self, conf):
        if type(conf) is dict:
            orig_keys = self.__dict__.keys()
            keys = conf.keys()
            for key in keys:
                if key not in orig_keys:
                    print("Key", key, "does not exist in original configs")
                self.__setattr__(key, conf[key])
        else:
            raise Exception("Argument is not a dictionary")

    def save(self, filename):
        keys = self.__dict__.keys()
        values = self.__dict__.values()
        # print(keys)
        # print(values)
        tmp_conf = {key: values for key, value in zip(keys, values)}
        # print(type(tmp_conf))
        # with open(filename, 'w', encoding='utf-8') as fh:  # открываем файл на запись
        #    fh.write(json.dumps(tmp_conf, ensure_ascii=False))


def config2dict(config):
    dict = {}
    for section in config.sections():
        dict[section] = {}

        for option in config.options(section):
            dict[section][option] = config.get(section, option)

    return dict


def print_config():
    print(config2dict(global_config))
