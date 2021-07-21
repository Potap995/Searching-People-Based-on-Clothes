import cv2
import numpy as np
from collections import defaultdict

from PyQt5 import QtWidgets, QtCore


class MyPersonWidget(QtCore.QObject):

    def __init__(self, ui_parent=None, parent=None):
        super(MyPersonWidget, self).__init__(ui_parent)
        self.controller = parent
        self.framePerson = ui_parent.framePerson
        self.buttonLoad = ui_parent.buttonLoad
        self.buttonClothesDet = ui_parent.buttonClothesDet
        self.buttonReID = ui_parent.buttonReID
        self.img_id_label = ui_parent.img_id_label

        self.img_np = None  # RGB
        self.img_id = None

        self.buttonLoad.clicked.connect(self._buttonLoad_clicked)
        self.buttonReID.clicked.connect(self._buttonReID_clicked)
        self.buttonClothesDet.clicked.connect(self._buttonClothes_clicked)

        self.models_available = False
        self.person_loaded = False
        self.clear()

    def _buttonLoad_clicked(self):
        current_file_name = \
            QtWidgets.QFileDialog.getOpenFileName(None, "Select Image File",
                                                  "./data/persons",
                                                  "*.jpg *.png")[0]
        if current_file_name == "":
            return
        else:
            self.clear()
            img = np.array(cv2.imread(current_file_name)[:, :, ::-1])  # ->RGB
            self.controller.set_track(img, current_file_name)

    def set_person_img(self, img):
        self.img_np = np.array(img).astype(np.uint8)
        self.framePerson.set_image(self.img_np)
        self.person_loaded = True
        self._update_search_buttons_state()

    def _buttonReID_clicked(self):
        new_img = self.controller.find_by_reid(self.img_np)
        if new_img is not None:
            self.framePerson.set_image(new_img)

    def _buttonClothes_clicked(self):
        new_img = self.controller.find_and_display_clothes(self.img_np)
        self.framePerson.set_image(new_img)

    def _update_search_buttons_state(self):
        state = False
        if self.models_available and self.person_loaded:
            state = True
        self.buttonReID.setEnabled(state)
        self.buttonClothesDet.setEnabled(state)

    def set_model_available(self, flag):
        self.models_available = flag
        self._update_search_buttons_state()

    def clear(self):
        img = np.ones((7, 5, 3)).astype(np.uint8)*255
        self.set_img_id(None)
        self.set_person_img(img)
        self.person_loaded = False
        self._update_search_buttons_state()

    def get_cur_img(self):
        return self.img_np

    def set_img_id(self, id_):
        self.img_id = id_
        text_id = str(id_)
        if id_ is None:
            text_id = "NoID"
        self.img_id_label.setText(text_id)

    def get_img_id(self):
        return self.img_id

    def loaded(self):
        return self.person_loaded
