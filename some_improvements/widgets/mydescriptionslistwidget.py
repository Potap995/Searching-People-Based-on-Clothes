from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor

from MyWidgets import MyDescriptionItemWidget
from processing import TrackItem


class MyDescriptionsListWidget(QtCore.QObject):

    def __init__(self, ui_parent=None, parent=None):
        super(MyDescriptionsListWidget, self).__init__(ui_parent)
        self.controller = parent
        self.scrollareaDescriprions = ui_parent.scrollareacontentsDescriptions
        self.layoutDescriprions = ui_parent.verticallayoutDescriptions
        self.comboboxClothes = ui_parent.comboboxClothes
        self.labelPersonID = ui_parent.labelPersonID
        self.labelFrameCount = ui_parent.labelFrameCount
        self.buttonClothesFind = ui_parent.buttonClothesFind
        self.buttonSaveClothes = ui_parent.buttonSaveClothes

        self.counter = 0
        self.descriptions = {}
        self.descriptions_changes = {"del": set(), "change": set(), "add": set()}
        self._load_combobox_clothes_list()
        self.comboboxClothes.activated[str].connect(self._combobox_added)
        self.buttonClothesFind.clicked.connect(self._find_by_descriptions)
        self.buttonSaveClothes.clicked.connect(self._save_descriptions)
        self.person = None

        self.clear_person()
        self.clear_description()

    def set_person(self, person):
        self.clear_person()
        self.clear_description()
        self.person = person
        if isinstance(person, TrackItem):
            self.labelPersonID.setText(str(person.track_id))
            self.labelFrameCount.setText(str(person.frames_num))
        if isinstance(person, str):
            self.labelPersonID.setText("Извне")

    def get_person(self):
        return self.person

    def set_person_description(self, descriptions):
        self.clear_description()
        for description in descriptions:
            self._add_description(description)

    def _combobox_added(self, name):
        clothes_obj = self.controller.get_clothes_obj(name)
        self._add_description(clothes_obj)
        self.comboboxClothes.setCurrentIndex(-1)

    def _add_description(self, description):
        self.counter += 1
        widget = MyDescriptionItemWidget(self.scrollareaDescriprions, self.counter, description)
        widget.delete.connect(self._del_description)
        widget.changed.connect(self._description_changed)
        pos = self.layoutDescriprions.count() - 1
        self.descriptions[self.counter] = (widget, description)
        self.layoutDescriprions.insertWidget(pos, widget, alignment=QtCore.Qt.AlignTop)
        self._make_change(self.counter, "add")

    def _del_description(self, description_id):
        # del self.descriptions[description_id]
        self._make_change(description_id, "del")

    def _description_changed(self, description_id):
        self._make_change(description_id, "change")

    def _make_change(self, desc_id, change_type):
        if not self.person:
            return

        self.buttonSaveClothes.setEnabled(True)
        if change_type == "change":
            if desc_id not in self.descriptions_changes["add"]:
                self.descriptions_changes["change"].add(desc_id)
        elif change_type == "del":
            if desc_id in self.descriptions_changes["add"]:
                self.descriptions_changes["add"].discard(desc_id)
            else:
                self.descriptions_changes["del"].add(desc_id)
                self.descriptions_changes["change"].discard(desc_id)
        elif change_type == "add":
            self.descriptions_changes["add"].add(desc_id)
        else:
            print("Неизвестный тип изменения для mydescriptionwidget")

    def _load_combobox_clothes_list(self):
        clothes = self.controller.get_clothes_classes()
        self.comboboxClothes.addItems(clothes)
        self.comboboxClothes.setCurrentIndex(-1)

    def clear_person(self):
        self.labelPersonID.setText("NoID")
        self.labelFrameCount.setText(str(0))

        self.buttonSaveClothes.setEnabled(False)

    def clear_description(self):
        self.counter = 0
        self.descriptions = {}
        self.descriptions_changes = {"del": set(), "change": set(), "add": set()}
        while self.layoutDescriprions.count():
            item = self.layoutDescriprions.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

            self.layoutDescriprions.removeItem(item)
        self.layoutDescriprions.addStretch()

    def _find_by_descriptions(self):
        ret = [None] * len(self.descriptions)
        for i, key in enumerate(self.descriptions):
            ret[i] = self.descriptions[key][1]
        self.controller.find_by_clothes(ret)

    def _save_descriptions(self):
        if self.person is None:
            return

        self.buttonSaveClothes.setEnabled(False)
        print(self.descriptions_changes)

        changes = dict()
        changes["del"] = [self.descriptions[desc_id][1] for desc_id in self.descriptions_changes["del"]]
        changes["add"] = [self.descriptions[desc_id][1] for desc_id in self.descriptions_changes["add"]]
        changes["change"] = [self.descriptions[desc_id][1] for desc_id in self.descriptions_changes["change"]]

        self.controller.save_description_changes(changes, self.person)

        for desc_id in self.descriptions_changes["del"]:
            del self.descriptions[desc_id]

        self.descriptions_changes = {"del": set(), "change": set(), "add": set()}




