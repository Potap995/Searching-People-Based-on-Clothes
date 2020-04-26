class ClothesList:
    def __init__(self, parent):
        self.parent = parent
        self.cats_path = "D:\Programming\CourseWork_3\code\data\categories.txt"
        self.listWidget = self.parent.GetClothesList()
        self.clothes_list = dict()
        self.displayClothesList()

    def displayClothesList(self):
        categories = self._getCategoriesList()
        for clothes in categories:
            self.add_chbox(clothes)
        self.listWidget.addStretch()

    def add_chbox(self, clothes):
        wid = self.parent.addCHeckBoxClothes(clothes)
        self.clothes_list[clothes] = wid

    def getCheckedClothes(self):
        checked = []
        for id, clothes in self.clothes_list.items():
            if clothes.isChecked():
                checked.append(id)
        return checked

    def _getCategoriesList(self):
        cats_file = open(self.cats_path, 'r').readlines()
        categories = []

        for ln in cats_file:
            cur = list(filter(None, ln[:-1].split(' ')))
            if cur[1] in ("1", "2", "3"):
                categories.append(cur[0])

        return categories
