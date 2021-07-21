import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPainter


class MyFrame(QtWidgets.QFrame):

    resizeSignal = QtCore.pyqtSignal(tuple)
    mouse_clicked = QtCore.pyqtSignal(tuple)

    def __init__(self, parent=None):
        super(MyFrame, self).__init__(parent)
        self.image = None
        self.cur_image = None

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setMidLineWidth(0)

    def _resizeImage(self):
        """ Метод для изменения отображаемой картинки

        Нужно вызывать тогда когда не совпали и высота и ширана виджета и картинки для отрисовки
        """

        if self.image is not None:
            self.cur_image = self.image.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            self.update()

    def _get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.copy(), # раньше тут было image.data, но затем перестало работать
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)
        return image

    def set_image(self, img):
        """ Метод для передачи картинки в виде numpy.array(width, height, 3)"""
        self.image = self._get_qimage(img)
        if img.shape[0] != self.height() and img.shape[1] != self.width():
            # print("resize on set")
            self._resizeImage()
        else:
            self.cur_image = self.image
            self.update()

    def paintEvent(self, event):
        """ Отрисовываем текущую картинку в нужной позиции """
        if self.cur_image is not None:
            painter = QPainter(self)
            x_pos = (self.width() - self.cur_image.width()) // 2
            y_pos = (self.height() - self.cur_image.height()) // 2
            painter.drawImage(x_pos, y_pos, self.cur_image)

    def resizeEvent(self, event):
        """ Отправляем сигнал о том что произошло измение размера """
        self.resizeSignal.emit((self.size().height(), self.size().width()))
        self._resizeImage()

    def mousePressEvent(self, event):
        if self.cur_image:
            pos = event.pos()
            margin_width = (self.width() - self.cur_image.width()) // 2
            margin_height = (self.height() - self.cur_image.height()) // 2
            relative_x = (pos.x() - margin_width) / self.cur_image.width()
            relative_y = (pos.y() - margin_height) / self.cur_image.height()
            if 0 <= relative_x <= 1 and 0 <= relative_y <= 1:
                self.mouse_clicked.emit((relative_x, relative_y, pos.x(), pos.y()))
