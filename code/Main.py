from PyQt5.QtWidgets import QApplication
from MyApp import MainApp
import sys

app = QApplication(sys.argv)
window = MainApp()
window.show()
sys.exit(app.exec_())