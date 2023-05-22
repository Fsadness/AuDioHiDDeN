from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QStyle, QMessageBox, QLabel, QProgressDialog, QFileDialog
# from PyQt5.QtUiTools import QUiLoader
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QIcon
# from draw_wave import result_show
from test_audio import hide, extract
import os

# class ih(QObject):
class IH(QtWidgets.QDialog):
    FILE_FILTERS = ""

    selected_filter = "wav(*.wav);;mp4(*.mp4)"

    def __init__(self, parent=None):
        super(IH, self).__init__(parent)

        self.setWindowTitle("AuDioHiDDeN")
        self.setMinimumSize(300, 120)
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        self.create_widgets()
        self.create_layout()
        self.create_connections()

    def create_widgets(self):
        self.label_text = QLabel('AuDioHiDDeN：可抵赖音频隐写工具')
        self.label_text.setFont(QtGui.QFont(""))
        self.func_text = QLabel('@接收端')

        self.key_le = QtWidgets.QLineEdit()

        self.filepath_le = QtWidgets.QLineEdit()
        self.select_file_path_btn = QtWidgets.QPushButton()
        self.select_file_path_btn.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap(21)))
        self.select_file_path_btn.setToolTip("Select File")

        self.apply_btn = QtWidgets.QPushButton("Apply")
        # self.apply_btn.clicked.connect(self.onButtonApply)
        self.close_btn = QtWidgets.QPushButton("Close")

    def create_layout(self):
        vLayout = QtWidgets.QVBoxLayout()
        vLayout.addSpacing(10)
        vLayout.addWidget(self.label_text)
        self.label_text.setFont(QtGui.QFont(self.font().family(), 14))
        vLayout.addSpacing(10)
        vLayout.addWidget(self.func_text)
        self.func_text.setFont(QtGui.QFont(self.font().family(), 11))
        vLayout.addSpacing(20)

        key_layout = QtWidgets.QVBoxLayout()
        key_layout.addWidget(self.key_le)
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Key: ", key_layout)

        file_path_layout = QtWidgets.QHBoxLayout()
        file_path_layout.addWidget(self.filepath_le)
        file_path_layout.addWidget(self.select_file_path_btn)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.close_btn)

        form_layout.addRow("Stego: ", file_path_layout)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(vLayout)
        main_layout.addLayout(key_layout)
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)

    def create_connections(self):
        self.select_file_path_btn.clicked.connect(self.show_file_select_dialog)
        self.apply_btn.clicked.connect(self.apply_key)
        self.close_btn.clicked.connect(self.close)

    def show_file_select_dialog(self):
        file_path, self.selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", self.FILE_FILTERS,
                                                                                self.selected_filter)
        if file_path:
            self.filepath_le.setText(file_path)

    def load_file(self):
        file_path = self.filepath_le.text()
        if not file_path:
            return

        file_info = QtCore.QFileInfo(file_path)
        if not file_info.exists():
            self.no_file_path_error(file_path)
            return

    def apply_key(self):
        file_path = self.filepath_le.text()
        if not file_path:
            return
        file_info = QtCore.QFileInfo(file_path)
        if not file_info.exists():
            self.no_file_path_error(file_path)
            return
        key = self.key_le.text()
        if key == "123456":
            # 使用正确的decoder解密
            extract(file_path, key)
            # extract(file_path, D1)
        else:
            # 使用错误的decoder解密
            extract(file_path, key)
            # extract(file_path, D2)

    def no_file_path_error(self, file_path):
        QMessageBox.critical(self, 'Error', 'File does not exist: {0}'.format(file_path))

    def onButtonApply(self):
        elapsed = 50000
        dlg = QProgressDialog('进度', '取消', 0, elapsed, self)
        dlg.setWindowTitle('等待......')
        dlg.setWindowFlags(dlg.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        for val in range(elapsed):
            dlg.setValue(val)
            QCoreApplication.processEvents()
            if dlg.wasCanceled():
                break
        dlg.setValue(elapsed)


if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon('logo.png'))
    key = ""
    ih = IH()
    ih.show()
    app.exec_()
