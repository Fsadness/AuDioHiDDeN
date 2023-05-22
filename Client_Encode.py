from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QStyle, QMessageBox, QLabel, QProgressDialog, QFileDialog
# from PyQt5.QtUiTools import QUiLoader
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal, QSize, QMutex, QMutexLocker, QWaitCondition
from PyQt5.QtGui import QIcon
# from draw_wave import result_show
from test_audio import hide, extract
import os
import warnings


class ApplyThread(QThread):
    # 声明一个信号
    update_ui_signal = pyqtSignal(str)

    def __init__(self):
        super(ApplyThread, self).__init__()

    def run(self):
        # 以下为一通逻辑操作，比如查询数据库
        file_path_ = a
        real_msg_ = b
        real_key_ = c
        fake_msg_ = d
        fake_key_ = e
        hide(file_path_, real_msg_, real_key_, fake_msg_, fake_key_)

        # 根据查询数据库的结果，比如查询到200，发送信号
        self.update_ui_signal.emit((str(200)))


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
        self.func_text = QLabel('@发送端')

        self.msg1_le = QtWidgets.QLineEdit()
        self.key1_le = QtWidgets.QLineEdit()
        self.msg2_le = QtWidgets.QLineEdit()
        self.key2_le = QtWidgets.QLineEdit()

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

        real_msg_layout = QtWidgets.QVBoxLayout()
        real_msg_layout.addWidget(self.msg1_le)
        real_layout = QtWidgets.QFormLayout()
        real_layout.addRow("Real Message: ", real_msg_layout)
        real_key_layout = QtWidgets.QVBoxLayout()
        real_key_layout.addWidget(self.key1_le)
        real_layout.addRow("Real Key: ", real_key_layout)

        fake_msg_layout = QtWidgets.QVBoxLayout()
        fake_msg_layout.addWidget(self.msg2_le)
        fake_layout = QtWidgets.QFormLayout()
        fake_layout.addRow("Fake Message: ", fake_msg_layout)
        fake_key_layout = QtWidgets.QVBoxLayout()
        fake_key_layout.addWidget(self.key2_le)
        fake_layout.addRow("Fake Key: ", fake_key_layout)

        work_layout = QtWidgets.QHBoxLayout()
        work_layout.addLayout(real_layout)
        work_layout.addLayout(fake_layout)

        file_path_layout = QtWidgets.QHBoxLayout()
        file_path_layout.addWidget(self.filepath_le)
        file_path_layout.addWidget(self.select_file_path_btn)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.close_btn)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Cover: ", file_path_layout)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(vLayout)
        main_layout.addLayout(work_layout)
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)

    def create_connections(self):
        self.select_file_path_btn.clicked.connect(self.show_file_select_dialog)
        self.apply_btn.clicked.connect(self.open_file)
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

    def open_file(self):
        # global file_path, real_msg, real_key, fake_msg, fake_key
        global a, b, c, d, e
        a = self.filepath_le.text()
        b = self.msg1_le.text()
        c = self.key1_le.text()
        d = self.msg2_le.text()
        e = self.key2_le.text()
        # real_msg = self.msg1_le.text()
        # real_key = self.key1_le.text()
        # fake_msg = self.msg2_le.text()
        # fake_key = self.key2_le.text()
        self.applyThread_object = ApplyThread()
        self.applyThread_object.start()
        self.applyThread_object.update_ui_signal.connect(self.update_label)
        # result_show(file_path, 321)
        # self.thread.run(file_path, real_msg, real_key, fake_msg, fake_key)
        # 在这里把Stego保存到本地。将隐写中保存集成为一个方法，在这里调用。
        print("###Dump decoder1, decoder2###")

    def update_label(self, str1):
        print("OK!")

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
    warnings.filterwarnings("ignore")
    app = QApplication([])
    app.setWindowIcon(QIcon('logo.png'))
    global file_path, real_msg, real_key, fake_msg, fake_key
    ih = IH()
    ih.show()
    app.exec_()
