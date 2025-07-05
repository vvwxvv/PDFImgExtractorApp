
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QFont


class NotificationBar(QWidget):
    def __init__(self, message, parent=None):
        super(NotificationBar, self).__init__(parent)
        self.setFixedHeight(50)
        self.setLayout(QHBoxLayout())
        self.label = QLabel(message)
        self.label.setFont(QFont('Arial', 14))
        self.label.setStyleSheet("color: #FFFFFF;")
        self.setStyleSheet("background-color: #FF69B4; border-radius: 20px;")
        self.layout().addWidget(self.label, alignment=Qt.AlignLeft)

        self.close_button = QPushButton("✕", self)
        self.close_button.setFont(QFont('Arial', 16))
        self.close_button.setFixedSize(30, 30)
        self.close_button.setStyleSheet("""
            QPushButton {
                color: #FFFFFF;
                background-color: rgba(255, 182, 193, 0);
                border-radius: 15px;
            }
        """)
        self.close_button.clicked.connect(self.hide)
        self.layout().addWidget(self.close_button, alignment=Qt.AlignRight)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hide)
        self.timer.start(3000)