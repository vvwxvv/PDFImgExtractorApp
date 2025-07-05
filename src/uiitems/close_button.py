from PyQt5.QtWidgets import QPushButton

class CloseButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("X", parent)
        self.initUI()

    def initUI(self):
        self.clicked.connect(self.onClick)
        self.setStyleSheet("""
            QPushButton {
                color: black;
                font-weight: bold;
                border: none;
                padding: 5px 10px;
                margin-right: 10px;
                background-color: white;
            }
        """)

    def onClick(self):
        if self.parent():
            self.parent().close()