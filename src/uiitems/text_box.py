from PyQt5.QtWidgets import QGraphicsBlurEffect, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont

class TextBox(QMainWindow):
    def __init__(self, bgColor="yellow", message="Start here.", width=600, height=300):
        super().__init__()
        self.bgColor = bgColor  # Set to yellow or any other color passed
        self.message_text = message
        self.width = width
        self.height = height
        self.initUI()
        self.oldPos = self.pos()  # For storing the old position to make the window draggable

    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Translucent background
        self.setGeometry(350, 350, self.width, self.height)  # Window size and position based on provided dimensions

        # Central widget with layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)  # Adjusted bottom margin to be equal

        # Style the central widget for yellow background and rounded corners
        self.central_widget.setStyleSheet(f"""
            background-color: {self.bgColor};
            border-radius: 20px;
        """)

        # Close button
        self.close_button = QPushButton("X", self)
        self.close_button.setFont(QFont('Arial', 14))
        self.close_button.setStyleSheet("background-color: transparent; color: black; border: none;")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button, 0, Qt.AlignRight | Qt.AlignTop)

        # Message label with custom line spacing
        self.message_label = QLabel(f"<p style='line-height: 150%;'>{self.message_text}</p>", self.central_widget)
        self.message_label.setFont(QFont('Arial', 16))
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        self.layout.addWidget(self.message_label)

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()