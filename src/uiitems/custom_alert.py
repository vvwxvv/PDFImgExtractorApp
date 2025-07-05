from PyQt5.QtWidgets import QMessageBox, QPushButton, QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

class CustomAlert(QWidget):
    close_app_signal = pyqtSignal()

    def __init__(self, parent=None, message="", is_error=False):
        super().__init__(parent)
        self.oldPos = QPoint()
        self.initUI(message, is_error)

    def initUI(self, message, is_error):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.message_label = QLabel(message)
        self.message_label.setStyleSheet("color: red;" if is_error else "color: green;")

        layout = QVBoxLayout()
        layout.addWidget(self.message_label)
        self.setLayout(layout)

        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.accept)  # Connect to the newly defined accept method
        layout.addWidget(close_button)

    def accept(self):
        self.close()  # Implement accept to close the widget
    def show_completion_alert(self):
        # Custom close button
        close_button = QPushButton('X', self)
        close_button.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: white;
                border: none;
                font-size: 20px;
                margin: 0;
            }
            QPushButton:hover {
                background-color: white;
            }
        """)
        close_button.setFixedSize(20, 20)
        close_button.clicked.connect(self.close)

        # Create a message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Generation Complete")
        msg_box.setText("The app generation is done. Do you want to close the window?")
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: pink;
                color: white;
                font-size: 16px;
                padding: 50px;
            }
            QPushButton {
                color: white;
                border: 2px solid white;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 50);
                padding: 5px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 100);
            }
        """)
        msg_box.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.button(QMessageBox.Yes).clicked.connect(self.yes_clicked)
        msg_box.button(QMessageBox.No).clicked.connect(msg_box.reject)

        # Move custom close button to the top-right corner of the message box
        close_button.setParent(msg_box)
        close_button.move(msg_box.width() - 30, 10)

        ret = msg_box.exec_()  # Use exec_ to block interaction with other windows

        if ret == QMessageBox.Yes:
            self.close()

    def yes_clicked(self):
        self.close_app_signal.emit()  # Emit signal when 'Yes' is clicked
        self.close()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def set_success_message(self, message):
        self.message_label.setText(message)
        self.message_label.setStyleSheet("color: green;")

    def set_error_message(self, message):
        self.message_label.setText(message)
        self.message_label.setStyleSheet("color: red;")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ex = CustomAlert()
    ex.show_completion_alert()
    sys.exit(app.exec_())