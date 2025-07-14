import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtCore import Qt

class DashedLine(QWidget):
    def __init__(self, color, orientation='horizontal', parent=None):
        super().__init__(parent)
        self.color = color
        self.orientation = orientation
        self.setMinimumSize(1, 10)  # Minimum size

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor(self.color))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        if self.orientation == 'horizontal':
            painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)
        else:
            painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DashedLine(color='pink', orientation='vertical')
    ex.show()
    sys.exit(app.exec_())