import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap
from src.assets.extract_pdf_images import (
    PDFImageExtractor,
    ExtractionConfig,
    ImageFormat,
)
from src.uiitems.close_button import CloseButton


class MainWorkflowApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pdf_file_path = ""
        self.output_folder_path = ""
        self.setMouseTracking(True)
        self.oldPos = self.pos()

    def init_ui(self):
        """初始化用户界面。"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("App")

        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Arial';
                background-color: transparent; 
                border: 2px solid #CDEBF0; 
                border-radius: 20px;
            }
            QPushButton {
                background-color: #CDEBF0;
                color: black;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #BEE0E8;
            }
            QLabel#Logo {
                background-color: transparent;
            }
            QMessageBox {
                background-color: #BEE0E8;
                color: white;
                font-size: 16px;
            }
            QMessageBox QPushButton {
                color: white;
                border: 2px solid white;
                border-radius: 8px;
                padding: 6px;
                font-size: 24px;
                min-width: 70px;
                min-height: 30px;
            }
            QMessageBox QPushButton:hover {
                background-color: #BEE0E8;
            }
        """
        )

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(5, 5, 5, 5)
        self.mainLayout.setSpacing(10)
        self.mainLayout.addLayout(self.create_title_bar())
        self.logoLabel = self.create_logo_label()
        self.mainLayout.addWidget(self.logoLabel)

        self.pdf_path_button = self.create_button(
            "Select PDF File", self.select_pdf_path
        )
        self.mainLayout.addWidget(self.pdf_path_button)

        self.output_path_button = self.create_button(
            "Select Output Folder", self.select_output_folder_path
        )
        self.mainLayout.addWidget(self.output_path_button)

        cooking_style = "font-size: 14px; color: #CDEBF0; background-color:black; border-radius: 20px; text-decoration:underline; padding:20px;"
        self.mainLayout.addWidget(
            self.create_button(
                "Start Cooking", self.toggle_cooking_section, cooking_style
            )
        )

        self.resize(540, 880)

    def create_button(self, text, slot, style=None):
        """创建按钮并设置点击事件。"""
        button = QPushButton(text, self)
        button.clicked.connect(slot)
        button.setStyleSheet(
            style
            if style
            else """
            QPushButton {
                background-color: #CDEBF0;
                color: black;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #BEE0E8;
            }
        """
        )
        return button

    def create_title_bar(self):
        """创建标题栏。"""
        title_bar = QHBoxLayout()
        close_button = CloseButton(self)
        title_bar.addWidget(close_button, alignment=Qt.AlignRight)
        return title_bar

    def create_logo_label(self):
        """创建 Logo 标签。"""
        logo = QLabel(self)

        # Get the directory where the script/exe is located
        if getattr(sys, "frozen", False):
            # Running as compiled executable
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to cover.png
        cover_path = os.path.join(base_path, "static", "cover.png")
        pixmap = QPixmap(cover_path).scaled(500, 800)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        logo.setObjectName("Logo")
        return logo

    def select_output_folder_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.output_folder_path = folder_path

    def select_pdf_path(self):
        """选择 PDF 文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_file_path = file_path

    def toggle_cooking_section(self):
        """切换烹饪部分。"""
        if self.pdf_file_path and self.output_folder_path:
            self.run_workflow()
        else:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please select a PDF file and an output folder before starting the process.",
            )

    def run_workflow(self):
        """运行工作流程。"""
        if not self.pdf_file_path:
            QMessageBox.warning(self, "Error", "No PDF file selected.")
            return
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "No output folder selected.")
            return
        try:
            pdf_basename = os.path.basename(self.pdf_file_path)
            pdf_basename_no_ext = os.path.splitext(pdf_basename)[0]
            new_output_folder = os.path.join(
                self.output_folder_path, f"{pdf_basename_no_ext}_extractimages"
            )

            # Create configuration for the extractor
            config = ExtractionConfig(
                target_size_kb=700,
                image_format=ImageFormat.WEBP,
                image_quality=85,
                dpi=150,
            )

            # Create and run the extractor
            extractor = PDFImageExtractor(config=config)
            stats = extractor.extract_images(self.pdf_file_path, new_output_folder)

            # Show success message with statistics
            success_msg = f"Extraction completed successfully!\n\n"
            success_msg += f"Images extracted: {stats.images_extracted}\n"
            success_msg += f"Pages rendered: {stats.pages_rendered}\n"
            success_msg += f"Duplicates skipped: {stats.duplicates_skipped}\n"
            success_msg += f"Processing time: {stats.processing_time:.2f}s\n"
            success_msg += f"Output folder: {new_output_folder}"

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setText(success_msg)
            msg.setWindowTitle("Success")
            msg.setWindowFlags(
                Qt.FramelessWindowHint | Qt.Dialog | Qt.CustomizeWindowHint
            )
            msg.exec_()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred during processing: {str(e)}"
            )

    def mousePressEvent(self, event):
        """鼠标按下事件。"""
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件。"""
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWorkflowApp()
    window.show()
    sys.exit(app.exec_())
