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
    QProgressBar,
)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QPixmap
from src.assets.extract_pdf_images import (
    PDFImageExtractor,
    ExtractionConfig,
    ImageFormat,
)
from src.uiitems.close_button import CloseButton
from src.assets.crop_image_remove_white import contour_crop_image


class ExtractWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, pdf_path, output_folder, config):
        super().__init__()
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.config = config
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # First, emit a status update to show processing is starting
            self.status_update.emit("Initializing extraction...")

            # Create output directory if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)

            extractor = PDFImageExtractor(config=self.config)

            # Patch extractor to emit progress
            orig_process_page = extractor._process_page

            def patched_process_page(pdf_document, page_num, output_folder):
                if not self._is_running:
                    raise Exception("Extraction cancelled by user.")

                # Update status with current page info
                self.status_update.emit(f"Processing page {page_num + 1}...")

                orig_process_page(pdf_document, page_num, output_folder)
                self.progress.emit(page_num + 1)

            extractor._process_page = patched_process_page

            # Execute extraction with better error handling
            stats = extractor.extract_images(self.pdf_path, self.output_folder)

            # Send a final status update
            self.status_update.emit("Finalizing extraction...")

            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class MainWorkflowApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pdf_file_path = ""
        self.output_folder_path = ""
        self.setMouseTracking(True)
        self.oldPos = self.pos()
        self.thread = None
        self.worker = None

        # Add a status label to show process information
        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet(
            "QLabel { color: #CDEBF0; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 8px; }"
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setVisible(False)
        self.mainLayout.addWidget(self.status_label)

        # Application state
        self.is_processing = False

        # Setup a timer for UI updates
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui)
        self.ui_update_timer.start(100)  # Update UI every 100ms

    def init_ui(self):
        """初始化用户界面。"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("App")

        # Responsive height: 70% of screen height
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_height = screen_size.height()
        window_height = int(screen_height * 0.7)
        window_width = 540  # Keep width fixed or adjust as needed
        self.resize(window_width, window_height)

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

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border-radius: 8px;
                height: 24px;
                text-align: center;
                font-weight: bold;
                background: rgba(0, 0, 0, 150);
                color: #CDEBF0;
                border: 1px solid #CDEBF0;
            }
            QProgressBar::chunk {
                background: #CDEBF0;
                border-radius: 7px;
            }
            """
        )
        self.mainLayout.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet(
            "QPushButton { background-color: #F08080; color: white; font-weight: bold; border-radius: 8px; padding: 10px; margin: 10px; } QPushButton:hover { background-color: #D9534F; }"
        )
        self.cancel_button.clicked.connect(self.cancel_extraction)
        self.mainLayout.addWidget(self.cancel_button)

        # self.resize(540, 880) # This line is now handled by init_ui

    def update_ui(self):
        """Periodic UI updates for a more responsive feel"""
        if self.is_processing:
            # Process UI events to prevent freezing
            QApplication.processEvents()

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

        # Better error handling for image loading
        if os.path.exists(cover_path):
            pixmap = QPixmap(cover_path).scaled(
                500, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            logo.setPixmap(pixmap)
        else:
            # Fallback if image is missing
            logo.setText("Logo Image Not Found")
            logo.setStyleSheet(
                "color: #CDEBF0; background-color: rgba(0, 0, 0, 150); padding: 10px;"
            )

        logo.setAlignment(Qt.AlignCenter)
        logo.setObjectName("Logo")
        return logo

    def select_output_folder_path(self):
        """Select output folder with better feedback"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.output_folder_path = folder_path
            self.status_label.setText(f"Output folder: {os.path.basename(folder_path)}")
            self.status_label.setVisible(True)
            # Auto-hide status after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setVisible(False))

    def select_pdf_path(self):
        """选择 PDF 文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.pdf_file_path = file_path
            self.status_label.setText(f"PDF: {os.path.basename(file_path)}")
            self.status_label.setVisible(True)
            # Auto-hide status after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setVisible(False))

    def toggle_cooking_section(self):
        """切换烹饪部分。"""
        if self.is_processing:
            # Don't allow starting another process if one is already running
            return

        if self.pdf_file_path and self.output_folder_path:
            self.run_workflow()
        else:
            QMessageBox.warning(
                self,
                "Input Error",
                "Please select a PDF file and an output folder before starting the process.",
            )

    def run_workflow(self):
        if not self.pdf_file_path:
            QMessageBox.warning(self, "Error", "No PDF file selected.")
            return
        if not self.output_folder_path:
            QMessageBox.warning(self, "Error", "No output folder selected.")
            return

        # Set processing state flag
        self.is_processing = True

        try:
            pdf_basename = os.path.basename(self.pdf_file_path)
            pdf_basename_no_ext = os.path.splitext(pdf_basename)[0]
            new_output_folder = os.path.join(
                self.output_folder_path, f"{pdf_basename_no_ext}_extractimages"
            )

            # Show extraction initialization in the UI
            self.status_label.setText("Preparing extraction...")
            self.status_label.setVisible(True)

            config = ExtractionConfig(
                target_size_kb=700,
                image_format=ImageFormat.WEBP,
                image_quality=85,
                dpi=150,
            )

            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self.cancel_button.setVisible(True)
            self.pdf_path_button.setEnabled(False)
            self.output_path_button.setEnabled(False)

            # Create and configure the worker thread
            self.worker = ExtractWorker(self.pdf_file_path, new_output_folder, config)
            self.thread = QThread()
            self.worker.moveToThread(self.thread)

            # Connect signals
            self.worker.progress.connect(self.update_progress)
            self.worker.status_update.connect(self.update_status)
            self.worker.finished.connect(self.extraction_finished)
            self.worker.error.connect(self.extraction_error)
            self.thread.started.connect(self.worker.run)

            # Clean up connections when done
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(lambda: self.set_processing_state(False))

            # Set progress bar max to total pages
            self.set_progress_max_pages()

            # Start the worker thread
            self.thread.start()

        except Exception as e:
            self.is_processing = False
            QMessageBox.critical(
                self, "Error", f"An error occurred during processing: {str(e)}"
            )
            self.reset_ui_after_processing()

    def set_progress_max_pages(self):
        """Set the progress bar maximum based on PDF page count"""
        try:
            import fitz

            with fitz.open(self.pdf_file_path) as pdf:
                page_count = len(pdf)
                self.progress_bar.setMaximum(page_count)
                self.status_label.setText(f"PDF has {page_count} pages")
        except ImportError:
            self.status_label.setText("PyMuPDF not available, using default progress")
            self.progress_bar.setMaximum(100)
        except Exception as e:
            self.status_label.setText(f"Error reading PDF: {str(e)}")
            self.progress_bar.setMaximum(100)

    def update_status(self, status_text):
        """Update the status label with current processing information"""
        self.status_label.setText(status_text)
        self.status_label.setVisible(True)
        # Process events to ensure UI updates
        QApplication.processEvents()

    def update_progress(self, value):
        """Update progress bar with smoother animation"""
        self.progress_bar.setValue(value)
        # Process events to keep UI responsive
        QApplication.processEvents()

    def set_processing_state(self, is_processing):
        """Update the processing state and manage UI accordingly"""
        self.is_processing = is_processing

    def reset_ui_after_processing(self):
        """Reset UI elements after processing completes or fails"""
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.pdf_path_button.setEnabled(True)
        self.output_path_button.setEnabled(True)
        # Hide status label after a delay
        QTimer.singleShot(5000, lambda: self.status_label.setVisible(False))

    def extraction_finished(self, stats):
        """Handle successful extraction completion"""
        self.set_processing_state(False)
        self.reset_ui_after_processing()

        success_msg = f"Extraction completed successfully!\n\n"
        success_msg += f"Images extracted: {stats.images_extracted}\n"
        success_msg += f"Pages rendered: {stats.pages_rendered}\n"
        success_msg += f"Duplicates skipped: {stats.duplicates_skipped}\n"
        success_msg += f"Processing time: {stats.processing_time:.2f}s\n"
        success_msg += f"Output folder: {stats.output_folder}"

        self.status_label.setText(f"Extracted {stats.images_extracted} images")

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(success_msg)
        msg.setWindowTitle("Success")
        msg.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.CustomizeWindowHint)
        msg.exec_()

    def extraction_error(self, error_msg):
        """Handle extraction errors"""
        self.set_processing_state(False)
        self.reset_ui_after_processing()

        self.status_label.setText(f"Error: {error_msg}")

        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")

    def cancel_extraction(self):
        """Cancel the extraction process"""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Cancelling extraction...")

        # Reset UI even if cancellation somehow fails
        self.set_processing_state(False)
        self.reset_ui_after_processing()

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

    def resizeEvent(self, event):
        """Keep window height at 70% of device window height on resize."""
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_height = screen_size.height()
        window_height = int(screen_height * 0.7)
        self.resize(self.width(), window_height)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Properly clean up when closing the application"""
        # Stop any running thread
        if getattr(self, "thread", None) and isinstance(self.thread, QThread):
            try:
                if self.thread.isRunning():
                    if self.worker:
                        self.worker.stop()
                    self.thread.quit()
                    self.thread.wait(1000)  # Wait up to 1 second for thread to finish
            except RuntimeError:
                pass  # Thread already deleted
        event.accept()


if __name__ == "__main__":
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWorkflowApp()
    window.show()
    sys.exit(app.exec_())
