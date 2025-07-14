from PyQt5.QtWidgets import QWidget, QTextEdit, QVBoxLayout

class PreviewBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        # Create a layout
        layout = QVBoxLayout(self)

        # Create a read-only QTextEdit
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Preview will appear here...")  # Updated placeholder text

        # Set styles for the QTextEdit
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255, 192, 203, 0.5);
                color: black;
                font-weight: bold;
                font-family: 'Roboto', sans-serif;
                border-radius: 10px;
            }
            QTextEdit::placeholder {
                color: gray;
                text-align: center;
            }
        """)

        # Add QTextEdit to the layout
        layout.addWidget(self.text_edit)

    def setText(self, text):
        """Set the text displayed in the text edit widget."""
        self.text_edit.setText(text)