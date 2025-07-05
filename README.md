# PDFImgExtractorApp

A PyQt5 application for extracting and compressing images from PDF files with automatic optimization.

## Building the Executable

### Prerequisites
- Python 3.8 or higher
- Windows 10/11
- Virtual environment (recommended)

### Quick Build (Windows)

#### Option 1: Using Batch Script
1. Double-click `build.bat` or run it from Command Prompt
2. Wait for the build process to complete
3. The executable will be created in `dist\PDFImgExtractorApp.exe`

#### Option 2: Using PowerShell Script
1. Right-click `build.ps1` and select "Run with PowerShell"
2. Wait for the build process to complete
3. The executable will be created in `dist\PDFImgExtractorApp.exe`

#### Option 3: Manual Build
1. Open Command Prompt or PowerShell in the project directory
2. Activate your virtual environment (if using one):
   ```cmd
   appenv\Scripts\activate
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
4. Build the executable:
   ```cmd
   pyinstaller build_exe.spec
   ```

### Installation on Windows 11

1. **Copy the executable**: Copy `dist\PDFImgExtractorApp.exe` to your desired installation location
2. **Create shortcut**: Right-click the executable and select "Create shortcut"
3. **Pin to Start Menu**: Right-click the shortcut and select "Pin to Start"
4. **Add to Desktop**: Copy the shortcut to your desktop for easy access

### Features
- Extract images from PDF files with automatic compression
- Support for multiple image formats (WEBP, JPEG, PNG)
- Automatic image optimization to target file size (700KB default)
- Duplicate image detection and removal
- Page rendering fallback when no embedded images found
- Modern PyQt5 interface with frameless window design
- Drag-and-drop window movement
- Automatic output folder creation with PDF name

### System Requirements
- Windows 10/11 (64-bit)
- 4GB RAM minimum
- 100MB free disk space

### Usage
1. **Select PDF File**: Click "Select PDF File" to choose the PDF you want to extract images from
2. **Select Output Folder**: Click "Select Output Folder" to choose where to save the extracted images
3. **Start Extraction**: Click "Start Cooking" to begin the extraction process
4. **Results**: Images will be saved in a folder named `{PDF_NAME}_extractimages` in your selected output directory

### Troubleshooting

#### Common Issues:
1. **"Windows protected your PC" message**: Click "More info" then "Run anyway"
2. **Missing DLL errors**: Ensure you have the latest Visual C++ Redistributable installed
3. **Antivirus warnings**: Add the executable to your antivirus exclusion list

#### Build Issues:
1. **PyInstaller not found**: Run `pip install pyinstaller`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Permission errors**: Run Command Prompt as Administrator

### File Structure
```
PDFImgExtractorApp/
├── main.py                 # Main application file
├── build_exe.spec         # PyInstaller configuration
├── requirements.txt       # Python dependencies
├── build.bat             # Windows batch build script
├── build.ps1             # PowerShell build script
├── static/               # Static assets
│   ├── cover.png         # Application cover image
│   └── favicon.ico       # Application icon
└── src/                  # Source code modules
    ├── assets/           # Core functionality modules
    │   └── extract_pdf_images.py  # PDF image extraction logic
    └── uiitems/          # UI components
        ├── close_button.py
        ├── custom_alert.py
        ├── dash_line.py
        ├── preview_box.py
        ├── text_box.py
        └── notification_bar.py
```

### Dependencies
- PyQt5 - GUI framework
- PyMuPDF (fitz) - PDF processing
- Pillow (PIL) - Image processing and optimization
- pathlib - Path handling
- hashlib - Duplicate detection

### Support
For issues or questions, please check the troubleshooting section above or create an issue in the project repository. 