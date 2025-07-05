@echo off
echo Building PDFImgExtractorApp executable...
echo.

REM Activate virtual environment if it exists
if exist "appenv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call appenv\Scripts\activate.bat
)

REM Install/upgrade required packages
echo Installing required packages...
pip install -r requirements.txt

REM Clean previous builds
echo Cleaning previous builds...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build

REM Build the executable
echo Building executable...
pyinstaller build_exe.spec

echo.
echo Build completed!
echo The executable is located in: dist\PDFImgExtractorApp.exe
echo.
pause 