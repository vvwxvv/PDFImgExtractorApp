Write-Host "Building PDFImgExtractorApp executable..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path "appenv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "appenv\Scripts\Activate.ps1"
}

# Install/upgrade required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

# Build the executable
Write-Host "Building executable..." -ForegroundColor Yellow
pyinstaller build_exe.spec

Write-Host ""
Write-Host "Build completed!" -ForegroundColor Green
Write-Host "The executable is located in: dist\PDFImgExtractorApp.exe" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue" 