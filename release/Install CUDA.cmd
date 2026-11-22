@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -ProfileName cuda -DesktopShortcut
pause
