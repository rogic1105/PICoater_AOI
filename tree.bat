@echo off
powershell -Command "tree /F /A > structure.txt"
echo Directory structure has been saved to structure.txt
pause