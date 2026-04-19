@echo off
call C:\Users\khanm16\AppData\Local\miniconda3\Scripts\activate.bat dqn
cd /d C:\Users\khanm16\Downloads\dqn
python train_minigrid.py --seed 0 --total-steps 2000000 --device cuda
pause
