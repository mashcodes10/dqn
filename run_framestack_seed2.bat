@echo off
call C:\Users\khanm16\AppData\Local\miniconda3\Scripts\activate.bat dqn
cd /d C:\Users\khanm16\Downloads\dqn
python train_minigrid_framestack.py --seed 2 --total-steps 2000000 --stack-k 10 --device cuda
pause
