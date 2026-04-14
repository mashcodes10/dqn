@echo off
call C:\Users\khanm16\AppData\Local\miniconda3\Scripts\activate.bat dqn
cd /d C:\Users\khanm16\Downloads\dqn
python train_atari.py --config configs/breakout_10m.json --seed 0 --device cuda
pause
