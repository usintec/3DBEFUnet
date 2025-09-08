pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python.exe -m pip install --upgrade pip
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

