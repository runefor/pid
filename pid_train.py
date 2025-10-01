# /home/rune/dev/pid/pid_train.py

# import torch
import os
from PIL import Image
from pid_train.runners.train import main

# Pillow의 DecompressionBombWarning 비활성화
Image.MAX_IMAGE_PIXELS = None
os.environ['HYDRA_FULL_ERROR'] = '1'
# torch.compile 최적화: 스칼라 출력을 캡처하여 그래프 중단(graph break) 방지
# torch._dynamo.config.capture_scalar_outputs = True

if __name__ == "__main__":
    main()
