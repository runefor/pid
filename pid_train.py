# /home/rune/dev/pid/pid_train.py

# import torch
import os
from PIL import Image
from pid_train.runners.train import main

# Pillow의 DecompressionBombWarning 비활성화
Image.MAX_IMAGE_PIXELS = None
# os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['PROJECT_ROOT'] = os.getcwd()
# CUDA_LAUNCH_BLOCKING 이거 사용하면 디버깅은 되는데 속도가 느려진다고 한다.
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # TODO fasterrcnn_mobilenet_v3_large_fpn 이 모델 쓰면 중간에 터지던데, 특징 분류기의 shape이 올바른지 분석해봐야함.
# torch.compile 최적화: 스칼라 출력을 캡처하여 그래프 중단(graph break) 방지
# torch._dynamo.config.capture_scalar_outputs = True

if __name__ == "__main__":
    main()
