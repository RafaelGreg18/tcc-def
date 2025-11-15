import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
