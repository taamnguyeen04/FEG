import torch
import os
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
# print(os.environ['CUDA_VISIBLE_DEVICES'])  # Đảm bảo giá trị này đúng
