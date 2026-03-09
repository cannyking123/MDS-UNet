import torch
print(torch.__version__)  # 版本号
print(torch.cuda.is_available())  # CUDA 是否可用
print(torch.version.cuda)  # 显示实际使用的 CUDA 版本