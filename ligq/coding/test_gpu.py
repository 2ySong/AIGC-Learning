import torch

# 1. 测试是否报错 (关键！)
print("PyTorch import successful!")

# 2. 测试 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")

# 3. 打印显卡名称
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # 做一个简单的计算测试
    x = torch.tensor([1.0]).cuda()
    print("CUDA Tensor calculation test passed.")