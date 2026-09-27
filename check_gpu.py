import sys
import torch

def verify_gpu():
    print("==========================================")
    print("       ADA HPC GPU VERIFICATION          ")
    print("==========================================")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Device Count   : {device_count}")
        for i in range(device_count):
            print(f"Device [{i}]    : {torch.cuda.get_device_name(i)}")
            print(f"  Memory Total : {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
        
        # Simple Tensor Matrix Multiplication Test on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("GPU Matrix Multiplication Test: SUCCESS!")
    else:
        print("WARNING: CUDA is not available. PyTorch is running on CPU.")
    print("==========================================")

if __name__ == "__main__":
    verify_gpu()
