import subprocess
import importlib.metadata as md

def cmd(x):
    try:
        return subprocess.check_output(x, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"Not found / error: {e}"

print("=== NVIDIA / CUDA ===")
print(cmd("nvidia-smi"))
print("\n=== nvcc / CUDA Toolkit ===")
print(cmd("nvcc --version"))

print("\n=== PyTorch ===")
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch error:", e)

print("\n=== vLLM ===")
try:
    print("vllm:", md.version("vllm"))
except Exception as e:
    print("vllm not installed:", e)