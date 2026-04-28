import sys
import os
import subprocess

def print_status(msg, status, color="green"):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }
    c = colors.get(color, colors["reset"])
    print(f"[{c}{status}{colors['reset']}] {msg}")

def check_python():
    print("=== Python 环境检查 ===")
    v = sys.version_info
    msg = f"Python 版本: {v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 8:
        print_status(msg, "OK")
    else:
        print_status(msg + " (建议 >= 3.8)", "WARN", "yellow")

def check_cuda_gpu():
    print("\n=== GPU & CUDA 检查 ===")
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print_status(f"发现 GPU 数量: {count}", "OK")
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  - GPU {i}: {name} (显存: {mem:.1f} GB)")
            print_status(f"PyTorch CUDA 版本: {torch.version.cuda}", "OK")
        else:
            print_status("未检测到可用的 GPU (CUDA不可用)!", "FAIL", "red")
    except ImportError:
        print_status("PyTorch 未安装!", "FAIL", "red")

def check_dependencies():
    print("\n=== 核心依赖包检查 ===")
    deps = ["torch", "transformers", "baukit", "datasets", "accelerate"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "未知版本")
            print_status(f"{dep} (版本: {version})", "OK")
        except ImportError:
            print_status(f"缺少依赖包: {dep}", "FAIL", "red")

def check_data_paths(data_root="/root/autodl-tmp/data"):
    print("\n=== 数据集完整性检查 ===")
    print(f"预期的数据根目录: {data_root}")
    
    if not os.path.exists(data_root):
        print_status(f"数据根目录 {data_root} 不存在!", "WARN", "yellow")
        print("提示: 请先运行 `python download_data.py` 下载数据。")
        return

    # 检查 XCOPA 和 MGSM
    xcopa_dir = os.path.join(data_root, "xcopa")
    mgsm_dir = os.path.join(data_root, "MGSM")
    
    if os.path.exists(mgsm_dir) and len(os.listdir(mgsm_dir)) > 0:
        print_status("MGSM 数据目录已就绪", "OK")
    else:
        print_status("MGSM 数据缺失", "WARN", "yellow")
        
    if os.path.exists(xcopa_dir) and len(os.listdir(xcopa_dir)) > 0:
        print_status("XCOPA 数据目录已就绪", "OK")
    else:
        print_status("XCOPA 数据缺失", "WARN", "yellow")

if __name__ == "__main__":
    print("=====================================")
    print("   INCLINE 复现环境自动诊断脚本")
    print("=====================================\n")
    check_python()
    check_cuda_gpu()
    check_dependencies()
    check_data_paths()
    print("\n=====================================")
    print("诊断结束。如果有 FAIL 或 WARN，请优先解决。")
