import argparse
import subprocess
import sys
import os


def run_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"开始执行 {script_name}...")
    print(f"{'=' * 60}")

    # TODO: 以下执行需要实际的GPU环境。
    # 当在AutoDL上运行时，这将把模型加载到CUDA上并执行干预（intervention）。
    try:
        subprocess.run([sys.executable, script_name], check=True, cwd="INCLINE")
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 时发生错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="在AutoDL上运行INCLINE演示")
    parser.add_argument(
        "--data_root", type=str, default="/root/autodl-tmp/data", help="数据目录的路径"
    )
    args = parser.parse_args()

    # 设置DATA_ROOT环境变量，以便如果需要的话，脚本可以读取它。
    # 注意：目前的脚本中硬编码了 DATA_ROOT="/root/autodl-tmp/data"
    os.environ["DATA_ROOT"] = args.data_root

    print("=" * 60)
    print("INCLINE 演示运行程序")
    print(f"预期的数据根目录: {args.data_root}")
    print("=" * 60)
    print()

    # 1. 判别与生成任务
    run_script("intervention.py")

    # 2. MGSM 任务
    run_script("intervention_llama.py")

    print("\n所有演示均已完成。")
    print("# TODO: 请检查上方的输出指标，并将结果粘贴到您的记录中。")


if __name__ == "__main__":
    main()
