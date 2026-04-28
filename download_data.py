import os
import json
import argparse
from datasets import load_dataset


def download_mgsm(data_root):
    print("开始下载并处理 MGSM 数据集...")
    mgsm_dir = os.path.join(data_root, "MGSM")
    os.makedirs(mgsm_dir, exist_ok=True)

    # 对应的语言代号
    langs = ["en", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]

    for lang in langs:
        print(f"  正在处理 MGSM ({lang})...")
        try:
            # MGSM 在 HuggingFace 的原始名称为 juletxara/mgsm
            dataset = load_dataset("juletxara/mgsm", lang, split="test")
            output_file = os.path.join(mgsm_dir, f"mgsm_{lang}.tsv")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    # 假设原始列包含 'question' 和 'answer_number' (或其他目标列)
                    # 我们按原代码预期的格式按制表符分割写入
                    q = str(item["question"]).replace("\t", " ").replace("\n", " ")
                    a = str(item["answer_number"]).replace("\t", " ").replace("\n", " ")
                    f.write(f"{q}\t{a}\n")
            print(f"    成功保存至 {output_file}")
        except Exception as e:
            print(f"    下载或处理 MGSM ({lang}) 失败: {e}")


def download_xcopa(data_root):
    print("\n开始下载并处理 XCOPA 数据集...")
    xcopa_dir = os.path.join(data_root, "xcopa")
    os.makedirs(xcopa_dir, exist_ok=True)

    # XCOPA 语言代号
    langs = ["en", "et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]

    for lang in langs:
        print(f"  正在处理 XCOPA ({lang})...")
        try:
            # 英语使用 super_glue 的 copa 作为标准测试集，其他语言使用 xcopa
            if lang == "en":
                dataset = load_dataset("super_glue", "copa", split="validation")
            else:
                dataset = load_dataset("cambridgeltl/xcopa", lang, split="test")

            output_file = os.path.join(xcopa_dir, f"test.{lang}.jsonl")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    # 将 HuggingFace 格式转回原版 INCLINE 代码预期的 JSONL 格式
                    out_dict = {
                        "premise": item["premise"],
                        "choice1": item["choice1"],
                        "choice2": item["choice2"],
                        "label": item["label"],
                    }
                    # 写入为 jsonl 格式
                    f.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
            print(f"    成功保存至 {output_file}")
        except Exception as e:
            print(f"    下载或处理 XCOPA ({lang}) 失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 INCLINE 实验所需的数据集")
    parser.add_argument(
        "--data_root", type=str, default="/root/autodl-tmp/data", help="数据根目录路径"
    )
    args = parser.parse_args()

    print("=== INCLINE 数据集下载脚本 ===")
    print(f"数据存放目录: {args.data_root}")
    print("=" * 30)

    download_mgsm(args.data_root)
    download_xcopa(args.data_root)

    print("\n所有默认演示所需的数据集下载任务已完成！")
