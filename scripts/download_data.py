import os
import json
import argparse
import urllib.request
import zipfile
import io


def download_mgsm(data_root):
    print("开始下载 MGSM 数据集...")
    mgsm_dir = os.path.join(data_root, "MGSM")
    os.makedirs(mgsm_dir, exist_ok=True)

    # 对应的语言代号
    langs = ["en", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]

    for lang in langs:
        print(f"  正在下载 MGSM ({lang})...")
        # 直接从 Google Research 官方仓库拉取原版 TSV 数据
        url = f"https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_{lang}.tsv"
        output_file = os.path.join(mgsm_dir, f"mgsm_{lang}.tsv")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"    成功保存至 {output_file}")
        except Exception as e:
            print(f"    下载 MGSM ({lang}) 失败: {e}")


def download_xcopa(data_root):
    print("\n开始下载 XCOPA 数据集...")
    xcopa_dir = os.path.join(data_root, "xcopa")
    os.makedirs(xcopa_dir, exist_ok=True)

    # XCOPA 语言代号 (不含英语)
    langs = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]

    print("  正在下载 English COPA (作为 XCOPA en)...")
    copa_url = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip"
    try:
        # 英语由于用的是原版的 COPA，所以从官方 SuperGLUE 库拉取并提取 val.jsonl
        req = urllib.request.urlopen(copa_url)
        with zipfile.ZipFile(io.BytesIO(req.read())) as z:
            with z.open("COPA/val.jsonl") as f_in:
                output_file = os.path.join(xcopa_dir, "test.en.jsonl")
                with open(output_file, "wb") as f_out:
                    f_out.write(f_in.read())
        print(f"    成功保存至 {output_file}")
    except Exception as e:
        print(f"    下载 English COPA 失败: {e}")

    for lang in langs:
        print(f"  正在下载 XCOPA ({lang})...")
        # 直接从剑桥大学官方 XCOPA 仓库拉取原版 JSONL 数据
        url = f"https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/{lang}/test.{lang}.jsonl"
        output_file = os.path.join(xcopa_dir, f"test.{lang}.jsonl")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"    成功保存至 {output_file}")
        except Exception as e:
            print(f"    下载 XCOPA ({lang}) 失败: {e}")


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
