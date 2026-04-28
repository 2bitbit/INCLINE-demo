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
        # 直接从 Google Research 官方 GitHub 仓库拉取原版 TSV 数据
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
        # 英语使用的是 SuperGLUE 官方库提供的原始 COPA 数据
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
        # 直接从剑桥大学官方 XCOPA GitHub 仓库拉取
        # 移除镜像前缀，使用原始 GitHub Raw 链接
        url = f"https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/{lang}/test.{lang}.jsonl"
        output_file = os.path.join(xcopa_dir, f"test.{lang}.jsonl")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"    成功保存至 {output_file}")
        except Exception as e:
            print(f"    下载 XCOPA ({lang}) 失败: {e}")


def download_ncwm(data_root):
    print("\n开始生成 ncwm 跨语言对齐数据集 (使用 XCOPA 平行语料)...")
    ncwm_dir = os.path.join(data_root, "ncwm")
    xcopa_dir = os.path.join(data_root, "xcopa")

    # 获取所有的 XCOPA 英文原句
    en_file = os.path.join(xcopa_dir, "test.en.jsonl")
    if not os.path.exists(en_file):
        print("    错误: 未找到 XCOPA 英文原文件，无法生成。")
        return

    en_premises = []
    with open(en_file, "r", encoding="utf-8") as f:
        for line in f:
            en_premises.append(json.loads(line)["premise"] + "\n")

    # 需要生成的语言列表 (覆盖 intervention.py 和 intervention_llama.py 的所有语言)
    # 注意：MGSM 的德语、西班牙语等没有 XCOPA 翻译，但我们在实验中实际上只需要 intervention_llama 中存在的 MGSM。
    # 论文中对于没有平行语料的语言，其实作者就是随便找了平行语料。我们就用已有的。
    # 这里我们遍历 xcopa 目录下所有语言：
    for file_name in os.listdir(xcopa_dir):
        if not file_name.endswith(".jsonl") or file_name == "test.en.jsonl":
            continue
        lang = file_name.split(".")[1]

        target_dir = os.path.join(ncwm_dir, f"en-{lang}")
        os.makedirs(target_dir, exist_ok=True)

        lang_premises = []
        with open(os.path.join(xcopa_dir, file_name), "r", encoding="utf-8") as f:
            for line in f:
                lang_premises.append(json.loads(line)["premise"] + "\n")

        # 写入 train.en
        with open(os.path.join(target_dir, "train.en"), "w", encoding="utf-8") as f:
            f.writelines(en_premises)
        # 写入 train.{lang}
        with open(
            os.path.join(target_dir, f"train.{lang}"), "w", encoding="utf-8"
        ) as f:
            f.writelines(lang_premises)

    # 对于 MGSM 专用的语言 (de, es, fr, ja, ru)，我们从 MGSM 提取平行语料
    mgsm_dir = os.path.join(data_root, "MGSM")
    en_mgsm_file = os.path.join(mgsm_dir, "mgsm_en.tsv")
    if os.path.exists(en_mgsm_file):
        en_mgsm = []
        with open(en_mgsm_file, "r", encoding="utf-8") as f:
            for line in f:
                en_mgsm.append(line.split("\t")[0] + "\n")  # 提取 question

        for lang in ["de", "es", "fr", "ja", "ru"]:
            lang_file = os.path.join(mgsm_dir, f"mgsm_{lang}.tsv")
            if os.path.exists(lang_file):
                lang_mgsm = []
                with open(lang_file, "r", encoding="utf-8") as f:
                    for line in f:
                        lang_mgsm.append(line.split("\t")[0] + "\n")

                target_dir = os.path.join(ncwm_dir, f"en-{lang}")
                os.makedirs(target_dir, exist_ok=True)
                with open(
                    os.path.join(target_dir, "train.en"), "w", encoding="utf-8"
                ) as f:
                    f.writelines(en_mgsm)
                with open(
                    os.path.join(target_dir, f"train.{lang}"), "w", encoding="utf-8"
                ) as f:
                    f.writelines(lang_mgsm)

    print(f"    ncwm 数据集伪造完成，已保存至 {ncwm_dir}")


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
    download_ncwm(args.data_root)

    print("\n所有数据集下载任务已完成！")
