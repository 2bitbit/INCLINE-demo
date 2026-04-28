#!/bin/bash

# ==============================================================================
# INCLINE 模型权重下载脚本
# 该脚本使用国内镜像和多线程加速库 (hf-transfer) 进行鲁棒下载，支持断点续传。
# ==============================================================================

# 1. 彻底斩断代理！不仅关 http/https，连全局代理 all_proxy 一起关！
unset http_proxy https_proxy all_proxy

# 2. 纯净环境指向国内镜像，并设置大模型数据盘缓存
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

echo "============================================================"
echo "开始下载第一个模型: bigscience/bloomz-7b1-mt"
echo "============================================================"
huggingface-cli download bigscience/bloomz-7b1-mt --resume-download

echo "============================================================"
echo "开始下载第二个模型: Mathoctopus/Parallel_7B"
echo "============================================================"
huggingface-cli download Mathoctopus/Parallel_7B --resume-download

echo "============================================================"
echo "所有必须的模型已成功下载至 /root/autodl-tmp/hf_cache！"
echo "============================================================"