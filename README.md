<div align="center">
<h1>
推理时的跨语言干预 (Inference-Time Cross-Lingual Intervention, INCLINE)
</h1>
</div>


### 1. 数据 (Data)

请下载下游任务数据并将其放置于 `./data/` 或 `/root/autodl-tmp/data` 目录中。

### 2. 干预实验 (Intervention)

我已经为您准备好了一键运行脚本！

在根目录下运行：

```bash
python run_demo.py
```

或者，如果您想单独运行具体的任务，请进入 `INCLINE` 目录执行：

对于判别式与生成式任务：
```bash
cd INCLINE
python intervention.py
```

对于 MGSM 任务：
```bash
cd INCLINE
python intervention_llama.py
```