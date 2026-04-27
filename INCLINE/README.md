<div align="center">
<h1>
推理时的跨语言干预 (Inference-Time Cross-Lingual Intervention, INCLINE)
</h1>
</div>


### 1. 数据 (Data)

请下载下游任务数据并将其放置于 `./data/` 或 `/root/autodl-tmp/data` 目录中。

### 2. 干预实验 (Intervention)

对于判别式与生成式任务：

```
python intervention.py
```

对于 MGSM 任务：

```
python intervention_llama.py
```

### 引用 (Citation)
如果您发现这项工作对您有帮助，或使用了我们的数据，请考虑引用我们的论文：

```
@inproceedings{DBLP:conf/acl/WangWHB25,
  author       = {Weixuan Wang and
                  Minghao Wu and
                  Barry Haddow and
                  Alexandra Birch},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {Bridging the Language Gaps in Large Language Models with Inference-Time
                  Cross-Lingual Intervention},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2025, Vienna, Austria,
                  July 27 - August 1, 2025},
  pages        = {5418--5433},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.acl-long.270/},
  timestamp    = {Thu, 24 Jul 2025 21:25:39 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/WangWHB25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
### 致谢 (Acknowledgements)

这项工作由欧盟Horizon Europe (HE)研究与创新计划资助（项目编号 101070631），并由英国研究与创新局(UKRI)在英国政府的 HE 资助基金下资助（项目编号 10039436）。
