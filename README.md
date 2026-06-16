![RecBole Logo](assets/logo.png)

--------------------------------------------------------------------------------
# RecBole (伯乐) 3.0
*“世有伯乐，然后有千里马。千里马常有，而伯乐不常有。”——韩愈《马说》*

[![Home](https://img.shields.io/badge/Home-RecBole-red)](https://github.com/RUCAIBox/RecBole)
[![arXiv](https://img.shields.io/badge/arXiv-RecBole-%23B21B1B)](https://arxiv.org/abs/2206.07351)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[RecBole 1.0] | [HomePage] | [Datasets] | [Paper] 

[RecBole 1.0]: https://github.com/RUCAIBox/RecBole
[HomePage]: https://recbole.io/
[Datasets]: https://github.com/RUCAIBox/RecDatasets
[Paper]: https://arxiv.org/abs/2206.07351

**Docs**: 
[Run](docs/running.md) |
[Architecture](docs/architecture-v2.md) |
[Develop](docs/component-authoring.md)

To support the emerging recommendation paradigms, we develop **RecBole3.0**, an LLM-inspired recommendation library covering representative up-to-date approaches.

## 1) Highlights

* **Flexible Architecture for Systematic Co-design**: 
We introduce a flexible architecture that decouples data, model, and optimization into self-contained units, facilitating both development and usage while adapting to emerging recommendation paradigms.
* **Comprehensive Model Support for Emerging Paradigms**:
RecBole3.0 currently supports representative models across 5 emerging paradigms, establishing a unified and reproducible benchmarking environment for fair comparisons.
* **Configurable Pipelines for Reproducible Experimentation**: 
RecBole3.0 enables flexible composition of datasets, models, trainers, and evaluators through configuration for consistent and reproducible experimentation across recommendation paradigms.

## 2) Implemented Models

Our library includes algorithms covering five major categories:

* **Generative Recommendation**: TIGER, LETTER, LC-Rec, ETEGRec, RPG, MiniOneRec, and CARE.
* **Scaling-based Recommendation**: RankMixer, HSTU, and LSRM.
* **Latent Reasoning Recommendation**: ReaRec, LARES, and CARE.
* **LLM-based Recommendation**: LLMRank, LLM4RS, LlamaRec, LC-Rec, MiniOneRec, E4SRec, and BIGRec.
* **Agent-based Recommendation**: AgentCF, AgentCF++, and STARec.

## 3) The Team

RecBole3.0 is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Enze Liu ([@BishopLiu](https://github.com/BishopLiu)), Zhuoxuan Li ([@ZhuoxuanLi-CS](https://github.com/ZhuoxuanLi-CS)), Dongze Wu ([@Joyful-bh](https://github.com/Joyful-bh)), Jiale Xu ([@JialeXu627](https://github.com/JialeXu627)), Xiaolei Wang ([@wxl1999](https://github.com/wxl1999)), Bowen Zheng ([@zhengbw0324](https://github.com/zhengbw0324)), Bingqian Li ([@Fotiligner](https://github.com/Fotiligner)), Kesha Ou ([@TayTroye](https://github.com/TayTroye)), and Chenghao Wu ([@wuchenghao0215](https://github.com/wuchenghao0215)).

## 4) Experimental Results

### Evaluation Settings:
```
model:
  history_max_length: 20
trainer:
  eval:
    protocol: full
    exclude_history: false
```

**Musical_Instruments**:

| Models     | Recall@5            | Recall@10 | NDCG@5 | NDCG@10 |
| ---------- | ------------------- | --------- | ------ | ------- |
| LSRM       | 0.0326              | 0.0529    | 0.0190 | 0.0255  |
| HSTU       | 0.0377              | 0.0614    | 0.0224 | 0.0300  |
| HSTU-Large | 0.0410              | 0.0659    | 0.0253 | 0.0333  |
| TIGER      | 0.0360              | 0.0562    | 0.0236 | 0.0301  |
| LETTER     | 0.0354              | 0.0552    | 0.0230 | 0.0294  |
| RPG        | 0.0369              | 0.0547    | 0.0244 | 0.0301  |
| LC-Rec     | 0.0329              | 0.0518    | 0.0216 | 0.0276  |
| E4SRec     | 0.0333              | 0.0529    | 0.0210 | 0.0273  | 
| LARES      | 0.0388              | 0.0610    | 0.0246 | 0.0318  |
| ReaRec     | 0.0345              | 0.0546    | 0.0219 | 0.0284  |

**Industrial_and_Scientific**:
| Models     | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| ---------- | -------  | --------- | ------ | ------- |
| LSRM       | 0.0240   | 0.0391    | 0.0142 | 0.0191  |
| HSTU       | 0.0288   | 0.0466    | 0.0165 | 0.0222  |
| HSTU-Large | 0.0325   | 0.0509    | 0.0196 | 0.0256  |
| TIGER      | 0.0271   | 0.0435    | 0.0177 | 0.0229  |
| LETTER     | 0.0248   | 0.0380    | 0.0160 | 0.0203  |
| RPG        | 0.0257   | 0.0384    | 0.0174 | 0.0215  |
| LC-Rec     | 0.0259   | 0.0401    | 0.0175 | 0.0220  |
| E4SRec     | 0.0242   | 0.0372    | 0.0156 | 0.0197  |
| LARES      | 0.0296   | 0.0466    | 0.0182 | 0.0236  |
| ReaRec     | 0.0237   | 0.0390    | 0.0153 | 0.0202  |

## RecBole Family Projects
The following table summarizes the open-source contributions of RecBole family projects on GitHub.


| **Projects**                                                 | **Stars**                                                    | **Forks**                                                    | **Issues**                                                   | **Pull requests**                                            |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [**RecBole**](https://github.com/RUCAIBox/RecBole)           | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole?style=social&logo=github)](https://github.com/RUCAIBox/RecBole/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole?style=social&logo=git)](https://github.com/RUCAIBox/RecBole/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole/pulls) |
| [**RecBole2.0**](https://github.com/RUCAIBox/RecBole2.0)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole2.0?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole2.0/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole2.0?style=social&logo=github)](https://github.com/RUCAIBox/RecBole2.0/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole2.0?style=social&logo=git)](https://github.com/RUCAIBox/RecBole2.0/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole2.0?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole2.0/pulls) |
| [**RecBole3.0**](https://github.com/RUCAIBox/RecBole3.0)     | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecBole3.0?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecBole3.0/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecBole3.0?style=social&logo=github)](https://github.com/RUCAIBox/RecBole3.0/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecBole3.0?style=social&logo=git)](https://github.com/RUCAIBox/RecBole3.0/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecBole3.0?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecBole3.0/pulls) |
| [**RecSysDatasets**](https://github.com/RUCAIBox/RecSysDatasets) | [![Stars](https://img.shields.io/github/stars/RUCAIBox/RecSysDatasets?style=social&logo=ReverbNation&logoColor=yellow)](https://github.com/RUCAIBox/RecSysDatasets/stargazers) | [![Forks](https://img.shields.io/github/forks/RUCAIBox/RecSysDatasets?style=social&logo=github)](https://github.com/RUCAIBox/RecSysDatasets/network/members) | [![Issues](https://img.shields.io/github/issues-closed/RUCAIBox/RecSysDatasets?style=social&logo=git)](https://github.com/RUCAIBox/RecSysDatasets/issues) | [![Pull requests](https://img.shields.io/github/issues-pr-closed/RUCAIBox/RecSysDatasets?style=social&logo=githubactions)](https://github.com/RUCAIBox/RecSysDatasets/pulls) |


## Cite
If you find RecBole useful for your research or development, please cite the following papers: [RecBole](https://arxiv.org/abs/2011.01731), [RecBole2.0](https://arxiv.org/pdf/2206.07351) and [RecBole3.0]().

```bibtex
@inproceedings{recbole,
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}

@article{recbole2.0,
  author    = {Wayne Xin Zhao and Yupeng Hou and Xingyu Pan and Chen Yang and Zeyu Zhang and Zihan Lin and Jingsen Zhang and Shuqing Bian and Jiakai Tang and Wenqi Sun and Yushuo Chen and Lanling Xu and Gaowei Zhang and Zhen Tian and Changxin Tian and Shanlei Mu and Xinyan Fan and Xu Chen and Ji{-}Rong Wen},
  title     = {RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  journal   = {arXiv preprint arXiv:2206.07351},
  year      = {2022}
}
```
