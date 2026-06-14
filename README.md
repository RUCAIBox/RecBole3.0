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
