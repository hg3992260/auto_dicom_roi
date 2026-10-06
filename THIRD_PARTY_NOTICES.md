# Third-Party Notices / 第三方组件声明

本仓库包含来自以下开源项目的代码。各组件版权归其原作者所有，
按其原始许可条款使用与分发。再分发本仓库时请一并保留本文件与对应的许可原文。

---

## segment-anything (Meta Platforms, Inc.)

- 路径：`model_adapters/medsam/segment_anything/`
- 许可：**Apache License 2.0**
- 版权：`Copyright (c) Meta Platforms, Inc. and affiliates.`
- 上游：https://github.com/facebookresearch/segment-anything
- 许可原文：见 `LICENSE-APACHE-2.0.txt`

源码文件头部保留了原始版权与许可声明。**注意**：上游 `LICENSE` 文件未随本仓库分发，
再分发时请从上游获取完整文本。

## TinyViT (Microsoft)

- 路径：`model_adapters/medsam/segment_anything/modeling/tiny_vit_sam.py`
- 许可：**MIT License**
- 版权：`Copyright (c) 2022 Microsoft`
- 上游：https://github.com/microsoft/Cream/tree/main/TinyViT

## MedSAM (bowang-lab)

- 路径：`model_adapters/medsam/`
- 许可：**Apache License 2.0**
- 上游：https://github.com/bowang-lab/MedSAM

## SAM-Med2D (OpenGVLab)

- 路径：`model_adapters/sam_med2d/`
- 许可：**Apache License 2.0**
- 上游：https://github.com/OpenGVLab/SAM-Med2D

## SAM-Med3D (uni-medical)

- 路径：`model_adapters/sam_med3d/`
- 许可：**Apache License 2.0**
- 上游：https://github.com/uni-medical/SAM-Med3D

---

## 模型权重

**模型权重不在本仓库中分发。** README 中列出的 `.pth` / `.pt` 文件需由使用者
自行从各上游项目下载，并遵守其各自的许可与使用条款
（部分医学影像权重仅限研究用途）。

---

## 本仓库自身代码

除上述第三方组件外，本仓库其余代码的许可由仓库所有者决定
（当前未声明，默认保留所有权利）。
若需开源，请补充 `LICENSE` 文件；在此之前，请注意默认条款与
上述 Apache-2.0 组件的分发要求可能存在不一致。
