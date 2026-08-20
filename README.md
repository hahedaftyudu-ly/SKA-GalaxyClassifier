# GalaxyClassifier — 射电源光学宿主识别

用深度学习为**射电源**自动识别它的**光学宿主星系**。
原始硕士项目（Kangzhi Lou，中科院国家天文台），数据用 **VLASS**（射电）+
**PanSTARRS**（光学 grizy 5 波段）+ **WISE/CatWISE**（红外）。

> 本 README 是交接给本科生、为迁移到 **EMU** 巡天而整理的（2026-06）。
> 这是真正完整、含**训练好权重**的工程。另一个仓库
> `AstronomicalSourcesClassifier` 是引用本工程数据的较晚工作副本。
> 文档已合并精简（2026-08-02），详细内容见下方文档索引。

---

## 现状结论（一句话）

**模型方法在北天自洽有效（Model B 准确率 97.7%、宿主召回率 87.3%），
但模型 A 仅接受光学输入（PS 5波段+WISE），验证组合只有"光学+北天"（正常）
与"光学+南天"（退化）两个——当前模型只适用于 SDSS 北天训练分布，
南天迁移和射电选源（RGZ）都需要重新训练。**

## 文档索引（精简后 4+2 个文档）

| 文档 | 内容 |
|------|------|
| [docs/GUIDE.md](docs/GUIDE.md) | **部署技术参考**：环境/目录结构/模型架构/运行方式/时间线/源代码改动/路径修复/已知问题 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | **验证权威（PPT 用）**：六次验证/论文方法复现/泛化边界/失败教训/DR1与DR2/数据溯源/改进方向 |
| [docs/REFERENCE.md](docs/REFERENCE.md) | **参考资料**：巡天/星表速查、数据来源一览、部署问题历史 |
| [docs/DATA_REGISTRY.md](docs/DATA_REGISTRY.md) | **数据登记表**：所有数据批次（来源/日期/参数/规模），下载前先查，避免重复获取 |
| [docs/DATA_GAPS.md](docs/DATA_GAPS.md) | **数据缺口清单**（汇报用）：对照论文缺什么数据、缺因、获取途径 |
| [docs/STRUCTURE_CHANGELOG.md](docs/STRUCTURE_CHANGELOG.md) | 2026-07-31 目录重构映射表（旧路径→新路径） |
| [docs/GITHUB_SHARING_PLAN.md](docs/GITHUB_SHARING_PLAN.md) | GitHub 共享改造方案（已记录待执行：路径修复/权重方式/协作策略） |
| [docs/training_notes/evaluation_report.md](docs/training_notes/evaluation_report.md) | CDFS 评估报告（脚本生成） |

---

## 方法：两阶段、多模态的堆叠模型（概要）

```
  模型 A: CelestialClassficationNet —  仅光学输入 (PS 5波段图 + 2个WISE星等)
          → 输出 [Galaxy, QSO, STAR] 三分类概率
  模型 B: RadioOpticalCrossmatchModel — 射电切图(仅属于B) + PS背景图 + A的3类概率 + 位置特征
          → 输出 [非宿主, 宿主] 二分类概率
```

模型 B 把模型 A 的类别概率当作输入特征 → 两个模型**串联**，推理时先跑 A 再跑 B。
完整架构图与权重清单见 [docs/GUIDE.md](docs/GUIDE.md) §4。

---

## 一部分核心文件核心文件

| 文件 | 作用 |
|---|---|
| `mynetwork.py` | 模型 A 定义 |
| `crossmatch/model_crossmatch.py` | 模型 B 定义 |
| `FitsImageFolder.py`, `optical_dataset.py` | 模型 A 数据加载（5波段+WISE） |
| `crossmatch/ps_dataset.py`, `crossmatch/xmatch_dataset.py` | 模型 B 数据加载 |
| `main.py` | 模型 A 训练/推理 |
| `crossmatch/cross_matching.py` | 模型 B 训练 + **端到端推理**（产出结果 CSV） |
| `utils.py`, `image_tool.py`, `tools/utils.py` | 预处理/工具 |
| `tools/validate_pipeline.py` | 管线完整性检查（README 状态） |
| `crossmatch/models/*.pt` | 训练好的权重 |
| `data/catalogs/preprocessed_cat/*.csv`, `data/catalogs/SDSS_clean_cat_Duncan.*` | 星表与标签 |

---

## 资产清单：什么在、什么不在

**训练好的权重**（`crossmatch/models/`，可直接推理，无需重训）
- `opt_classification_model_wts.pt`（45MB，模型 A）
- `RGZ_all_negative_crossmatch_model_wts.pt`（90MB，模型 B 主用版）
- `RGZ1_1 / RGZ3_1 / RGZ_ROGUE1000 / RGZ_ROGUE_crossmatch_model_wts.pt`（负样本策略对比用）

 **训练/测试星表（位置+标签）**：`data/catalogs/preprocessed_cat/`
- 正样本（宿主）：`PS_p_RGZ_samples.csv`、`PS_p_Norris06_samples.csv`(测试集) …
- 负样本（假宿主）：`PS_n_samples_RGZ_all.csv`、`PS_n_Norris06_samples.csv` …

 **SDSS 三分类标签**（模型 A 的标签来源）：`data/catalogs/SDSS_clean_cat/SDSSxWISE_cat.tbl`(1.3GB)/`data/catalogs/SDSS_clean_cat_Duncan.csv`
 **WISE 交叉认证表**：`data/catalogs/*_wise_crossmatched.tbl`
 **射电形态/宿主真值等**：在 `~/Desktop/VLASS/`（`images_with_label/`、`2_Host_ID_Table/`）

 **唯一缺失：原始下载的切图图像**（模型真正吃的 fits 立方体）
- 光学切图曾在 `/Volumes/Expansion/VLASS/sources_for_Sean`（Expansion 移动硬盘，**当前未挂载**）
- 射电+PS 切图曾在 Linux 服务器 `/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/`（**已废弃**）
- 影响：① 想**重训**得按星表重下切图；② 但若只做**推理/应用 EMU**，只需为 EMU 新目标下载切图，现有权重可直接用。
-  先做的事：把 **Expansion 硬盘**插上看看 `sources_for_Sean` 还在不在——在的话连重下都省了。

---

## 已知需要修的地方（交接前）

1. **硬编码路径**：`main.py` 的 `src_root_path` 指向 `/Volumes/Expansion/...`；
   `cross_matching.py` 的 `VLASS_IMAGE_ROOT`/`PS_IMAGE_ROOT` 指向已废弃的 `/mnt/DataDisk/...`。需改成本机路径。
   （2026-07-31 重构已更新为新结构，见 [docs/STRUCTURE_CHANGELOG.md](docs/STRUCTURE_CHANGELOG.md)）
2. **catalog 目录名对不上**：代码引用 `data/preprocessed_cat_new/`，实际目录是 `data/preprocessed_cat/`。改路径即可（星表都在）。
3. **依赖陈旧**：`requirements.txt` 锁 torch 1.10 / tensorflow 2.7（Python 3.8 时代）。新机器/Python 3.12 需升级到 torch 2.x，少数旧 API（`pretrained=` → `weights=`）要顺手改。
4. **仓库较乱**：含重复入口（`main.py` 与 `optical/main.py`、`main_new.py`）、网络动物园（`models/coatnet.py` 等未必用到）、第三方 `d2l/`。交接核心见上表。

---

## 迁移到 EMU 的注意点

1. **光学巡天可能要换**：PanSTARRS 只覆盖 δ>-30°，**EMU 是南天**，深空区无 PanSTARRS。
   南天需换 DES/DELVE/Rubin/SkyMapper（光学）或 VHS/VIKING（近红外）；这会改变模型 A/B 的输入通道数与 WISE 之外的波段定义。
2. **射电预处理常数按 VLASS 标定**：`cross_matching.py` 的 `CenterCrop(216)` 等、以及切图视场/像素尺度对应 VLASS。ASKAP/EMU 分辨率不同，需重标定。
3. **WISE 星等输入**：模型 A/B 都用 2 个 WISE asinh 星等作辅助特征，EMU 区域同样需要 WISE/CatWISE 交叉认证（全天有覆盖，问题不大）。
4. **射电源星表字段**：原读 VLASS 的 `Component_name/RA/DEC`，换成 EMU 分量表对应列。
5. **模型 A 只吃光学输入**：迁移时无需关心射电图像通道，只需保证光学巡天波段与输入 5ch 对齐（DES grizY 可映射，需小实验验证）。
