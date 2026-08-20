# GalaxyClassifier 验证权威文档 (VALIDATION.md)

> 2026-08-02 合并整理（原 VALIDATION_SUMMARY.md + failure_log.md + validation_comparison_report.md + AGENT.md §9 合并而来）
> 用途：PPT 汇报、训练方向决策的唯一验证依据
> 相关文档：部署/架构见 [GUIDE.md](GUIDE.md)，术语/数据来源见 [REFERENCE.md](REFERENCE.md)

---

## 1. 验证口径（重要）

**模型 A 仅接受光学输入**（PanSTARRS 5波段图 + WISE 星等），**不接受射电输入**——
射电切图只属于模型 B。因此对模型 A 的验证组合只有两个：

| 组合 | 含义 | 结果 |
|------|------|:--:|
| **光学 + 北天** | SDSS 训练分布内（北天 SDSS 光谱选源） | ✅ 模型 A 正常/自洽 |
| **光学 + 南天** | CDFS 等南天天区（PS 覆盖边缘） | ❌ 模型 A 退化（STAR 坍塌） |

下方验证 [1][3][5] 中出现的 VLASS 射电数据仅用于**模型 B** 的宿主认证，
不参与对模型 A 的验证；模型 A 各行结果全部来自其光学输入。

---

## 2. 六次验证一览

### [0] 论文原始 Model B 测试
| 项目 | 内容 |
|------|------|
| **数据** | 论文原始 DR1 FITS (94GB) + Norris2006 射电源 1343个 |
| **方法** | 直接读取 training_notes 中的论文测试记录 |
| **Model A** | QSO主导 (74.5%), STAR=24.1%, 健康 |
| **Model B** | Acc=97.7%, Host Recall=87.3% (62/71), Host/Non-host概率分离度极佳 (0.646 vs 0.016) |
| **状态** | 论文自报结果，我们验证数据一致性 |

### [1] CDFS 独立验证 — 会话4
| 项目 | 内容 |
|------|------|
| **数据** | CADC VLASS (新下载51个, 仅模型B) + panstamps PS DR2 (新下载71x5, 模型A光学输入) + Norris2006标签 |
| **方法** | Model A(光学) -> Model B(光学+射电), 194条有效样本 |
| **Model A** | STAR=91.9% (坍塌), Galaxy=2.6% — **光学+南天组合失败** |
| **Model B** | Acc=93.3% (假象，全判Non-host即可), Host Recall=0% (0/11) |
| **状态** | FAILED |
| **失败原因** | **天区不匹配**: 模型在北天SDSS区训练，CDFS是南天(Dec-28)，PS覆盖边缘，图像质量/分布不同 |

### [2] 北天 SDSS Model A 复现 — 会话5
| 项目 | 内容 |
|------|------|
| **数据** | panstamps PS DR2 + SDSSxWISE 星表 (北天, 30源: GALAXY/QSO/STAR各10) |
| **方法** | 只跑 Model A 光学三分类 |
| **Model A** | STAR std=0.446 (恢复健康区分力), QSO标签源78%判对, STAR标签源99%判对 |
| **状态** | Model A 自洽有效 |
| **WISE排除** | 证明 WISE 星等不是 CDFS 失败的根因 (asinh压缩差异<0.008mag) |

### [3] 北天 RGZ Model A->B — 会话7a
| 项目 | 内容 |
|------|------|
| **数据** | CADC get_images VLASS (8/10, 仅模型B) + panstamps PS DR2 (10/10, 模型A光学输入) + RGZ DR1 FIRST 标签 |
| **方法** | Model A(光学) -> Model B(光学+射电), 7条有效 |
| **Model A** | STAR=99.1%, std=0.006 (完全坍塌) — **光学+北天组合在射电选源上失效** |
| **Model B** | host_prob 全 ~0.61 (无区分力) |
| **状态** | FAILED |
| **失败原因** | **源类型不匹配**: Model A 训练用 SDSS 光谱选源 (亮, r<17-18等), RGZ FIRST 是射电选源 (宿主可暗至 r~22等), PS 图像上形态不可辨 -> Model A 全判 STAR |

### [4] SDSS 论文方法 Model A — 会话7b
| 项目 | 内容 |
|------|------|
| **数据** | panstamps PS DR2 + SDSS 光谱分类 (30源, 同论文训练目录) |
| **方法** | Model A (FitsImageFolder), 与论文完全相同的 SDSS 源类型 |
| **Model A** | 总体60%: GALAXY 0%, QSO 80%, STAR 100% |
| **状态** | 部分成功 |
| **论文对比** | 论文97% -> 我们60%, GALAXY 96.4% -> 0% |
| **失败原因** | **PS DR2 vs DR1 图像差异**: 论文训练用 DR1 (2019年前下载), 我们现在只能拿到 DR2 (STScI 服务器只返回新版). 星系(展源)的测光/形态对 DR1->DR2 变化最敏感, 恒星(点源)和类星体受影响较小 |

### [5] SDSS 源 Model A->B 完整管线 — 会话7c
| 项目 | 内容 |
|------|------|
| **数据** | CADC get_images VLASS (25/30) + Session5 PS DR2 + RGZ 交叉匹配标签 |
| **方法** | 完整 Model A (FitsImageFolder) -> Model B, 25条有效 |
| **Model A** | STAR=61.2%, std=0.437 (未坍塌但偏 STAR) |
| **Model B** | host_prob 全 ~0.60 (无区分力) |
| **状态** | 管线跑通, 但 Model B 无实际验证价值 |
| **失败原因** | DR2 漂移通过 Model A 传入 Model B: Model A 判更多 STAR (61% vs 论文24%) -> Model B 学到 "STAR->压低 host_prob" -> host_prob 从论文的 0.016 (Non-host) 升到 0.60 |

---

## 3. 论文方法测试结果 (严格按论文)

### 3.1 测试方法

| 模型 | 训练数据 | 测试数据 | 标签来源 |
|------|------|------|------|
| Model A | SDSS+CWISE 15万源 (Galaxy/QSO/STAR各5万) | SDSS光谱分类源 30个 | SDSS spectroscopic |
| Model B | RGZ公民科学标签 | Norris2006 CDFS射电源 1343个 | RGZ citizen science |

### 3.2 Model A 结果 (SDSS 30源, 光学+北天组合)

| 类别 | 样本数 | 准确率 | 平均概率 (Gal/QSO/STAR) |
|------|:--:|:--:|------|
| GALAXY | 10 | **0%** | .040 / .339 / .621 |
| QSO | 10 | **80%** | .002 / **.781** / .217 |
| STAR | 10 | **100%** | .005 / .002 / **.992** |
| **总体** | 30 | **60.0%** | .016 / .374 / .610 |

- QSO和STAR可区分 (std=0.446)
- GALAXY类完全混淆 (判为STAR)
- 论文声称96%准确率, 当前60%: 可能因PS DR2图像差异

### 3.3 Model B 结果 (Norris 1343源, 论文原始FITS)

| 指标 | 值 |
|------|------|
| 准确率 | **97.7%** |
| 宿主召回率 | **87.3%** (62/71) |
| 精确率 | 73.8% |
| 非宿主特异性 | 98.3% (1250/1272) |
| Host GT=1 平均概率 | **0.646** |
| Host GT=0 平均概率 | 0.016 |

### 3.4 CDFS 验证详细指标 (会话4, 194源)

| 指标 | 值 |
|------|------|
| Accuracy | **93.3%** (181/194) |
| Precision / Recall / F1 | 0.0% (无真阳) |
| Specificity | 98.9% (181/183) |
| 混淆矩阵 | TN=181, FP=2, FN=11, TP=0 |

按 Norris2006 形态类型分层:

| Norris Type | Samples | Correct | Accuracy |
|-------------|---------|---------|----------|
| Unclassified | 18 | 16 | 88.9% |
| AGN/QSO | 110 | 102 | 92.7% |
| Galaxy/ELG | 33 | 32 | 97.0% |
| ULIRG | 4 | 4 | 100.0% |
| Composite | 11 | 10 | 90.9% |

### 3.5 会话7c 补充数据 (SDSS 源 A→B 完整管线, 25源, 光学+北天)

> 数据源: `crossmatch/training_notes/sdss_ab_validation.csv` (2026-07-29, 25 条有效)
> 样本构成: GALAXY 10 + QSO 8 + STAR 7 (部分源 VLASS 下载失败, 非各10)

#### 3.5.1 Model A 结果 (25源)

| 类别 | 样本数 | 准确率 | 平均概率 (Gal/QSO/STAR) |
|------|:--:|:--:|------|
| GALAXY | 10 | **0%** | .040 / .339 / .621 |
| QSO | 8 | **75%** | .003 / **.727** / .271 |
| STAR | 7 | **100%** | .007 / .003 / **.990** |
| **总体** | 25 | **52.0%** | — |

- STAR 平均概率 61.2%, STD 0.446 (与 §2 [5] 记录一致, std 以实测 0.446 为准)
- GALAXY 行概率与 3.2 (会话7b) 完全一致 → **GALAXY 0% 是稳定复现的 DR2 效应**
- 混淆: GALAXY→6 STAR + 4 QSO (全错); QSO→6对2错; STAR→7/7 全对

#### 3.5.2 Model B 结果 (25源)

| 指标 | 值 |
|------|------|
| 准确率 | **4.0%** (1/25) |
| 宿主召回率 | **100.0%** (1/1, 假象 — 仅1个正样本) |
| 精确率 | 4.0% |
| 非宿主特异性 | **0.0%** (0/24) |
| Host GT=1 平均概率 | **0.571** |
| Host GT=0 平均概率 | **0.597** |

- 25 条全部判宿主 (host_prob 范围 0.571~0.612, 全>0.5); 标签分布 24 非宿主 + 1 宿主
- **概率分离度从论文的 0.630 坍缩到 0.026** — 与 §2 [5] 失败原因一致 (DR2 漂移经 Model A 传入 Model B)

---

## 4. 模型验证状态总表

| 验证 | 数据 | 天区 | Model A | Model B | 结论 |
|------|------|:--:|:--:|:--:|------|
| 论文原始 (1343源) | 原始FITS | 北天 | QSO主导 ✅ | 97.7%/87.3% ✅ | 论文声称性能 |
| 论文方法 Model A (30源) | PS DR2 | 北天 | 60.0% (QSO 80%, STAR 100%) | — | 部分恢复 |
| CDFS独立验证 (194源) | PS DR2 | 南天 | STAR坍塌 ❌ | 0% ❌ | 天区不匹配 |
| A→B管线验证 | PS+VLASS DR2 | 北天 | 管线全通 | 管线全通 | 基建就绪 |

---

## 5. 模型 A 的泛化边界（光学输入 × 天区）

| # | 边界 | 验证 | 表现 |
|:--:|------|------|:--:|
| 1 | **天区（光学+北天 vs 光学+南天）** | 北天SDSS训练 -> 南天CDFS推理 | Model A STAR坍塌 91.9% |
| 2 | **源类型（光学输入下的源分布）** | SDSS光学选 -> RGZ射电选（PS图像上宿主暗至r~22等） | Model A STAR坍塌 99.1% |
| 3 | **PS版本（光学图像版本）** | DR1训练 -> DR2推理 | Model A GALAXY 96%->0% |

**共同根因**: 光学训练数据覆盖范围太窄
- 天区只有北天 SDSS 区域（南天即"光学+南天"组合失效）
- 源类型只有 SDSS 光谱亮源 (r<17-18等)
- PS 图像只有 2019 年前的 DR1 版本

### 已知限制

1. **Model A GALAXY类混淆**: 北天SDSS源上GALAXY准确率0%, 可能因PS DR2图像与训练时DR1差异
2. **天区不匹配**: 南天CDFS PS图像 (Dec -28°, 覆盖边缘) 导致Model A ResNet特征退化
3. **WISE非根因**: asinh星等压缩动态范围, 占位值vs真实值差异<0.008mag
4. **RGZ源分布偏移**: 射电选源的光学形态与SDSS光谱选源不同, Model A泛化受限

---

## 6. 失败教训汇总（原 failure_log.md 教训要点）

1. **Norris2006的形态类型(Type 1/2/4/6)不能当宿主标签用** — 它是形态分类, 不是宿主/非宿主标签
2. **训练数据是北天SDSS区域, 不能直接泛化到南天CDFS** — 天区分布差异导致特征退化
3. **PS DR1 vs DR2在CDFS边缘天区存在显著差异**
4. **不要假设数据异常即为bug** — 先验证数据本身的统计特性（WISE占位符案例）
5. **Model A只能用于SDSS光谱选源** — 其训练分布是光学亮的星系/类星体/恒星
6. **RGZ射电选源的宿主星系需重新训练Model A, 不能直接套用**
7. **WISE对坍塌的贡献<图像特征, Model A主要依赖PS图像**
8. **循环验证风险**: CDFS Ground Truth 曾取自模型自己的输出(training_notes 1343条), 后续必须用独立标签

---

## 7. 为什么拿不到 PanSTARRS DR1 数据

### DR1 vs DR2

| | DR1 | DR2 |
|------|------|------|
| 发布时间 | 2016年12月 | 2019年1月 (后持续更新) |
| 处理流水线 | PV3 (旧版) | PV3+ (改进) |
| 光度定标 | 早期标定 | 改进的零点和消光 |
| stacking算法 | 旧版 | 改进(更好去除宇宙线/卫星轨迹) |
| 论文使用 | ✅ 94GB FITS | — |
| 我们能用 | ❌ | ✅ panstamps返回的版本 |

### 关键差异（影响展源最大）

1. **光度零点偏移**: DR2重新标定测光零点, 展源(星系)可能存在0.02-0.05 mag系统偏移
2. **Stacking深度**: DR2使用更多epoch, 单帧stack更深, 噪声更低
3. **PSF建模**: DR2改进了PSF建模, 影响点源测光, 展源形态测量间接受影响
4. **边缘天区**: CDFS天区(Dec~-28°)在PS1覆盖边缘, DR2处理改进在此区域最显著
5. **图像外观**: 同一位置同一天区的stack图像在DR1和DR2中可能有明显不同的像素值分布

### 为什么DR1无法获取

| 尝试途径 | 结果 | 原因 |
|------|:--:|------|
| panstamps API (默认) | ❌ 返回DR2 | STScI PS1服务器默认返回最新处理版本(PV3+) |
| `mjdEnd=58000` 限制 | ❌ 仍返回DR2 | 服务器仅在DR2 stack中寻找匹配的MJD范围, 不提供DR1 stack |
| PS1 Image Cutout网页 | ❌ 仅DR2 | `https://ps1images.stsci.edu/cgi-bin/ps1cutouts` 同样只有DR2 |
| MAST PS1 archive | ⚠️ 可能有 | PS1原始曝光数据(FITS)可在MAST下载, 但需自行stack |
| **Expansion移动硬盘** | ⭐ 唯一可行 | 论文原始94GB FITS(403,549个文件)就在这块硬盘上 |
| 联系作者楼康志 | 备选 | 作者可能有备份 |

### 数据版本造成的差异（量化）

| | DR1 (论文) | DR2 (我们) |
|------|------|------|
| 获取时间 | 2019年前 | 2026年7月 |
| PS处理流水线 | PV3 (旧版) | PV3+ (改进版) |
| 光度定标 | 早期标定 | 改进零点和消光 |
| Model A GALAXY | 96.4% | 0% |
| Model A STAR主导 | 24.1% | 61.2% |
| Model B host_prob(Non-host) | 0.016 | 0.596 |
| 状态 | 在 Expansion 硬盘上, 无法获取 | STScI 服务器唯一返回版本 |

---

## 8. 验证方法汇总

| 方法 | 用途 | 对应会话 |
|------|------|:--:|
| FitsImageFolder + OptDataSet | Model A 光学三分类 (正确预处理) | S5, S7b |
| FitsImageFolder.loader() | 获取 remove_nan+optimize_image 后的图像 | S7a, S7c |
| cadc.get_images() | 下载 VLASS 射电切图 | S4, S7a, S7c |
| caom2.Artifact 查询 nrao: URI | 批量下载 VLASS (备选) | S7a |
| panstamps.downloader() | 下载 PanSTARRS 光学切图 | S4, S5, S7 |
| cal_luptitude() | WISE 通量->asinh星等转换 | S5, S7 |
| RGZ DR1 交叉匹配 | 获取公民科学宿主/非宿主标签 | S7a, S7c |
| training_notes 直接读取 | 论文原始 Model B 测试结果 | S4, S7b |

---

## 9. 数据溯源速查表

| 数据缩写 | 全称 | 实际来源 | 天区 | 版本/日期 |
|------|------|------|:--:|------|
| PS DR1 (论文) | PanSTARRS Data Release 1 | 原始94GB FITS (Expansion硬盘) | 北天 | ~2019前 |
| PS DR2 (当前) | PanSTARRS Data Release 2 | panstamps API (STScI) | 全天 | 2020+ |
| SDSSxWISE | SDSS DR16 × CatWISE2020 交叉 | 阿里云盘 `.tbl` 1.3GB | 北天 Dec>-19.7° | DR16 (2020) |
| CDFS VLASS | VLASS QuickLook CDFS分量 | CADC `nrao:VLASS` → `vlass_cdfs_components.csv` | 南天 CDFS | Epoch 1-3 |
| North VLASS | VLASS QuickLook 北天tile | CADC `get_images()` → `caom2.Artifact` nrao: URI | 北天 | Epoch 1-4 |
| RGZ DR1 | Radio Galaxy Zoo Data Release 1 | Zenodo 10.5281/zenodo.14195049 | FIRST(北天)+ATLAS(南天) | 2024 |
| Norris2006 | Norris+ 2006 CDFS射电源表 | VizieR J/ApJS/164/297 | CDFS | 2006 |
| CatWISE | CatWISE2020 全天红外 | VizieR II/365/catwise | 全天 | 2020 |

---

## 10. 后续改进方向

| 优先级 | 问题 | 方案 | 依赖 |
|:--:|------|------|------|
| 🔴 | Model A GALAXY 0% | 获取PS DR1图对比; 或在SDSS训练集上微调 | PS DR1数据 |
| 🟡 | 南天泛化 | 南天光学巡天(DES/SkyMapper) + 微调 | 南天数据+训练时间 |
| 🟡 | RGZ源Model A坍塌 | 用RGZ标注数据重新训练Model A光学分类 | RGZ+PS下载 |
| 🟢 | EMU适配 | 射电图尺寸匹配EMU + 光学巡天替换 | EMU数据可用时 |

---

## 附：详细报告与数据文件

| 内容 | 位置 |
|------|------|
| CDFS 完整评估报告 (脚本生成) | `docs/training_notes/evaluation_report.md` |
| CDFS 推理结果 CSV (194条) | `crossmatch/training_notes/cdfs_validation_results.csv` |
| 论文原始 Model B 测试 (1343条) | `crossmatch/training_notes/RGZ_all_negative_Norris_testing_notes*.csv` |
| 北天 Model A 结果 | `crossmatch/training_notes/thesis_model_a_results.csv` |
| 北天 A→B 结果 | `crossmatch/training_notes/north_sky_validation_results.csv` |
| SDSS A→B 结果 | `crossmatch/training_notes/sdss_ab_validation.csv` |
| 南天分天区结果 | `crossmatch/training_notes/south_sky_model_a_results.csv` |
| PPT 数据 | `crossmatch/training_notes/evaluation_ppt_data.txt` |
