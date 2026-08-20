# GalaxyClassifier 参考资料 (REFERENCE.md)

> 2026-08-02 合并整理（原 ACRONYMS.md + README 数据来源表 + DEPLOY_ISSUES.md 历史问题合并而来）
> 用途：术语速查、数据来源、历史问题记录
> 相关文档：部署/架构见 [GUIDE.md](GUIDE.md)，验证结果见 [VALIDATION.md](VALIDATION.md)

---

## 1. 巡天速查（按"这个数据到底是干嘛的"组织）

### 光学巡天（拍可见光+近红外照片）

| 缩写 | 全称 | 通俗解释 | 在这个项目里的角色 |
|------|------|------|------|
| **PanSTARRS** | Panoramic Survey Telescope and Rapid Response System | 夏威夷1.8米望远镜，拍北天(Dec>-30°)的光学照片，5个颜色(grizy)，像素0.25" | **模型A的"菜"** — 模型A仅光学输入，PS 5波段切图是其唯一图像输入；模型B也用PS背景图 |
| **SDSS** | Sloan Digital Sky Survey | 新墨西哥州2.5米望远镜，拍北天(Dec>-20°)的光学照片+光谱 | **模型A的"老师"** — 用它的光谱分类(GALAXY/QSO/STAR)作为训练标签 |
| **DES** | Dark Energy Survey | 智利4米望远镜，拍南天，比PanSTARRS更深 | 如果要做南天，得用它替代PanSTARRS |
| **SkyMapper** | SkyMapper Southern Sky Survey | 澳大利亚1.3米望远镜，拍南天 | 南天替代方案之一 |

### 射电巡天（拍无线电波照片）

| 缩写 | 全称 | 通俗解释 | 在这个项目里的角色 |
|------|------|------|------|
| **VLASS** | Very Large Array Sky Survey | 美国VLA射电望远镜阵列，2-4GHz，拍全天天(Dec>-40°)，分辨率~2.5" | **模型B的"眼睛"** — 射电图像输入，看射电波段的辐射结构 |
| **FIRST** | Faint Images of the Radio Sky at Twenty-cm | VLA的1.4GHz巡天，覆盖北天~10,000平方度 | RGZ DR1的北天射电数据来源 |
| **EMU** | Evolutionary Map of the Universe | 澳大利亚ASKAP射电望远镜，南天巡天，0.9-1.7GHz | 项目要去迁移的目标巡天 |

### 红外巡天（拍热辐射照片）

| 缩写 | 全称 | 通俗解释 | 在这个项目里的角色 |
|------|------|------|------|
| **WISE** | Wide-field Infrared Survey Explorer | NASA太空望远镜，3.4/4.6/12/22微米四个红外波段 | **模型的辅助信息** — W1/W2两个红外星等作为Model A的额外输入 |
| **CatWISE** | CatWISE2020 | WISE+NEOWISE数据的综合星表，全天覆盖 | 论文从中获取W1/W2红外星等 |

---

## 2. 星表/数据集速查

| 缩写 | 全称 | 是什么 | 多少条 | 在这个项目里的角色 |
|------|------|------|:--:|------|
| **SDSSxWISE** | SDSS × CatWISE 交叉匹配 | SDSS光学源和WISE红外源的配对表 | 394万 | **Model A的训练标签来源** + WISE星等查询表 |
| **RGZ** | Radio Galaxy Zoo | 公民科学项目：志愿者看射电+红外图，标记"哪个星系是射电源的宿主" | DR1有9.9万北天+583南天 | **Model B的训练/验证标签** |
| **Norris2006** | Norris+ 2006 星表 | CDFS天区(南天一小块)的726个射电源，专家人工识别了光学对应体 | 594条 | **论文的独立测试集**（71个用于Model B验证） |
| **CIRADA** | Canadian Initiative for Radio Astronomy Data Analysis | VLASS的增强产品，含宿主星系识别表 | — | 尝试获取宿主标签但接口不通 |

---

## 3. 数据发布版本（同名数据的不同处理版本）

| 缩写 | 全称 | 发布时间 | 在这个项目里的角色 |
|------|------|:--:|------|
| **PS DR1** | PanSTARRS Data Release 1 | 2016年12月 | **论文训练用的版本**，Expansion硬盘上94GB |
| **PS DR2** | PanSTARRS Data Release 2 | 2019年1月 | **我们下载到的版本**，STScI服务器现在只返回这个 |
| **SDSS DR16** | SDSS Data Release 16 | 2020年 | SDSSxWISE星表使用的SDSS版本 |

---

## 4. 各数据集之间的关系

```
训练数据流:
  SDSS(光谱分类标签)
    ↓ 交叉匹配
  SDSSxWISE(394万源)
    ↓ panstamps下载
  PanSTARRS DR1 5波段切图(94GB) ───→ Model A 训练
    ↓
  Model A输出(三分类概率)
    +
  RGZ(公民科学宿主标签)
    +
  VLASS射电切图
    ↓
  Model B 训练

验证数据流:
  Norris2006(CDFS射电源) → VLASS切图 + PS切图 → Model B测试
  SDSSxWISE(北天源) → PS切图 → Model A测试
```

---

## 5. 最容易混淆的几对

| 容易混的 | 区别 |
|------|------|
| **PS DR1 vs DR2** | 同一个巡天(PanSTARRS)的两次数据发布，DR2是DR1的改进重处理。像Windows 10 vs Windows 11 |
| **SDSS vs PanSTARRS** | 两个不同的望远镜/巡天。SDSS拍光学+光谱，PanSTARRS只拍光学照片。SDSS更深但有光谱，PanSTARRS更广但只有图像 |
| **FIRST vs VLASS** | 两个不同的射电巡天，都是VLA望远镜。FIRST是1.4GHz(旧)，VLASS是2-4GHz(新)。RGZ的射电数据来自FIRST，模型B用的是VLASS |
| **RGZ vs Norris2006** | RGZ是公民科学标记(人多、主观但量大)，Norris2006是专家标记(人少、权威但量小)。论文用RGZ训练，用Norris测试 |
| **WISE vs CatWISE** | WISE是望远镜/巡天名称，CatWISE是处理后的星表产品。CatWISE = WISE + NEOWISE合并处理 |

---

## 6. 数据来源一览

| 数据 | 来源 | URL/工具 | 状态 |
|------|------|----------|:----:|
| VLASS射电FITS (51个) | CADC VLASS QuickLook | `nrao:VLASS/VLASS1.1.ql.*.fits` | ✅ |
| PanSTARRS光学FITS (71×5波段) | STScI PS1 Image Server | `panstamps` Python库 | ✅ |
| PanSTARRS北天测试FITS (30×5波段) | STScI PS1 Image Server | `panstamps` Python库 | ✅ |
| CatWISE红外星等 | VizieR II/365/catwise | `astroquery.vizier` | ⚠️ 87/1089 |
| Norris2006射电源星表 | VizieR J/ApJS/164/297 | `data/catalogs/norris06_table1.csv` | ✅ |
| VLASS CDFS分量星表 | CADC VLASS catalog | `data/catalogs/vlass_cdfs_components.csv` | ✅ |
| SDSSxWISE交叉表 (1.3GB) | 阿里云盘 | `data/catalogs/SDSS_clean_cat/SDSSxWISE_cat.tbl` | ✅ |
| 模型权重 (7个, 600MB) | GitHub master分支 | `crossmatch/models/*.pt` | ✅ |
| RGZ DR1 FIRST标签 (北天) | Zenodo 14195049 | `data/catalogs/DR1_FIRST_*.csv` | ✅ |
| RGZ DR1 ATLAS标签 (南天) | Zenodo 14195049 | `data/catalogs/DR1_ATLAS_*.csv` | ✅ |

---

## 7. EMU 迁移的标注资源现状 (2026-08 调研)

> 目标：为迁移到 EMU 巡天评估南天标注数据的量与质。核心关切：**标注过少会导致过拟合**。
> 结论速览：南天人工宿主标注当前约 4,000 量级（微调够、从零训练不够）；
> 模型 A 的南天光谱标签缺口应由 DESI/GAMA 补齐；RGZ-EMU 数据发布是 2026 年最大的量级跳变契机。

### 7.1 EMU 巡天本身 — 无宿主标注

| 项 | 状态 |
|------|------|
| EMU-PS1 先导巡天 | 270 deg², ~22万源, 图像+Selavy星表已公开 (CSIRO Data Access Portal) |
| EMU 正式巡天 | 2025年观测中 (2025-03: 293天区块已发布, 经CASDA陆续放出), **尚无正式DR1里程碑** |
| 宿主标注 | 射电图像/星表本身无宿主标注, 全部依赖外部标注项目 |

### 7.2 南天可用的标注数据 (按量级排序)

| 数据集 | 标注类型 | 数量 | 可用性 |
|------|------|------|:--:|
| **RGZ DR1 ATLAS** (Wong et al. 2025) | 宿主星系+射电形态 (公民科学, 可靠性0.83) | **583源** (文件实际行数, 旧文档写1394有误) | ✅ 已下载 `data/catalogs/DR1_ATLAS_*.csv` |
| **EMU DRAGNs** (Paper I) | 双瓣射电星系宿主 (unWISE认证, 2″半径, 3.2%误配) | 3,557个双瓣源, 3,182个(89.4%)有CATWISE2020宿主 | ✅ 已发布 (EMU-PS1天区) |
| **RGZ-EMU** (Tang 2025 / Vardoulaki 2025) | 射电形态标签 + WISE 3.4μm + DES光学宿主坐标 | Phase 1: 6,230张图, 1,435志愿者, **53,000+分类** | ⚠️ 2024-07启动, **数据尚未正式发布** (将进EMUCAT补充目录) |
| **GAMA23 深场** (Gürkan et al. 2022) | **光谱分类标签**: 1,636 SFG + 384复合 + 261 AGN + 13 LINER + 4,022未分类 | 39,812射电源, 20,007 GAMA对应体, 5,934光谱红移 | ✅ 已发布 (VizieR J/MNRAS/512/6104, 83 deg²) |
| RG-CAT (Gupta et al. 2024) | 21万源, 73% IR对应体 | ML产物, **非人工标注** | ⚠️ 只能当候选 |

### 7.3 对模型 A/B 的过拟合风险评估

| 模型 | 标签需求 | 南天现状 | 结论 |
|------|------|------|------|
| **模型 B** (宿主认证) | 宿主/非宿主标签 | 人工宿主标注 ~583(ATLAS)+3,182(DRAGNs) ≈ **4,000量级**; RGZ-EMU未发布 | 北天9.9万充足; **南天从零训练不够, 微调勉强** |
| **模型 A** (光学三分类) | GALAXY/QSO/STAR标签 | SDSS光谱仅北天; 南天可用 **GAMA23光谱 ~2,300条** (SFG+AGN+复合) + **DESI光谱巡天** (数百万条, 南天) | **DESI是补"光学+南天"标签缺口的关键** |

### 7.4 过拟合缓解策略 (标注量有限前提下)

1. **微调而非重训**: 南天用 2,000-4,000 标注微调, 保留北天预训练权重 (北天自洽已验证)
2. **负样本策略沿用**: 项目已有 RGZ_ROGUE 等负样本变体, 南天继续
3. **数据增强**: 现有 CDFS 71分量 + 北天30源 + 南天39源切图可旋转/翻转增强
4. **弱标注只取高共识子集**: RGZ-EMU 一图多分类是弱标注, 仅用多数投票共识高的宿主坐标
5. **关注 RGZ-EMU 数据发布** (2026年内): 南天宿主标注从千级跳到万级的契机

---

## 8. 部署问题历史（原 DEPLOY_ISSUES.md，2026-07-25 快照）

> 以下为项目早期记录的部署问题，多数已被后续会话解决/证实。详细验证结论见 [VALIDATION.md](VALIDATION.md)。

| # | 严重度 | 问题 | 当前状态 |
|---|:--:|------|------|
| 1 | 🔴 | SDSSxWISE与PS源天区不重叠, 无法获取WISE星等 | ✅ 已解决 — WISE asinh差异<0.008mag可忽略 (会话5) |
| 2 | 🔴 | RGZ训练星表完全丢失 (不在GitHub、不在云盘) | 🔴 未解决 — 用Norris06子集冒充训练集仅用于验证管线 |
| 3 | 🔴 | 模型A训练数据(FITS切图94GB)完全丢失 | ✅ 部分解决 — CDFS 71分量已重下; 完整重训仍需Expansion硬盘 |
| 4 | 🟡 | 模型A的SDSS分类标签未确认 | 🟡 — SDSSxWISE_cat.tbl 的 class_01 列待验证 |
| 5 | 🔵 | 云盘内容不完整(仅3文件) | 🔵 — 权重在GitHub master, 星表已重建 |
| 6 | 🔵 | 1343条训练笔记是模型B(Norris06测试)输出, 非模型A数据 | 🔵 — 性质已确认 |
| 7 | 🔵 | GitHub仓库结构: master含权重, handoff纯代码 | 🔵 — 结构确认 |

---

## 9. 目录重构记录

2026-07-31 目录重构（文档集中 + 数据按天区/类型归类 + 删除2GB重复数据）的完整映射表见
[STRUCTURE_CHANGELOG.md](STRUCTURE_CHANGELOG.md)。
