# EMU 模型B适用性测试方案 (EMU_TEST_PLAN.md)

> 创建：2026-08 | 目的：用 AS101 (EMU 先导巡天) 数据回答 **"模型B能否适用于 EMU"**
> 定位：这是**迁移实验的第一步**（拿基线+诊断），不是最终判决；最终判决靠第 6 步微调。
> 相关：数据登记见 [DATA_REGISTRY.md](DATA_REGISTRY.md)，缺口见 [DATA_GAPS.md](DATA_GAPS.md)，验证口径见 [VALIDATION.md](VALIDATION.md)

---

## 0. 结论速览

- **零样本直接跑 EMU，结果预期很差**（射电 18″ vs 训练 2.5″、光学 DES vs PS、A 南天退化），
  这是有文档依据的已知预期（VALIDATION 验证[1] CDFS 已演示过南天失败模式），不是 bug。
- 本方案第一步（步骤 1–5）的目标 = **跑通 EMU 数据管线 + 拿到南天基线 + 量化 A 概率通道影响**；
- 真正回答"B 适不适用 EMU"的是**步骤 6 微调**（用 EMU DRAGNs 标签微调 B 的射电/光学分支）。
- 手动改 A 概率（步骤 4 的 prior 组）**只能覆盖 B 三个输入通道之一**，定位为诊断实验。

---

## 0.5 执行状态 (2026-08 实测更新)

**已完成：**
- 4 个 AS101 文件已下载并核实：`taylor.0.fits`（✅ 18″ 圆波束，2″/px，44911×33569）、
  `taylor.1.fits`（18″，谱指数项，暂不用）、`UNSMOOTH`（椭圆束 13.9″×10.9″，暂不用）、
  `EMU_PS_CATALOG.xml`（VOTable，178,921 分量）
- `EMU_PS_CATALOG.xml` → `data/catalogs/emu/selavy_crossmatched.csv`（178,921 行）
  — **自带 WISE W1/W2 通量**（95.4%，已反向换算为 mJy，与 cal_luptitude 同零点）和
  **DES 匹配**（60.2%）、DESI（44%）——CatWISE 查询环节可完全跳过
- 候选池统计：SNR>8 且有 DES+WISE(<5″) 匹配的候选 **57,168** 个，其中延展>10″ 13,275 个
- 脚本就绪：选源、射电切图、三组对照推理、DES 下载（4 个，全部语法通过）

**待办（需你在浏览器/本机终端完成）：**
1. **下载 DRAGNs 目录**（3,557 行宿主标签）→ `data/catalogs/emu/dragns_catalog.fits`——
   论文 [arXiv:2507.23337](https://arxiv.org/abs/2507.23337) 的 data availability 里有存放地址
   （VizieR 或期刊补充材料），这是**选源和标签的唯一缺口**
2. **运行 DES 下载脚本**（本机终端，需网络）：`python tools/download_des_cutouts.py --sources ...`

**实测标定修正（相对初版方案）：**
- 拼图 2″/px → 射电切图默认视场 432″（=216px，恰好匹配 CenterCrop(216)，无需重采样）
- centroid/geodesic 缩放：VLASS 训练时 ×4（1″/px→0.25″/px），EMU 应为 **×8**（2″/px→0.25″/px），
  推理脚本已加 `--centroid-scale 8`
- WISE 输入直接用星表 w1flux/w2flux（占位符方案作废）

---

## 1. 需要的数据（全部来源与判断依据）

### 1.1 必须下载

| # | 数据 | 来源 | 用途 | 大小/说明 |
|---|------|------|------|------|
| 1 | `EMU_PS_IMAGE.taylor.0.fits` | CSIRO DAP，AS101 "derived data products" 集合 | 射电切图母图（18″ 统一分辨率 Stokes I 拼图） | 数百 MB~GB，需 OPAL 账号 |
| 2 | Selavy 分量星表（交叉匹配版） | 同上集合 | 100 源坐标入口（Component_name/RA/DEC + CATWISE/DES 对应体） | 含 CATWISE2020、DES DR1、DESI、WISExSuperCOSMOS 匹配 |
| 3 | EMU DRAGNs 目录 | [arXiv:2507.23337](https://arxiv.org/abs/2507.23337)（PASA 2025），VizieR | **模型B测试标签**（宿主位置）+ 步骤6微调的正样本 | 3,557 双瓣源，3,182 有 CATWISE2020 宿主 |

### 1.2 不下载（明确排除）

- `EMU_PS_IMAGE.taylor.1.fits`（谱指数，本轮用不上）
- `EMU_PS_IMAGE_UNSMOOTH.taylor.0.fits`（beam 随位置变化，预处理麻烦）
- AS101 "catalogues" 集合（原始逐波束 Selavy，被 1.1#2 的后处理交叉匹配版取代）
- AS101 "images and visibilities" 集合（visibility 原始数据，模型用不上）

### 1.3 模型额外需要的输入（不在 AS101 里）

| 输入 | 来源 | 说明 |
|------|------|------|
| 光学 5 波段切图 | **DES DR1**（NOIRLab Astro Data Lab 切图服务） | EMU 为南天，PanSTARRS 不覆盖；DES grizY→PS grizy 波段映射需小实验（README 注意点1） |
| WISE W1/W2 星等 | 星表自带则免查；否则 CatWISE（VizieR II/365） | 已有 `tools/query_catwise.py`；占位符影响 <0.008 mag（VALIDATION 会话5），首轮可用占位 |

---

## 2. 步骤 1：100 源选源（`tools/emu_select_sources.py`）

- **正样本 50**：从 DRAGNs 取有 CATWISE2020 宿主的源（固定 seed=42，可复现）
- **负样本 50**：从 Selavy 取**不在任何 DRAGN 30″ 内**的分量
- 输出与 `data/catalogs/preprocessed_cat/` **同 schema** 的 CSV（`VLASS_component_name` 字段装 EMU 分量名），
  使 `ps_dataset.py` 无需改动即可复用
- 幂等：输出已存在则跳过，`--force` 覆盖（DATA_REGISTRY 惯例）

## 3. 步骤 2：射电切图（`tools/download_emu_radio_cutouts.py`）

- 从 `EMU_PS_IMAGE.taylor.0.fits` 按 WCS 裁 `fields` 角秒（默认 5′）中心切图，存 `source_cutouts/emu_ps1/radio/{name}.fits`
- ⚠️ **分辨率重标定是待做项**：`CenterCrop(216)` 等常数按 VLASS 标定（README 注意点2），
  本方案先用现有常数近似跑基线，正式方案需按 18″ beam 重标定——这一步本身就是一个待测问题

## 4. 步骤 3：光学切图（DES，TODO）

- NOIRLab Astro Data Lab cutout 服务按宿主坐标下 DES DR1 grizY 5 波段，存 `source_cutouts/emu_ps1/optical/{name}/`，
  文件名加波段前缀（`1_g.fits`…`5_Y.fits`）保证排序=波段顺序
- **TODO**：下载后先核对 5 波段排序与 PS grizy 的对应关系（`create_PS_sample_cutout` 按文件名排序叠波段）

## 5. 步骤 4：三组 A 概率对照推理（`crossmatch/run_emu_inference.py --probs-mode ...`）

| 组 | A 概率来源 | 回答的问题 |
|:--:|------|------|
| raw | 真实跑 Model A（DES+WISE） | 端到端南天基线（预期差，作为对照锚点） |
| uniform | `[1/3, 1/3, 1/3]` | 中性占位：B 在没有 A 信息时的行为 |
| prior | 物理先验 `[0.45, 0.45, 0.10]`（可 `--prior` 改） | **手动改 A 概率能否恢复区分度**（射电选源≈非恒星假设） |

- 先验值**在下载数据前定死**，禁止按标签调参（防测试集泄漏）
- 模型 B 权重不动、模型定义不动；只替换 `ps_source_class_probs` 输入

## 6. 步骤 5：评估（口径=论文）

对每组输出 CSV 计算（对照 VALIDATION §3.3 论文口径）：

- 宿主召回率、非宿主特异性、Accuracy
- **分离度**：Host GT=1 平均 host_prob vs Host GT=0 平均 host_prob
  （论文 0.646 vs 0.016，差 0.630；CDFS 失败案例坍缩到 0.026——**差即失败信号**）
- 按形态分层（DRAGN 类型）报告：18″ 下双瓣退化最重，致密源退化最轻

> 三组对照的意义：组2 vs 组3 = "中性" vs "物理先验"的增益；
> 它量化 **B 对 A 概率通道的依赖**——这是手动改概率**唯一能回答**的子问题。

## 7. 步骤 6：微调 B（真正回答"B 适不适用 EMU"）——方案

### 7.1 为什么必须微调

B 的射电分支从没见过 18″ 分辨率图像（训练是 VLASS 2.5″）；光学分支没见过 DES。
零样本差 ≠ 任务做不了——18″ 下"质心附近有没有光学星系"信号反而更强，只是权重没见过这种图。

### 7.2 实验设计

| 项 | 方案 |
|------|------|
| 正样本 | DRAGNs 3,182 有宿主源（射电切图+宿主坐标） |
| 负样本 | Selavy 随机分量（远离 DRAGN），同 100 源选择逻辑放大到数千 |
| 训练/测试划分 | 按源随机 80/20，固定 seed；**测试集绝不参与任何调参** |
| 策略 A（推荐先做） | 冻结 `cnn_opt`(光学) 与 A 概率分支，只微调 `cnn_radio`(射电) + fc1/fc2，小学习率 1e-4~1e-5 |
| 策略 B | 全网络微调（含 cnn_opt），对照 A 看光学域差异贡献 |
| 损失/指标 | BCELoss；分离度 + host 召回（论文口径） |
| 预期 | 南天人工标注 ~4,000 量级 → 微调勉强够（REFERENCE §7.3 结论），别期待从零训练级效果 |

### 7.3 前置依赖

- 训练用射电切图（EMU taylor.0 裁切）与光学切图（DES）批量下载脚本（步骤 2/3 复用）
- `crossmatch/cross_matching.py` 的 DataLoader/训练循环适配（num_workers=0，Windows）
- 现有权重作初始化（迁移学习起点），不改 `detect_pos_dims` 架构

---

## 8. 执行顺序与验收标准

| 顺序 | 动作 | 验收 | 状态 |
|:--:|------|------|:--:|
| 1 | DAP 下载 1.1 三项 + 登记 DATA_REGISTRY | 文件落盘 | ✅ |
| 2 | `emu_select_sources.py` | 100 源 CSV（50+50） | ✅ |
| 3 | `download_emu_radio_cutouts.py` | 100 个射电 FITS | ✅ |
| 4 | DES 切图（1.3） | 100×5 光学 FITS | ✅ (legacy grz 冒烟版) |
| 5 | `run_emu_inference.py` 三组 | 3 个结果 CSV + 分离度报告 | ✅ |
| 6 | 微调 B（7.2 策略 A/B） | 分离度对比零样本，量化增益 | ⏳ 待执行 |

## 9. 实测结果（2026-08，零样本）

### 9.1 三组对照汇总

| 组 | N | Acc | Host 召回 | 非宿主特异性 | mu_host(GT1) | mu_host(GT0) | 分离度 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| raw（真实A） | 100 | 0.490 | 0/50 | 0.980 | 0.0000 | 0.0178 | **-0.018** |
| uniform | 100 | 0.490 | 0/50 | 0.980 | 0.0000 | 0.0177 | **-0.018** |
| prior [0.45,0.45,0.10] | 100 | 0.490 | 0/50 | 0.980 | 0.0000 | 0.0177 | **-0.018** |

- raw 组 Model A 坍塌到 STAR（平均 0.995）——与 VALIDATION 南天预期一致
- **A 概率通道对 B 输出几乎零影响**：跨模式最大差 raw↔uniform=0.0024、uniform↔prior=0.0004——
  在此 OOD 数据上 B 的决策被射电+光学图像分支完全主导。→ 回答"手动改 A 概率能否救 B"：**不能**（至少零样本下）。
- 唯一离群源 `EMU_PS_J204812.9-520851`（GT=0）三组都判 host_prob≈0.88（图像特征触发）
- 结果文件：`crossmatch/training_notes/emu_ps1_results_{raw,uniform,prior}.csv`

### 9.2 结论速读

1. **零样本 B 在 EMU 上彻底失败**（宿主召回 0/50，分离度 -0.018 反向）——与 CDFS 失败模式一致，属预期基线，**非结论**
2. **手动改 A 概率方案被证伪**（对 B 输出无实质影响）——诊断结论 1 已拿到，无需再做
3. 判决实验仍是**步骤 6 微调**：用 EMU 射电切图 + DRAGNs 标签微调 B 的射电分支
4. ⚠️ 数据口径提醒：本轮光学为 legacy des-dr1 **grz+占位**（通道3/5=z 复制），
   正式基线需 Data Lab token 重下真 grizY（`--backend datalab --token <TOKEN>`）；测试源 50/50 由 DRAGNs(人眼) 定义

### 9.3 XGBoost 模型A 对照（xgb 组，2026-08 追加）

| 项 | 结果 |
|------|------|
| 模型 | `data/cache/xgb_model_a.json`（WISE-only v1，特征 w1mpro/w2mpro/w1-w2，北天 SDSS 标签训练） |
| 接入 | `--probs-mode xgb`：星表 w1mag/w2mag → XGBoost 概率 → 喂 B（无需图像，100% 离线） |
| B 输出 | 分离度 **-0.0177**，与其他三组完全一致 → **A 概率通道对 B 无影响**（第 4 次确认） |
| **XGBoost A 自身分类** | 宿主源: GAL=0.180 QSO=0.187 **STAR=0.633**；非宿主: **GAL=0.453** QSO=0.268 STAR=0.278 |
| 对比 CNN A | 宿主源 CNN 坍塌 STAR=0.995 → **XGBoost 明显更健康且含 P/N 区分信号**（宿主偏 STAR、非宿主偏 GALAXY） |

> 启示：① XGBoost A 是比 CNN A 更可用的"光学分类器"候选（WISE-only、免图像、南天不坍塌）；
> ② 宿主源仍偏 STAR 可能与双瓣源 WISE 匹配对象有关（最近邻组件匹配，可能匹配到瓣间恒星）——
> 微调阶段建议用 DRAGNs 自带 CATWISE ID 反查真实宿主星等。
> 结果文件：`crossmatch/training_notes/emu_ps1_results_xgb.csv`

### 9.4 微调 B 射电分支实测（2026-08，判决实验 ✅）

| 项 | 值 |
|------|------|
| 数据 | 训练 1,187 源（600 DRAGN 宿主 + 587 远处 Selavy，排除测试 100 源），射电切图 216px@2″/px + DES grz 光学 |
| 策略 | 冻结 `cnn_opt` + 固定 A 概率=uniform logits；训练 `cnn_radio`+fc1+fc2；Adam lr=3e-4 wd=1e-4，梯度裁剪 1.0，10 epoch，best-val 保存 |
| 验证集 (10%) | **分离度 0.475**（acc 0.797，epoch 9） |
| **留出测试 100 源** | **分离度 -0.018 → +0.404；Host 召回 0/50 → 32/50 (64%)；acc 0.490 → 0.760** |

| 测试指标 | 零样本 | 微调后 | 论文参照 |
|------|:--:|:--:|:--:|
| Accuracy | 0.490 | **0.760** | 0.977 |
| Host 召回 | 0/50 | **32/50 (64%)** | 87.3% |
| 非宿主特异性 | 0.980 | 0.880 | 98.3% |
| mu_host GT1 / GT0 | 0.000 / 0.018 | **0.588 / 0.185** | 0.646 / 0.016 |
| 分离度 | -0.018 | **+0.404** | 0.630 |

**结论（判决）**：**B 方案在 EMU 上可学**——仅用 ~1,200 源微调射电分支，分离度从负值翻正到 0.40、宿主召回 0→64%。绝对数字低于论文（训练量小、光学为 grz 占位、epoch 9 单点检查点），方向已确立。

**已知差距（后续提升空间）**：① 训练量 1,200→全量 3,181+数万负样本；② 真 grizY 光学（Data Lab）；③ 更多 epoch + 学习率调度；④ 负样本标签噪声（WISE<5″ 匹配的远处分量可能含真实宿主）；⑤ DRAGN 宿主 WISE 星等应用 CATWISE ID 反查。

产物：`crossmatch/models/emu_finetuned_b.pt`、`crossmatch/finetune_emu_b.py`、
`crossmatch/training_notes/emu_ps1_results_uniform_finetuned.csv`

## 10. 预期结果与汇报口径

- 零样本：分离度大概率 <0.1（接近 CDFS 的 0.026），accuracy 虚高（全判非宿主）——**这是基线，不是结论**
- prior 组若显著优于 uniform 组 → "A 概率通道可被先验覆盖"（诊断结论 1）
- 微调后分离度若能回升到 0.3+ → "B 方案在 EMU 上可学"（判决结论 2）
- 若微调也无效 → 需考虑：18″ 下形态信息不足、需换特征（如用 Selavy 结构参数做传统特征）
