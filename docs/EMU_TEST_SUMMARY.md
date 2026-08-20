# EMU 模型B适用性测试总结 (EMU_TEST_SUMMARY.md)

> 2026-08 | 目的：回答 **"模型 B 能否适用于 EMU 巡天数据"**
> 数据：AS101 (EMU 先导巡天) 944 MHz 图像 + DRAGNs 人工认证宿主标签
> 详细过程见 [EMU_TEST_PLAN.md](EMU_TEST_PLAN.md)，验证口径见 [VALIDATION.md](VALIDATION.md)

---

## 1. 一句话结论

**零样本直接跑 EMU 失败（预期内）；三种"改 A"方法都不能救零样本的 B；
但微调 B 的射电分支后，留出测试集分离度从 -0.018 → +0.404、宿主召回 0 → 64%——
"B 方案在 EMU 上可学"成立（判决结论）。**

---

## 2. 数据准备（全部就位）

| 数据 | 内容 | 规模 |
|------|------|:--:|
| `EMU_PS_IMAGE.taylor.0.fits` | 18″ 圆波束 Stokes I 拼图，2″/px | 44911×33569 |
| `EMU_PS_CATALOG.xml` → `selavy_crossmatched.csv` | 178,921 分量：**95.4% 自带 WISE 通量**、60.2% DES、44% DESI | 178,921 行 |
| DRAGNs 星表 (Cambridge supp Table7.fits) | 3,557 双瓣源，3,181 有 CATWISE 宿主，含形态 Tags | 3,557 行 |
| 测试集 | 50 DRAGN 宿主 + 50 远处 Selavy（seed=42，WISE<5″） | 100 源 |
| 切图 | 射电 216px @ 2″/px（=432″）；光学 DES grz 240px @ 0.25″/px（⚠️ 通道3/5=z占位） | 测试100 + 训练1200 |

**关键标定修正**：拼图 2″/px → 射电切图 432″=216px 免重采样；centroid/geodesic 缩放 ×4→**×8**；WISE 输入直取星表（免 CatWISE 查询）。

---

## 3. 方法一：零样本三组对照（诊断"A 概率通道"）

| 组 | A 概率来源 | 分离度 | 说明 |
|:--:|------|:--:|------|
| raw | CNN A 真实输出 | -0.018 | Model A 南天坍塌（STAR=0.995，验证[1]预期） |
| uniform | [⅓,⅓,⅓] | -0.018 | 中性占位 |
| prior | [0.45,0.45,0.10] 物理先验 | -0.018 | 射电选源≈非恒星 |

**结果**：宿主召回 0/50；跨模式最大差 <0.0024——**A 概率通道对 B 输出几乎零影响**（被射电+光学图像分支主导）。

**结论**：零样本 B 在 EMU 上彻底失败（预期基线）；"手动改 A 概率"方案被证伪。

## 4. 方法二：XGBoost 模型 A（WISE-only）

- 特征：W1/W2 星等（**星表自带，免图像、100% 离线**）
- B 输出：分离度仍 -0.018（第 4 次确认 A 通道无影响）
- **XGBoost A 自身**：宿主源 STAR=0.633（CNN 是 0.995）、非宿主 GAL=0.453 —— **比 CNN A 健康，且输出含 P/N 区分信号**
- 结论：XGBoost A 是更可用的"A"候选（但改它救不了 B，因为 B 不吃 A 通道）

## 5. 方法三：微调 B 射电分支（判决实验 ✅）

| 项 | 配置 |
|------|------|
| 训练集 | 1,187 源（600 DRAGN 宿主 + 587 远处 Selavy，**排除测试 100**） |
| 冻结 | `cnn_opt`（光学，`.eval()` 保 BN 运行统计）+ 固定 uniform A 概率 |
| 训练 | `cnn_radio` + fc1 + fc2；Adam lr=3e-4 wd=1e-4；梯度裁剪 1.0；10 epoch，best-val 保存 |
| 验证 (10%) | 分离度 **0.475**（acc 0.797） |

**留出测试 100 源（判决数字）：**

| 指标 | 零样本 | 微调后 | 论文参照 |
|------|:--:|:--:|:--:|
| Accuracy | 0.490 | **0.760** | 0.977 |
| Host 召回 | 0/50 | **32/50 (64%)** | 87.3% |
| 非宿主特异性 | 0.980 | 0.880 | 98.3% |
| mu_host GT1 / GT0 | 0.000 / 0.018 | **0.588 / 0.185** | 0.646 / 0.016 |
| 分离度 | -0.018 | **+0.404** | 0.630 |

**结论**：**B 方案在 EMU 上可学**——~1,200 源微调射电分支即翻正分离度、召回 0→64%。

---

## 6. 方法论主线（三种方法的关系）

```
问题: "B 适用于 EMU 吗?"
 ├─ 方法一 (零样本+改A概率): 诊断 A 概率通道 → 证伪"改A能救B", 拿基线 (-0.018)
 ├─ 方法二 (XGBoost A): 诊断替代A的可行性 → A 可替换且更健康, 但救不了 B
 └─ 方法三 (微调射电分支): 判决实验 → B 可学 (+0.404, 64%)
```

每个方法回答一个子问题；方法三是最终答案，方法一二的结论是"为什么必须走方法三"的证据链。

## 7. 工程踩坑记录（已修复，脚本已固化）

1. VOTable 255MB 解析（astropy parse ~25s）
2. FITS 字符串列是 bytes+空格填充 → CATWISE ID 解析需 decode+strip
3. 拼图 4D (1,1,Ny,Nx) → Cutout2D 需 squeeze 成 2D
4. FitsImageSet 的 VLASS 名称解码不兼容 EMU 名 → EmuFitsImageSet 覆写
5. 训练 NaN：DES 全零平面 → optimize_image 0/0 → 双重 nan_to_num
6. 冻结分支 BN 需 eval 模式（批统计漂移）
7. DES legacy 服务仅 grz 三波段 + 限速 + 偶发 500（13/1200 剔除）

## 8. 工具链（全部幂等可复现）

| 脚本 | 作用 |
|------|------|
| `tools/emu_convert_catalog.py` | VOTable → CSV（含 WISE mJy 换算） |
| `tools/emu_select_sources.py` | 选源（--exclude-csv 防泄漏） |
| `tools/download_emu_radio_cutouts.py` | 拼图裁切 |
| `tools/download_des_cutouts.py` | DES 光学（legacy grz / datalab grizY） |
| `crossmatch/run_emu_inference.py` | 推理（--probs-mode raw/uniform/prior/xgb） |
| `tools/eval_emu_results.py` | 四组对比评估 |
| `crossmatch/finetune_emu_b.py` | 微调射电分支（含预处理缓存） |

## 9. 局限与下一步

**局限**：① 光学为 grz+占位（真 grizY 需 Data Lab token）；② 训练量 1,187（可扩 3,181+数万负样本）；③ 负样本标签噪声（WISE<5″ 远处分量可能含真实宿主）；④ 单点检查点无 LR 调度；⑤ 测试仅 100 源。

**下一步（按优先级）**：扩训练集全量 → 真 grizY 光学 → LR 调度+更多 epoch → 形态分层评估 → 用 DRAGNs CATWISE ID 反查宿主真实 WISE。
