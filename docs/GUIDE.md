# GalaxyClassifier 部署与技术参考 (GUIDE.md)

> 最后更新：2026-08-02 (文档合并 — 原 AGENT.md/PATH_ISSUES.md/DEPLOY_ISSUES.md 合并而来)
> 项目：射电源光学宿主识别 — 两阶段多模态 CNN 模型
> 论文：楼康志-国科大-2023-硕士《应用深度学习识别河外射电展源的宿主星系》
> 仓库：https://github.com/duncanlou/GalaxyClassifier (handoff + master 分支)
> 相关文档：验证结果见 [VALIDATION.md](VALIDATION.md)，术语/数据来源见 [REFERENCE.md](REFERENCE.md)

---

## 1. 环境信息

| 项目 | 版本/说明 |
|------|----------|
| OS | Windows 11 Home China 10.0.26200 |
| Python | 3.12.6 (C:\dev\python\3.12.6) |
| GPU | NVIDIA GeForce RTX 4060 Laptop (4GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| 项目路径 | `C:\Users\HaheDaftYuDu\SKA项目\GalaxyClassifier\` |

---

## 2. 目录结构

```
C:\Users\HaheDaftYuDu\SKA项目\
├── GalaxyClassifier/
│   ├── main.py                  # 模型A 训练/推理入口
│   ├── mynetwork.py             # 模型A 定义 (CelestialClassficationNet)
│   ├── FitsImageFolder.py       # 模型A 数据加载
│   ├── optical_dataset.py       # 模型A 数据集封装
│   ├── download_data.py         # PS切图下载工具(原始)
│   ├── rebuild_catalogs.py      # ⭐ 训练星表重建脚本(新增)
│   ├── docs/                    # ⭐ 全部文档 (2026-07-31 重构后集中)
│   │   ├── README.md 位置说明    # README.md 保留在项目根目录作入口
│   │   ├── GUIDE.md             # 本文档 (部署/架构/运行)
│   │   ├── VALIDATION.md        # 验证权威 (六次验证+论文复现+边界)
│   │   ├── REFERENCE.md         # 术语/数据来源/历史问题
│   │   ├── STRUCTURE_CHANGELOG.md # 目录重构映射表(2026-07-31)
│   │   └── training_notes/      # evaluation_report.md (脚本生成)
│   ├── crossmatch/
│   │   ├── cross_matching.py    # 模型B 训练+端到端推理入口
│   │   ├── model_crossmatch.py  # 模型B 定义
│   │   ├── ps_dataset.py        # 模型B FITS数据加载(需CSV)
│   │   ├── xmatch_dataset.py    # 模型B 数据集封装
│   │   ├── models/              # ✅ 7个权重文件已就位(600MB)
│   │   └── training_notes/      # 训练记录(1343条Norris测试输出+验证结果csv)
│   ├── data/
│   │   ├── catalogs/            # ⭐ 全部星表
│   │   │   ├── preprocessed_cat/ # ✅ 4个CSV已重建(星表)
│   │   │   ├── SDSS_clean_cat/   # ✅ SDSSxWISE_cat.tbl (1.3GB)
│   │   │   ├── norris06_table0.csv  # Norris2006射电组件(639条)
│   │   │   ├── norris06_table1.csv  # Norris2006射电源+ID(594条)
│   │   │   ├── vlass_cdfs_components.csv # CDFS天区VLASS(305条)
│   │   │   ├── DR1_FIRST_*.csv / DR1_ATLAS_*.csv  # RGZ DR1标签
│   │   │   └── thesis_method_test_sample.csv  # 论文方法测试样本(90行)
│   │   ├── logs/                # ⭐ 下载/查询日志(6个txt)
│   │   └── cache/               # ⭐ catwise_cache.json (WISE查询缓存)
│   ├── tools/                   # ⭐ 8个新增脚本(下载/查询/评估)
│   └── source_cutouts/          # ✅ 切图按天区/测试集重组 (2026-07-31)
│       ├── cdfs/                # 会话4: CDFS独立验证 (Dec~-28°)
│       │   ├── ps/              # ✅ 71×5 PanSTARRS FITS (VLASS命名)
│       │   └── radio/           # ✅ 51个VLASS QL FITS (~7MB)
│       ├── north/               # 会话5/7: 北天测试
│       │   ├── sdss/            # SDSS 30源测试 (论文方法)
│       │   │   ├── ps/          # ✅ GALAXY/QSO/STAR各10 (原ps_sdss_test)
│       │   │   └── radio/       # ✅ 25个VLASS (原radio_sdss)
│       │   └── rgz/             # RGZ FIRST北天源 (会话7a)
│       │       ├── ps/          # ✅ 12个rgz_* + _dr1_test (原ps_north)
│       │       └── radio/       # ✅ 9个VLASS (原radio_north)
│       └── south_test/          # 会话8: 南天分天区测试
│           └── regions/         # Mid/North/South 各含GALAXY/QSO/STAR
```

---

## 3. 项目现状总览

### 3.1 数据完整性

| 数据项 | 状态 | 来源 | 说明 |
|--------|:--:|------|------|
| Python代码 | ✅ | GitHub handoff/master | |
| 模型A权重 (43MB) | ✅ | GitHub master | `opt_classification_model_wts.pt` |
| 模型B权重 (6个,510MB) | ✅ | GitHub master | 含RGZ1_1/3_1/ROGUE/ROGUE1000/all_negative变体 |
| SDSSxWISE_cat.tbl (1.3GB) | ✅ | 云盘 | 含394万行, 仅北天(Dec>-19.7°) |
| 模型B训练星表(4个CSV) | ⚠️ | 重建 | training_notes→1343条, WISE asinh差异可忽略(会话5) |
| CDFS射电切图 | ✅ | CADC下载 | 51个VLASS QL FITS (会话4) |
| CDFS光学切图 | ✅ | panstamps下载 | 71×5 PanSTARRS FITS (会话4) |
| 北天测试切图 | ✅ | panstamps下载 | 30源, 按论文方法 (会话5) |
| RGZ DR1标签数据 | ✅ | Zenodo | 9.9万条FIRST北天, 含宿主+WISE (会话6) |
| FITS光学切图 (94GB原始) | ❌ | 丢失 | Expansion硬盘 |
| 原始RGZ训练星表 | ❌ | 丢失 | Linux服务器 |

### 3.2 GitHub仓库发现

- **handoff分支**: 纯代码 + training_notes, 无权重无数据
- **master分支**: 全代码 + 7个模型权重(.pt), 无数据文件
- **云盘仅3文件**: SDSSxWISE_cat.tbl + robocopy日志 + download_ps_images.py

---

## 4. 模型架构

```
模型 A: CelestialClassficationNet (光学三分类, 准确率96%)
├── ⚠️ 仅接受光学输入: PanSTARRS 5波段图(5,240,240) + WISE星等(2维)
├──   不接受射电输入 — 射电数据不参与模型A的任何验证/推理
├── ResNet18 (conv1: 5→64通道) + WISE星等(2维)
├── → fc(22→8) → fc(8→3)
└── 输出: [Galaxy, QSO, STAR]

模型 B: RadioOpticalCrossmatchModel (宿主认证)
├── 射电切图(1ch) 仅输入到模型B, 与模型A无关
├── ResNet18(射电1ch) ⊕ ResNet18(光学5ch) ⊕ 模型A输出 ⊕ 位置/质心/测地
├── → fc(31→8) → fc(8→2)
└── 输出: [非宿主, 宿主]
```

**验证口径**：模型 A 的验证组合只有两个 — **光学+北天**（SDSS 训练分布内，正常）与
**光学+南天**（CDFS 等南天天区，退化）。射电数据只属于模型 B。详见 [VALIDATION.md](VALIDATION.md)。

---

## 5. 运行方式

### 5.1 模型A 推理
```bash
cd C:\Users\HaheDaftYuDu\SKA项目\GalaxyClassifier
python main.py
# 需要: source_cutouts/cdfs/ps/ 下有FITS切图
```

### 5.2 模型B 推理
```bash
cd C:\Users\HaheDaftYuDu\SKA项目\GalaxyClassifier\crossmatch
python cross_matching.py
# 需要: source_cutouts/cdfs/radio/ + source_cutouts/cdfs/ps/ + data/catalogs/preprocessed_cat/*.csv
```

### 5.3 重建星表
```bash
python rebuild_catalogs.py
```

### 5.4 管线完整性检查
```bash
python tools/validate_pipeline.py
# 输出 READY FOR INFERENCE 表示权重/切图/星表齐全
```

---

## 6. 独立验证路线 (路线A: CDFS天区) — ✅ 已完成

> 会话4完成7个阶段全部跑通，会话5完成根因定位。详细验证结果见 [VALIDATION.md](VALIDATION.md)。

```
验证口径: 模型A仅接受光学输入(PS 5波段+WISE), 验证组合只有两个:
  ┌ 光学+北天: SDSS训练分布内 → 模型A正常 (论文方法自洽)
  └ 光学+南天: CDFS等南天天区 → 模型A退化 (天区不匹配)
  射电数据(VLASS)只属于模型B, 不参与对模型A的验证

关键发现 (会话5):
  🔴 根因是天区不匹配(北天SDSS训练→南天CDFS推理), 非WISE占位符
     - WISE asinh星等压缩动态范围, 占位符(619mJy)与真实值(1.95mJy)差异<0.008 mag
     - Model A(仅光学输入)在南天CDFS图像上ResNet特征退化 → 92%判STAR(北天STAR仅23%)
     - Model B跟随退化: host_prob均值从0.745→0.015
  🟢 论文方法在北天数据上自洽有效: Model A恢复区分力, Model B学到Galaxy→host规则
```

### 路线A产出

| 产出 | 路径 | 说明 |
|------|------|------|
| 射电FITS | `source_cutouts/cdfs/radio/` | 51个VLASS QL FITS (~140KB each) |
| 光学FITS | `source_cutouts/cdfs/ps/` | 71×5 PanSTARRS FITS (~1.8GB total) |
| CatWISE缓存 | `data/cache/catwise_cache.json` | 150条查询缓存, 可续传 |
| Norris交叉匹配 | `data/catalogs/norris_crossmatch_results.csv` | 45/71匹配, 中值距离0.74 arcsec |
| 推理结果 | `crossmatch/training_notes/cdfs_validation_results.csv` | 194条端到端结果 |
| 评估报告 | `docs/training_notes/evaluation_report.md` | 完整评估(脚本生成) |
| PPT数据 | `crossmatch/training_notes/evaluation_ppt_data.txt` | 会议用数据 |

---

## 7. 项目进展时间线

### 会话1-2 (2026-07-25) — 环境搭建
- Python 3.12 + PyTorch 2.6 CUDA, SDSSxWISE_cat.tbl (1.3GB), 7个权重 (600MB)

### 会话3 (2026-07-25) — 数据重建
- 重建4个CSV星表, Norris2006 + VLASS CDFS数据, 规划路线A

### 会话4 (2026-07-26) — CDFS独立验证
- 7阶段全通, 准确率93.3%但宿主召回率0%, 初步诊断WISE占位符

### 会话5 (2026-07-27) — 根因定位
- 确认: 天区不匹配 (北天训练→南天CDFS), WISE非根因 (asinh压缩差异<0.008mag)
- 北天30 SDSS源 → Model A恢复区分力

### 会话6 (2026-07-28) — RGZ标签+文档
- RGZ DR1下载 (9.9万北天+583南天), 项目文档更新

### 会话7 (2026-07-29) — 论文方法验证 + A→B管线
- VLASS北天下载成功 (CADC Artifact表→nrao: URI, get_images())
- PanSTARRS下载 (panstamps.downloader)
- Model A→B 完整管线跑通 (FitsImageFolder预处理)
- **按论文方法验证 Model A (SDSS光谱源30个) + Model B (Norris射电源1343条)**

### 会话8 (2026-07-30) — 南天分天区测试
- south_sky_test.py: 40条南天源 (Mid/North/South Dec分带)
- 结论: GALAXY 0% 是DR2所致, 与天区无关 (详见 VALIDATION.md)

---

## 8. 源代码改动记录 (Source Code Changes)

> 记录所有对原有代码的修改，用于PPT和交接

### 8.1 `crossmatch/model_crossmatch.py` — 恢复原始架构
**问题**: 代码在迁移中`cnn_opt.fc`从10维改成20维，`fc1`从35/36维改成49维，与已保存权重(600MB)不匹配。
**改动**:
- `cnn_opt.fc`: `nn.Linear(512, 20)` → `nn.Linear(512, 10)` — 恢复原始10维光学特征输出
- `fc1`: `nn.Linear(49, 8)` → 动态`nn.Linear(33+pos_dims, 8)` — 匹配权重中35或36维输入
- `forward()`: 位置特征从6维(2+2+2)缩减为`pos_dims`维 — 取每对的首分量
- 新增`detect_pos_dims()`静态方法 — 从权重文件自动检测架构
- **原因**: 已保存的7个权重文件都是用原始架构训练的，改模型定义会导致加载失败

### 8.2 `crossmatch/ps_dataset.py` — 2处兼容修复
**改动1**: 移除`assert len(ps_dirs) == len(radio_files)` — 允许PS和Radio数量不一致
- **原因**: 51个分量有Radio FITS，71个有PS FITS，原断言会导致`AssertionError`
- 改为`warnings.warn()`警告

**改动2**: `np.float()` → `float()` (两处，第157/160行)
- **原因**: NumPy 1.20+废弃了`np.float`别名，Python 3.12报`AttributeError`

### 8.3 新增文件 (全部新建)

| 文件 | 用途 | 数据来源 |
|------|------|----------|
| `tools/download_vlass_radio.py` | 下载VLASS射电FITS切图 | CADC (nrao:VLASS), 51/71个分量下载成功 |
| `tools/download_ps_cutouts.py` | 下载PanSTARRS光学5波段切图 | STScI PanSTARRS (panstamps库), 71/71成功 |
| `tools/query_catwise.py` | 查询CatWISE真实WISE星等 | VizieR II/365/catwise, 87/1343个PS源获得真实值 |
| `tools/crossmatch_norris.py` | VLASS-Norris2006交叉匹配 | Norris2006 CDFS星表, 45/71匹配到Norris标签 |
| `tools/validate_pipeline.py` | 管线完整性验证 | 检查所有权重/数据/环境 |
| `tools/readline.py` | Windows下panstamps的dummy readline | — |
| `crossmatch/run_cdfs_inference.py` | 端到端推理脚本 | 从source_cutouts加载FITS + preprocessed_cat加载CSV |
| `crossmatch/evaluate_cdfs_results.py` | 评估报告生成(Markdown+PPT) | 推理结果 + Norris标签 |
| `data/catalogs/preprocessed_cat/filtered_PS_*_Norris06_samples.csv` | 过滤后仅含有效FITS的CSV | 从原始CSV过滤, 194条测试样本 |

### 8.4 关键Bug修复 (7个)
1. **VLASS名称前导零**: `J33323.71`→`zfill(6)`→`J033323.71` — RA整数部分补齐HHMMSS
2. **CADC URL命名空间**: `ad:VLASS`(需过期RUNID)→`nrao:VLASS`(免认证)
3. **GBK编码**: emoji/中文无法打印→`.encode('ascii','replace')`
4. **panstamps Windows兼容**: 缺少`readline`→创建dummy `tools/readline.py`
5. **panstamps参数**: `arcsec`→`arcsecSize`
6. **CatWISE列名**: `W1mpro`/`W2mpro`→`mW1`/`mW2` (VizieR II/365)
7. **NumPy兼容**: `np.float()`→`float()` (NumPy 1.20+废弃)

---

## 9. 路径问题 (原 PATH_ISSUES.md, 已修复)

> **更新 (2026-07-31)**：目录重构后全部 P0/P1 路径已更新为新结构
> （`source_cutouts/cdfs/`、`source_cutouts/north/`、`source_cutouts/south_test/`、`data/catalogs/`、`data/logs/`、`data/cache/`），
> 下述"修复方案"即当前生效路径。重构映射详见 [STRUCTURE_CHANGELOG.md](STRUCTURE_CHANGELOG.md)。

### 9.1 问题概览

| 环境 | 路径前缀 | 状态 |
|------|----------|------|
| Linux 服务器 | `/mnt/DataDisk/Duncan/` | 已废弃 |
| macOS + 移动硬盘 | `/Volumes/Expansion/` | 硬盘未挂载 |
| 本地相对路径 | `data/catalogs/preprocessed_cat/` | ✅ 可用（需补云盘文件） |

### 9.2 逐文件修复记录

| 优先级 | 文件 | 修复动作 |
|--------|------|----------|
| 🔴 P0 | `main.py:23` | `src_root_path` → `source_cutouts/cdfs/ps` |
| 🔴 P0 | `crossmatch/cross_matching.py:25-26` | `VLASS_IMAGE_ROOT`/`PS_IMAGE_ROOT` → `cdfs/radio`、`cdfs/ps` |
| 🟡 P1 | `optical/predicting_ps_data.py:29` | `opt_root_dir` → `source_cutouts/cdfs/ps` |
| 🟡 P1 | `utils.py:108` / `tools/utils.py:104` | 废弃 `images4` 路径, 通过参数传入 |
| 🟢 P2 | 其余工具脚本 | 用到时再改 |

**已修复的旧路径**（2026-07-31 重构前的问题根源）：
- `main.py` / `cross_matching.py` 曾指向 `/Volumes/Expansion/`、`/mnt/DataDisk/...` 等不可用存储
- `data/preprocessed_cat_new/` vs `data/preprocessed_cat/` 目录名差异 — 代码实际引用 `preprocessed_cat/`，✅ 无碍
- `mynetwork.py` 已用新 API `weights=models.ResNet18_Weights.DEFAULT`（非旧版 `pretrained=True`）✅

---

## 10. 已知问题记录

> 原 AGENT.md §11 + DEPLOY_ISSUES.md 7 问题合并。✅/❌ 为已解决状态。

| # | 严重度 | 问题 | 状态 |
|---|:--:|------|------|
| 1 | 🔴 | SDSSxWISE仅北天, WISE星等无法匹配 | ✅ 已解决 — WISE asinh星等影响可忽略(<0.008mag, 会话5) |
| 2 | 🔴 | RGZ训练集原始星表丢失 | 🔴 未解决 — 用Norris子集替代(NOT host labels) |
| 3 | 🔴 | FITS切图(94GB)全部丢失 | ✅ 部分解决 — 已重下CDFS区域71分量切图 |
| 4 | 🟡 | 模型A的SDSS标签待确认 | 🟡 — SDSSxWISE_cat.tbl 的 class_01 列需验证 |
| 5 | 🔵 | 云盘内容不完整(仅3文件) | 🔵 — 权重在GitHub master, 星表已重建 |
| 6 | 🔵 | 1343条训练笔记是模型B测试输出 | 🔵 — 非模型A数据, 性质已确认 |
| 7 | 🔵 | GitHub两分支结构: master有权重, handoff仅有代码 | 🔵 — 结构确认 |
| 8 | 🟡 | 模型A/B南天泛化能力待改进 | 🟡 — 天区迁移需要适配训练 |

---

## 11. 注意事项

1. **num_workers=8**: Windows下需改为 `num_workers=0`
2. **路径格式**: 使用 `C:/Users/...` 正斜杠
3. **磁盘空间**: C盘约209GB可用
4. **Python命令**: 用 `python` 而非 `python3`
5. **终端编码**: Windows GBK不支持emoji, 脚本避免使用Unicode特殊字符
6. **模型天区限制**: 当前模型仅在**北天**(Dec > -20°)有效, 南天需适配
7. **临时目录**: `tools/north_sky_validate.py`/`south_sky_test.py` 运行时在 `source_cutouts/.tmp_north`/`.tmp_south` 自动重建, 可随时删除
