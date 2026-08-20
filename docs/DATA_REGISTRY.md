# GalaxyClassifier 数据登记表 (DATA_REGISTRY.md)

> 创建：2026-08-03 | 用途：**任何下载/抽样前先查本表**——命中同一批次（来源+参数+天区）直接复用现有文件，避免重复获取。
> 配套机制：所有数据脚本应内置幂等检查（输出已存在则跳过，`--force` 覆盖）。
> 判定"是否为同一批"的字段：来源 + 日期 + 参数 + 文件数/行数。

---

## 一、切图数据 (source_cutouts/)

| 批次 | 内容 | 来源 | 日期 | 参数 | 位置 | 规模 | 用途/引用验证 |
|------|------|------|:--:|------|------|------|------|
| CDFS 光学切图 | 71 分量 × 5 波段 PanSTARRS (VLASS命名) | STScI panstamps | 2026-07-26 | arcsec=180, DR2 | `source_cutouts/cdfs/ps/` | 71 目录 × 5 = 355 FITS, 709MB | 会话4 验证[1] (光学+南天), run_cdfs_inference.py |
| CDFS 射电切图 | 51 个 VLASS QuickLook | CADC `nrao:VLASS` | 2026-07-26 | — | `source_cutouts/cdfs/radio/` | 51 FITS, 7MB | 会话4 验证[1] (仅模型B) |
| 北天 SDSS 光学切图 | 30 源 (GALAXY/QSO/STAR各10) | STScI panstamps | 2026-07-26 | arcsec=60, DR2 | `source_cutouts/north/sdss/ps/` | 30 目录 × 5 = 150 FITS, 36MB | 会话5 验证[2], 7b 验证[4], 7c 验证[5] |
| 北天 SDSS 射电切图 | 25 个 VLASS | CADC get_images | 2026-07-29 | 145×145px | `source_cutouts/north/sdss/radio/` | 25 FITS, 5MB | 会话7c 验证[5] (仅模型B) |
| RGZ 北天光学切图 | 12 源 + _dr1_test | STScI panstamps | 2026-07-28~29 | arcsec=60 | `source_cutouts/north/rgz/ps/` | 12×5+5 = 60 FITS, 14MB | 会话7a 验证[3] |
| RGZ 北天射电切图 | 9 个 VLASS | CADC get_images | 2026-07-28~29 | — | `source_cutouts/north/rgz/radio/` | 9 FITS, 2MB | 会话7a 验证[3] (仅模型B) |
| 南天分天区切图 | Mid/North/South 各含 GALAXY/QSO/STAR | STScI panstamps | 2026-07-30 | arcsec=60, DR2 | `source_cutouts/south_test/regions/` | 39 源 × 5 = 195 FITS, 47MB | 会话8 south_sky_test |
| EMU-PS1 射电切图 | 100 源 (50 DRAGN + 50 非宿主) | AS101 taylor.0 拼图自裁 | 2026-08 | 216px @ 2″/px = 432″ | `source_cutouts/emu_ps1/radio/` | 100 FITS, ~20MB | EMU 模型B测试 |
| EMU-PS1 光学切图 | 100 源 × 5 波段 (⚠️ legacy des-dr1 仅 grz, 通道3/5=z复制占位) | Legacy Surveys cutout | 2026-08 | 240px @ 0.25″/px = 60″ | `source_cutouts/emu_ps1/optical/` | 500 FITS | EMU 模型B测试 (冒烟口径, 正式需 Data Lab grizY) |

> 2026-07-31 已删除根目录散落的 102 个重复切图目录 (与上表 cdfs/ps、north/sdss/ps 重复, 详见 STRUCTURE_CHANGELOG)。

---

## 二、星表 (data/catalogs/)

| 批次 | 内容 | 来源 | 日期 | 规模 | 用途 |
|------|------|------|:--:|------|------|
| SDSSxWISE 交叉表 | SDSS DR16 × CatWISE 交叉, 含 w1flux/w2flux | 阿里云盘 (论文云盘) | 2026-07-25 | 394万行, 1GB | 模型A标签来源; WISE星等查询表; rebuild_catalogs 输入 |
| SDSS 三分类 CSV | source_id + w1flux + w2flux (精简版) | 从 SDSSxWISE 生成 | 2026-07-26 | 394万行, 127MB | FitsImageFolder 的 WISE 查询 |
| Norris2006 射电组件 | CDFS 射电分量 | VizieR J/ApJS/164/297 | 2026-07-25 | 639 行 | 会话3 交叉匹配 |
| Norris2006 射电源+ID | 源+光学ID+分类 | VizieR J/ApJS/164/297 | 2026-07-25 | 594 行 | 论文测试集标签 (形态类型, 非宿主标签!) |
| VLASS CDFS 分量 | CDFS 天区 VLASS 305 分量 | CADC VLASS catalog | 2026-07-25 | 305 行 | 会话4 下载坐标来源 |
| RGZ DR1 FIRST 宿主属性 | 北天射电源宿主 + WISE 星等 | Zenodo 14195049 | 2024-09-11 | 99,602 行 | 会话7a 验证[3] 标签 |
| RGZ DR1 FIRST 射电形态 | 北天射电形态分类 | Zenodo 14195049 | 2024-11-04 | 99,602 行 | 会话7a 验证[3] 标签 |
| RGZ DR1 ATLAS 宿主属性 | 南天宿主标签 (583 源) | Zenodo 14195049 | 2024-11-06 | 583 行 | ⭐ 南天唯一宿主标签, 尚未使用 |
| RGZ DR1 ATLAS 射电形态 | 南天射电形态 | Zenodo 14195049 | 2024-07-31 | 583 行 | ⭐ 尚未使用 |
| RGZ DR1 原始压缩包 | Zenodo 打包文件 | Zenodo 14195049 | 2026-07-28 | 7MB | 原始备份 |
| RGZ-FIRST 匹配结果 | 北天源 × RGZ 交叉 | 自建 (tools/crossmatch_rgz_labels.py) | 2026-07-28 | 68,925 行 | 会话7a/7c 标签来源 |
| Norris 交叉匹配结果 | 71 分量 × Norris 匹配 | 自建 (tools/crossmatch_norris.py) | 2026-07-26 | 71 行 | 会话4 45/71 匹配, 中值 0.74″ |
| 论文方法测试样本 | SDSS 30源候选 ×3类 (90行) | 本地 SDSSxWISE 表随机抽 (seed=42) | 2026-07-26 | 90 行 | 会话5/7b 测试[2][4] 输入清单 |
| 模型B星表 (4个CSV) | PS_p/N_n Norris06 + PS_p/N RGZ | 自建 (rebuild_catalogs.py) | 2026-07-26 | 15/255/56/1017 行 | 模型B训练/测试星表 (Norris 冒充) |
| CIRADA_VLASS_hostid.csv | CIRADA 宿主表 | CIRADA API | 2026-07-25 | 1 行 (错误日志) | ⚠️ 接口失败产物, 不可用 |
| EMU Selavy 交叉匹配表 | AS101 178,921 分量 (含 WISE 通量 mJy/95.4%, DES 60.2%, DESI) | CSIRO DAP EMU_PS_CATALOG.xml → 自转 | 2026-08 | 178,921 行 | `data/catalogs/emu/selavy_crossmatched.csv`; EMU 选源+WISE输入 |
| EMU DRAGNs 星表 | 3,557 双瓣源 (人眼认证, 宿主=CATWISE ID, 形态 Tags) | Cambridge Core supp Table7.fits | 2026-08 | 3,557 行 | `data/catalogs/emu/dragns_catalog.fits` + `cambridge_supp/`; EMU 正样本+标签 |
| EMU DRAGNs 形态标签 | source_id → Tags 映射 | 自建 (从 Table7 生成) | 2026-08 | 3,557 行 | `data/catalogs/emu/dragns_tags.csv`; 形态分层评估 |
| EMU 100 源清单 | 50 DRAGN 宿主 + 50 非宿主 (seed=42, WISE<5″) | 自建 (tools/emu_select_sources.py) | 2026-08 | 100 行 ×3 CSV | `data/catalogs/emu_ps1_samples/`; EMU 测试输入 |

---

## 三、模型权重 (crossmatch/models/)

| 权重 | 大小 | 日期 | 用途 |
|------|:--:|:--:|------|
| `opt_classification_model_wts.pt` | 43MB | 2026-07-25 | 模型 A (光学三分类) |
| `crossmatch_model_wts.pt` | 86MB | 2026-07-25 | 模型 B (基础版) |
| `RGZ_all_negative_crossmatch_model_wts.pt` | 86MB | 2026-07-25 | 模型 B 主用版 |
| `RGZ1_1 / RGZ3_1 / RGZ_ROGUE / RGZ_ROGUE1000` | 各86MB | 2026-07-25 | 模型 B 负样本策略变体 |

来源：GitHub master 分支 | 共 7 个, 556MB

---

## 四、验证结果 (crossmatch/training_notes/) — 勿删除, PPT 引用

| 文件 | 行数 | 日期 | 对应验证 |
|------|:--:|:--:|------|
| `RGZ_all_negative_Norris_testing_notes1.csv` | 1343 | 2026-06-18 | 论文原始 Model B 测试 (97.7%/87.3%) |
| `cdfs_validation_results.csv` | 194 | 2026-07-26 | 会话4 验证[1] CDFS |
| `thesis_model_a_results.csv` | 30 | 2026-07-29 | 会话7b 验证[4] Model A |
| `north_sky_validation_results.csv` | 7 | 2026-07-29 | 会话7a 验证[3] |
| `sdss_ab_validation.csv` | 25 | 2026-07-29 | 会话7c 验证[5] |
| `south_sky_model_a_results.csv` | 39 | 2026-07-30 | 会话8 南天分天区 |
| `emu_ps1_results_{raw,uniform,prior}.csv` | 100×3 | 2026-08 | EMU-PS1 零样本三组对照 (分离度 -0.018, 详见 EMU_TEST_PLAN.md §9) |
| `emu_ps1_results_xgb.csv` | 100 | 2026-08 | XGBoost 模型A 组 (WISE-only, 免图像; B 输出仍 -0.018, A 自身含 P/N 信号) |
| `emu_ps1_results_uniform_finetuned.csv` | 100 | 2026-08 | 微调后测试 (分离度 +0.404, 宿主召回 64%, 详见 EMU_TEST_PLAN §9.4) |
| `emu_finetuned_b.pt` | — | 2026-08 | 微调权重 (冻结光学分支, 射电分支训练; crossmatch/models/) |
| 其余 `*_training_notes*` / `*_test*` csv | — | 2026-06-18 | 论文原始训练记录 (GitHub handoff) |

---

## 五、缓存与日志

| 项 | 位置 | 说明 |
|------|------|------|
| CatWISE 查询缓存 | `data/cache/catwise_cache.json` | query_catwise.py 续传缓存 |
| 下载/查询日志 ×5 | `data/logs/*.txt` | 会话4 各阶段日志 |

---

## 六、登记惯例（新数据必须遵守）

1. **下载/抽样前**：查本表, 命中同批次 → 复用, 不重复获取
2. **脚本幂等**：输出已存在 → 跳过, 仅 `--force` 覆盖
3. **产出后登记**：新数据批次补一行到对应分区 (含来源/日期/参数/规模)
4. **验证结果**：跑完实验把结果文件登记到第四节, 并在 VALIDATION.md 补验证记录
5. **临时产物**：一律进 `source_cutouts/.tmp_*` 或 `data/logs/`, 不落根目录
