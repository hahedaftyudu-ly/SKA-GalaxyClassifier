# GalaxyClassifier 目录重构记录 (2026-07-31)

> 本次重构**只移动/归类文件，不改动任何数据内容、验证结果与结论**。
> 目的：① 文档集中；② 下载数据按天区/类型归类；③ 清理重复数据与临时产物（省 ~2GB）。
> 所有代码中的硬编码路径已同步更新，`validate_pipeline.py` 验证通过（READY FOR INFERENCE）。
>
> **二次精简 (2026-08-02)**：文档又合并为 README(总览) + GUIDE(部署) + VALIDATION(验证) + REFERENCE(参考) 四个文档，
> 下表中除 README、evaluation_report.md、paper_extract.txt 外的文档已被合并删除，内容详见新文档。

---

## 一、文档移动（内容零改动）

| 原位置 | 新位置 | 2026-08-02 后的去向 |
|--------|--------|------|
| `README.md` | 保持不动（项目入口） | 重写为总览索引 |
| `AGENT.md` | `docs/AGENT.md` | 并入 `docs/GUIDE.md` |
| `ACRONYMS.md` | `docs/ACRONYMS.md` | 并入 `docs/REFERENCE.md` |
| `PATH_ISSUES.md` | `docs/PATH_ISSUES.md`（头部加"已修复"标注） | 并入 `docs/GUIDE.md` §9 |
| `DEPLOY_ISSUES.md` | `docs/DEPLOY_ISSUES.md` | 并入 `docs/GUIDE.md` §10 + `docs/REFERENCE.md` §7 |
| `VALIDATION_SUMMARY.md` | `docs/VALIDATION_SUMMARY.md` | 并入 `docs/VALIDATION.md` |
| `crossmatch/training_notes/evaluation_report.md` | `docs/training_notes/evaluation_report.md` | 保留（脚本生成物） |
| `crossmatch/training_notes/failure_log.md` | `docs/training_notes/failure_log.md` | 并入 `docs/VALIDATION.md` |
| `crossmatch/training_notes/validation_comparison_report.md` | `docs/training_notes/validation_comparison_report.md` | 并入 `docs/VALIDATION.md` |
| `data/paper_extract.txt` | `docs/paper_extract.txt` | 保留 |

---

## 二、切图数据重组（source_cutouts/，按天区）

| 新路径 | 内容 | 原路径 |
|--------|------|--------|
| `source_cutouts/cdfs/ps/` | CDFS 71 分量 × 5 波段 PanSTARRS（VLASS 命名） | `source_cutouts/ps/` |
| `source_cutouts/cdfs/radio/` | CDFS 51 个 VLASS QL FITS | `source_cutouts/radio/` |
| `source_cutouts/north/sdss/ps/` | SDSS 30 北天测试源（GALAXY/QSO/STAR 各 10） | `source_cutouts/ps_sdss_test/` |
| `source_cutouts/north/sdss/radio/` | SDSS 源 25 个 VLASS FITS | `source_cutouts/radio_sdss/` |
| `source_cutouts/north/rgz/ps/` | RGZ FIRST 北天 12 源 + `_dr1_test/`（DR1 对比） | `source_cutouts/ps_north/` |
| `source_cutouts/north/rgz/radio/` | RGZ 源 9 个 VLASS FITS | `source_cutouts/radio_north/` |
| `source_cutouts/south_test/regions/` | 南天分天区测试 Mid/North/South | `source_cutouts/ps_south_test/{Mid,North,South}/` |

---

## 三、星表/日志/缓存分类（data/）

| 新路径 | 内容 | 原位置 |
|--------|------|--------|
| `data/catalogs/` | 全部星表 | `data/` 根下 15 个条目 |
| ├─ `preprocessed_cat/` | 模型 B 训练/测试 4 个 CSV | `data/preprocessed_cat/` |
| ├─ `SDSS_clean_cat/` | SDSSxWISE_cat.tbl (1.3GB) | `data/SDSS_clean_cat/` |
| ├─ `SDSS_clean_cat_Duncan.csv` | SDSS 三分类 CSV (128MB) | `data/SDSS_clean_cat_Duncan.csv` |
| ├─ `norris06_table0/1.csv`、`vlass_cdfs_components.csv` | Norris2006、CDFS VLASS 星表 | `data/` 根 |
| ├─ `DR1_FIRST_*.csv`、`DR1_ATLAS_*.csv` | RGZ DR1 标签（北天 9.9 万 + 南天 1394） | `data/` 根 |
| ├─ `RGZ_DR1_tables.tar.gz` | Zenodo 原始压缩包 | `data/` 根 |
| └─ `thesis_method_test_sample.csv`、`rgz_first_matched.csv`、`CIRADA_VLASS_hostid.csv`、`norris_crossmatch_results.csv` | 测试样本/匹配结果 | `data/` 根 |
| `data/logs/` | 5 个下载/查询/推理日志 | `data/` 根下 `*.txt` |
| `data/cache/` | `catwise_cache.json`（WISE 查询缓存） | `data/` 根 |
| — | `data/crossmatch/`（空目录） | 已删除 |

---

## 四、删除内容（均验证过有副本或为垃圾）

| 删除项 | 数量 | 验证方式 |
|--------|:--:|------|
| 根目录散落的坐标切图目录 | 102 | 71 个与 `cdfs/ps/` 坐标 71/71 匹配；30 个与 `thesis_method_test_sample.csv` 30 源坐标匹配；1 个（217.839p23.381）与 `north/rgz/ps/_dr1_test/` 同名完整 |
| `source_cutouts/ps_south_test/_tmp/` | 40 个空目录 | 文件数 0 |
| `source_cutouts/ps_north/_tmp/` | 11 个空目录 | 文件数 0 |
| `source_cutouts/_north_tmp/`、`_south_tmp/` | 运行时产物 | 脚本每次运行自动重建；`_south_tmp` 39 源与 `south_test/regions/` 交集 39/39 |
| `_test_conn/` | 连接测试残留 | 不完整副本 |
| `__pycache__/` | 3 处 | 编译缓存 |

> ⚠️ 注意：`tools/north_sky_validate.py` / `tools/south_sky_test.py` 的临时目录已改为
> `source_cutouts/.tmp_north` / `source_cutouts/.tmp_south`（运行时会自动重建，可随时删除）。

---

## 五、代码路径更新范围

| 文件 | 改动 |
|------|------|
| `main.py` | `src_root_path` → `source_cutouts/cdfs/ps` |
| `crossmatch/cross_matching.py` | `VLASS/PS_IMAGE_ROOT` → `cdfs/radio`、`cdfs/ps` |
| `crossmatch/run_cdfs_inference.py` | 切图路径 → `cdfs/`；星表 → `data/catalogs/`；日志 → `data/logs/` |
| `crossmatch/evaluate_cdfs_results.py` | 报告路径 → `docs/training_notes/` |
| `crossmatch/data_processing.py`、`data_processings.py` | 星表 → `data/catalogs/preprocessed_cat/` |
| `FitsImageFolder.py` | WISE CSV → `data/catalogs/SDSS_clean_cat_Duncan.csv` |
| `optical/predicting_ps_data.py` | → `cdfs/ps`、`data/catalogs/` |
| `rebuild_catalogs.py` | → `data/catalogs/` |
| `check_trainingsets_class.py` | 修正为真实 tbl 路径 `data/catalogs/SDSS_clean_cat/SDSSxWISE_cat.tbl` |
| `tools/` 共 15 个脚本 | `download_vlass_radio` / `download_ps_cutouts` → `cdfs/`；`north_sky_validate` / `sdss_ab_validate` / `thesis_validate` / `test_thesis_method` / `show_dr_diff` / `south_sky_test` → `north/`、`south_test/`；`crossmatch_rgz_labels` / `crossmatch_norris` / `query_catwise` / `validate_pipeline` → `data/catalogs|logs|cache` |

全部 27 个文件 `py_compile` 语法检查通过；`tools/validate_pipeline.py` 端到端验证 READY。

---

## 六、做 PPT 时验证数据的当前位置（未移动，与重构前一致）

| 验证 | 结果文件 |
|------|----------|
| 论文原始 Model B 测试（1343 条） | `crossmatch/training_notes/RGZ_all_negative_Norris_testing_notes*.csv` |
| CDFS 独立验证（会话4，194 条） | `crossmatch/training_notes/cdfs_validation_results.csv` + `docs/training_notes/evaluation_report.md` |
| 北天 SDSS Model A 复现（会话5，30 源） | `crossmatch/training_notes/thesis_model_a_results.csv` |
| RGZ 北天 A→B（会话7a） | `crossmatch/training_notes/north_sky_validation_results.csv` |
| SDSS A→B 完整管线（会话7c） | `crossmatch/training_notes/sdss_ab_validation.csv` |
| 南天分天区测试（会话8） | `crossmatch/training_notes/south_sky_model_a_results.csv` |
| 六次验证总结 + 泛化边界 | `docs/VALIDATION.md` |
| 失败原因详录 + DR1/DR2 差异 | `docs/VALIDATION.md` |
| PPT 数据文件 | `crossmatch/training_notes/evaluation_ppt_data.txt` |

---

## 七、当前完整结构速览

```
GalaxyClassifier/
├── README.md                     # 项目入口 (总览+交接索引)
├── docs/                         # 全部文档
│   ├── GUIDE.md                  # 部署技术参考 (环境/架构/运行/路径/问题)
│   ├── VALIDATION.md             # 验证权威 (六次验证/边界/教训/溯源/改进方向)
│   ├── REFERENCE.md              # 参考资料 (术语/数据来源/历史问题)
│   ├── STRUCTURE_CHANGELOG.md    # 本文档
│   ├── ISSUES_LOG.md             # 脚本生成 (rebuild_catalogs.py 输出)
│   └── training_notes/           # evaluation_report.md (脚本生成)
├── crossmatch/                   # 模型B + training_notes(验证结果数据)
├── data/
│   ├── catalogs/  ├── logs/  └── cache/
├── tools/                        # 下载/查询/评估脚本
├── source_cutouts/
│   ├── cdfs/      ├── north/(sdss + rgz)  └── south_test/regions/
├── main.py / mynetwork.py / FitsImageFolder.py / optical_dataset.py ...   # 模型A
└── notebooks/ / optical/ / radio/ / test_images/                          # 原始代码
```
