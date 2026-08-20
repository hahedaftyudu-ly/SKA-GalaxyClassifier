"""
重建训练/测试星表 CSV

背景:
  - 1343条数据来自模型B(宿主认证)的Norris06测试集输出
  - 每个记录: VLASS射电分量 -> PanSTARRS光学源, 标注是否宿主
  - PS源坐标在Dec ~ -28°(南天), SDSSxWISE只覆盖Dec > -19.7°(北天)
  - 因此无法通过坐标交叉匹配获取WISE星等

数据来源:
  1. crossmatch/training_notes/RGZ_all_negative_Norris_testing_notes1.csv (1343条)
  2. data/catalogs/SDSS_clean_cat/SDSSxWISE_cat.tbl (394万行, 但仅北天!)

输出:
  data/catalogs/preprocessed_cat/PS_p_Norris06_samples.csv  (正样本-宿主, 测试)
  data/catalogs/preprocessed_cat/PS_n_Norris06_samples.csv  (负样本-非宿主, 测试)
  data/catalogs/preprocessed_cat/PS_p_RGZ_samples.csv       (正样本-训练, Norris子集)
  data/catalogs/preprocessed_cat/PS_n_samples_RGZ_all.csv   (负样本-训练, Norris子集)

已知问题(记录到DEPLOY_ISSUES.md):
  1. CRITICAL: SDSSxWISE仅北天, PS源在南天, WISE星等无法获取
  2. CRITICAL: RGZ训练集原始星表丢失
  3. CRITICAL: 模型A的FITS切图全部丢失
  4. WARNING: 模型A的SDSS标签集丢失
"""

import os
import sys
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
NOTES_FILE = os.path.join(PROJECT_ROOT, "crossmatch", "training_notes",
                          "RGZ_all_negative_Norris_testing_notes1.csv")
SDSS_FILE = os.path.join(DATA_ROOT, "catalogs", "SDSS_clean_cat", "SDSSxWISE_cat.tbl")
OUT_DIR = os.path.join(DATA_ROOT, "catalogs", "preprocessed_cat")
ISSUES_LOG = os.path.join(PROJECT_ROOT, "docs", "ISSUES_LOG.md")

os.makedirs(OUT_DIR, exist_ok=True)

issues = []

def log_issue(severity, title, detail, workaround=""):
    issues.append({
        "severity": severity,
        "title": title,
        "detail": detail,
        "workaround": workaround
    })
    tag = {"CRITICAL": "[!!!]", "WARNING": "[!]", "INFO": "[i]"}.get(severity, "")
    print(f"{tag} [{severity}] {title}")

# ============================================================
# Step 1: 读取数据
# ============================================================
print("=" * 60)
print("Step 1: 读取数据源")
print("=" * 60)

df_notes = pd.read_csv(NOTES_FILE, skipinitialspace=True)
df_notes.columns = [c.strip() for c in df_notes.columns]
print(f"  training_notes: {len(df_notes)} 条")
print(f"  列: {list(df_notes.columns)}")

gt_counts = df_notes['Groud_Truth'].value_counts()
print(f"  正样本(宿主=1): {gt_counts.get(1,0)}, 负样本(非宿主=0): {gt_counts.get(0,0)}")

# 天区范围
print(f"  PS源RA: {df_notes['PS source ra'].min():.1f}-{df_notes['PS source ra'].max():.1f} deg")
print(f"  PS源Dec: {df_notes['PS source dec'].min():.1f}-{df_notes['PS source dec'].max():.1f} deg")

t_sdss = Table.read(SDSS_FILE, format='ascii.ipac')
df_sdss = t_sdss.to_pandas()
print(f"\n  SDSSxWISE: {len(df_sdss)} 条")
print(f"  SDSS Dec范围: {df_sdss['dec'].min():.1f} to {df_sdss['dec'].max():.1f} deg")

# ============================================================
# Step 2: 坐标匹配诊断
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 坐标交叉匹配诊断")
print("=" * 60)

notes_dec_min = df_notes['PS source dec'].min()
sdss_dec_min = df_sdss['dec'].min()
print(f"  PS源最小Dec: {notes_dec_min:.1f} deg")
print(f"  SDSS最小Dec:  {sdss_dec_min:.1f} deg")
print(f"  重叠: {'是' if notes_dec_min >= sdss_dec_min else '否 — 完全不重叠!'}")

# 用中位数做占位符
median_w1 = df_sdss['w1flux'].median()
median_w2 = df_sdss['w2flux'].median()
print(f"  WISE w1flux中位数: {median_w1:.2f}")
print(f"  WISE w2flux中位数: {median_w2:.2f}")

log_issue("CRITICAL",
    "SDSSxWISE与PS源天区不重叠 — 无法获取WISE星等",
    f"PS源在Dec -28°(南天VLASS天区), SDSSxWISE_cat.tbl只覆盖Dec -19.7°以上(北天)。"
    f"1343个PS源全部无法通过坐标匹配获取w1flux/w2flux。"
    f"原始WISE数据来自CatWISE全天星表，不在当前项目文件中。",
    f"暂时用SDSSxWISE中位数填充(w1flux={median_w1:.1f}, w2flux={median_w2:.1f})。"
    f"正式使用前需通过Vizier查询CatWISE获取真实WISE星等。")

# ============================================================
# Step 3: 构建CSV
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建输出CSV")
print("=" * 60)

df_out = pd.DataFrame({
    'VLASS_component_name': df_notes['VLASS_component_name'].values,
    'Pan-STARRS_RAJ2000': df_notes['PS source ra'].values,
    'Pan-STARRS_DEJ2000': df_notes['PS source dec'].values,
    'Pan-STARRS_objID': df_notes['PS source id'].values,
    'w1flux': median_w1,  # 占位符: SDSSxWISE中位数
    'w2flux': median_w2,  # 占位符: SDSSxWISE中位数
    'PS_class': df_notes['Groud_Truth'].map({1: 'P', 0: 'N'})
})

df_positive = df_out[df_out['PS_class'] == 'P'].copy()
df_negative = df_out[df_out['PS_class'] == 'N'].copy()
print(f"  正样本(宿主): {len(df_positive)}")
print(f"  负样本(非宿主): {len(df_negative)}")

# ============================================================
# Step 4: 拆分训练/测试集
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 拆分训练集/测试集")
print("=" * 60)

log_issue("CRITICAL",
    "RGZ训练星表完全丢失 — 不在GitHub、不在云盘",
    "原始训练星表(PS_p_RGZ_samples.csv, PS_n_samples_RGZ_all.csv)存储在已废弃的"
    "Linux服务器/mnt/DataDisk/Duncan/上，未备份到GitHub或云盘。"
    "目前仅保留Norris06测试集的1343条输出记录。",
    "用Norris06数据的80%子集冒充训练集，仅用于验证代码管线能跑通。"
    "真正训练必须联络原始作者(Kangzhi Lou)或重新构建RGZ训练集。")

np.random.seed(42)
# 80/20 split for train/test placeholder
p_idx = np.random.permutation(len(df_positive))
p_split = max(1, int(len(df_positive) * 0.8))
df_p_rgz = df_positive.iloc[p_idx[:p_split]]
df_p_norris = df_positive.iloc[p_idx[p_split:]]

n_idx = np.random.permutation(len(df_negative))
n_split = max(1, int(len(df_negative) * 0.8))
df_n_rgz = df_negative.iloc[n_idx[:n_split]]
df_n_norris = df_negative.iloc[n_idx[n_split:]]

print(f"  RGZ训练正样本: {len(df_p_rgz)}")
print(f"  RGZ训练负样本: {len(df_n_rgz)}")
print(f"  Norris测试正样本: {len(df_p_norris)}")
print(f"  Norris测试负样本: {len(df_n_norris)}")

# ============================================================
# Step 5: 写入文件
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 写入CSV")
print("=" * 60)

outputs = {
    'PS_p_RGZ_samples.csv': df_p_rgz,
    'PS_n_samples_RGZ_all.csv': df_n_rgz,
    'PS_p_Norris06_samples.csv': df_p_norris,
    'PS_n_Norris06_samples.csv': df_n_norris,
}

for fname, df in outputs.items():
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  {fname}: {len(df)}行")

# ============================================================
# Step 6: 记录模型A相关问题
# ============================================================
log_issue("CRITICAL",
    "模型A训练数据(FITS切图)完全丢失",
    "PanSTARRS 5波段FITS切图(94GB, 403,549文件)存储在Expansion移动硬盘上,"
    "该硬盘当前不可用。robocopy日志显示曾复制到D:盘, 但D:盘已被其他设备占用。"
    "没有FITS切图, 模型A无法训练也无法推理。",
    "方案1: 找到Expansion硬盘; 方案2: 用download_ps_images.py重新下载PS切图;"
    "方案3: 跳过模型A, 仅用模型B权重验证代码结构。")

log_issue("WARNING",
    "模型A的SDSS分类标签未确认",
    "AGENT.md列出的SDSS_clean_cat_Duncan.tbl(~1.3GB)在云盘中的对应文件是"
    "SDSSxWISE_cat.tbl, 但该文件含SDSS+WISE交叉匹配数据, 与预期的纯SDSS分类标签可能不同。"
    "文件已放入data/catalogs/SDSS_clean_cat/, 但代码中直接使用的路径是FITS切图目录而非此表。",
    "需确认SDSSxWISE_cat.tbl中的class_01列是否就是模型A训练所需的标签。")

log_issue("INFO",
    "云盘内容不完整",
    "云盘(https://www.alipan.com/s/2kRLFEAB7tV)仅有3个文件: SDSSxWISE_cat.tbl, "
    "VLASS_sources_for_sean_copy.log, download_ps_images.py。"
    "缺少: 模型权重(已从GitHub master获取), 训练星表CSV(本次重建), WISE交叉表。",
    "权重文件托管在GitHub master分支而非云盘; 星表CSV从training_notes重建; "
    "WISE交叉表需要从CatWISE重新查询。")

# ============================================================
# Step 7: 写问题文档
# ============================================================
print("\n" + "=" * 60)
print("问题清单")
print("=" * 60)

with open(ISSUES_LOG, 'w', encoding='utf-8') as f:
    f.write("# GalaxyClassifier 部署问题清单\n\n")
    f.write("> 项目: SKA射电源光学宿主识别 — 两阶段多模态CNN\n")
    f.write(f"> 生成日期: 2026-07-25\n")
    f.write("> 用于PPT汇报\n\n")
    f.write("---\n\n")

    for i, iss in enumerate(issues, 1):
        tag = {"CRITICAL": "[!!!]", "WARNING": "[!]", "INFO": "[i]"}.get(iss['severity'], "")
        summary = f"## {i}. {tag} [{iss['severity']}] {iss['title']}\n\n"
        detail = f"**详情:** {iss['detail']}\n\n"
        workaround = f"**临时方案:** {iss['workaround']}\n\n" if iss['workaround'] else ""
        block = summary + detail + workaround + "---\n\n"
        f.write(block)
        print(summary.strip())
        print(detail.strip())
        if workaround:
            print(workaround.strip())
        print()

    # 数据完整性总结
    f.write("## 数据完整性总结\n\n")
    f.write("| 数据项 | 状态 | 说明 |\n")
    f.write("|--------|:----:|------|\n")
    f.write("| Python代码 | ✅ | GitHub handoff/master |\n")
    f.write("| 模型A权重 (43MB) | ✅ | GitHub master |\n")
    f.write("| 模型B权重 (6个, 510MB) | ✅ | GitHub master |\n")
    f.write("| SDSSxWISE星表 (1.3GB) | ✅ | 云盘下载 |\n")
    f.write("| 模型B训练星表 | ⚠️ 已重建 | training_notes → 1343条, WISE星等为占位符 |\n")
    f.write("| 模型A训练标签 | ⚠️ 待确认 | SDSSxWISE_cat.tbl含class_01列, 待验证 |\n")
    f.write("| FITS光学切图 (94GB) | ❌ 丢失 | 需重新下载 |\n")
    f.write("| FITS射电切图 | ❌ 丢失 | VLASS数据需重新获取 |\n")
    f.write("| 原始RGZ训练星表 | ❌ 丢失 | 在已废弃Linux服务器 |\n")
    f.write("| WISE全天交叉表 | ❌ 丢失 | 需从CatWISE查询 |\n")

print(f"\n问题清单已写入: {ISSUES_LOG}")
print("Done!")
