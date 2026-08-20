# GitHub 共享改造方案 (GITHUB_SHARING_PLAN.md)

> 创建：2026-08-03 | 状态：**执行中 (2026-08: 本地提交就绪, 待 push)**
> 目标：与合作伙伴通过 GitHub 共享代码/权重/文档；具体数据（星表、切图）存服务器, 不上传 GitHub。
> 触发执行条件：合作方确认仓库结构后。

**执行记录 (2026-08)**：
- 仓库：`https://github.com/hahedaftyudu-ly/SKA-GalaxyClassifier`
- 分支：`feature/emu-test`（本地 2 个提交：代码+文档 2.3MB / 权重 641MB）
- 权重决策：**方式 A（直接进仓库）**——8 个权重全部 <100MB 单文件限制
- 待办：`git push -u origin feature/emu-test`（需本机 Git Credential Manager 登录，沙箱内无法弹认证框）

---

## 一、总体方案

| 进 GitHub | 留服务器 |
|-----------|----------|
| 全部代码 (`main.py` / `mynetwork.py` / `crossmatch/` / `tools/` / `optical/` / `radio/`) | `data/` (SDSSxWISE 1.3GB 等星表) |
| 文档 (`README.md` + `docs/`) | `source_cutouts/` (~800MB 切图) |
| 权重文件 (556MB, 方式见第三节) | 运行日志 / 缓存 / 中间产物 |

数据共享通过 DATA_REGISTRY.md (docs/) 对表核对批次, 合作者在服务器按相同目录结构放置。

---

## 二、执行前必须解决的 4 个问题

### 必改 1：硬编码绝对路径 (16 处, 14 个文件)
**现状量化**：`C:\Users\HaheDaftYuDu\SKA项目\GalaxyClassifier` 硬编码在：
- `main.py`、`crossmatch/cross_matching.py`、`optical/predicting_ps_data.py`、`rebuild_catalogs.py`
- `tools/` 下 10 个脚本 (crossmatch_norris / download_ps_cutouts / download_vlass_radio / generate_report_pdf / north_sky_validate / query_catwise / sdss_ab_validate / south_sky_test / thesis_validate / validate_pipeline)

**修复方案**：统一为已存在的相对定位模式（参考 `run_cdfs_inference.py:21`）：
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
删除 `os.chdir(PROJ)` 调用。

### 必改 2：`.gitignore` + 目录占位
```gitignore
# 数据 (服务器存储)
data/
source_cutouts/
# 权重: 若选 Releases 方式则加
crossmatch/models/*.pt
# 运行时产物
__pycache__/  *.pyc  .tmp*/  *.log
validation_report.html  validation_report.pdf
accuracy_vs_*  # 训练过程产物
```
用 `.gitkeep` 保住 `data/catalogs/`、`source_cutouts/`、`crossmatch/models/` 目录结构。

### 必改 3：数据根目录可配置 (服务器协同关键)
```python
DATA_ROOT = os.environ.get("GALAXY_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
```
- 本地默认 `./data`
- 服务器: `export GALAXY_DATA_ROOT=/data/galaxy`
- 同一套代码两边跑, 所有 `os.path.join(PROJECT_ROOT, "data", ...)` 改走 `DATA_ROOT`

### 必改 4：README / docs 通用化
- README.md、docs/GUIDE.md 中 `C:\Users\...` 路径描述 → 相对路径 + 环境变量说明
- 运行方式章节改为通用命令 (不依赖 Windows 特定路径)

---

## 三、权重文件处理 — ⚠️ 待决策

| 方式 | 优点 | 缺点 |
|------|------|------|
| **A. 直接进仓库** | 最简单, clone 即用 | 每次 clone ~600MB; 一旦 push 进历史永久存在 (需 filter-repo 才能移除) |
| **B. GitHub Releases (推荐)** | clone 只有代码 ~1MB; 权重按需下载 | 需手动上传管理 |

- 现状: 7 个权重, 最大 86MB (均 < GitHub 100MB 单文件限制), 总量 556MB
- 不推荐 Git LFS: 免费额度 1GB 存储, 556MB 占一半, 且下载需装 LFS 客户端
- 决策后: 选 B 则 `.gitignore` 需加 `crossmatch/models/*.pt`

---

## 四、分支与协作策略

- `main`: 稳定版 (论文方法验证通过的代码)
- 每人一个 feature 分支 + PR 合并
- 服务器数据目录约定: `/data/galaxy/{catalogs,cutouts,models,logs}` (与本地结构镜像)
- 环境要求: Python 3.12 + PyTorch 2.6 (requirements.txt 已更新), 注意 Windows/Linux 差异 (num_workers=0 等)

---

## 五、执行步骤 (记录, 待批准)

1. [ ] 批量修复 16 处硬编码路径 → 相对定位模式
2. [ ] 代码加 `GALAXY_DATA_ROOT` 环境变量支持 (替换 data 路径引用)
3. [ ] 写 `.gitignore` + `.gitkeep` 占位
4. [ ] README/GUIDE 路径通用化
5. [ ] `git init` + 首次提交 (不含 data/source_cutouts)
6. [ ] 权重方式决策 (A 直接进 / B Releases)
7. [ ] 关联 remote + push
8. [ ] 合作者在服务器部署 (设 GALAXY_DATA_ROOT, 按 DATA_REGISTRY 放置数据)

---

## 六、已知风险提醒

1. 权重进 Git 历史后不可轻易移除 (决策点必须先行)
2. GitHub 单文件 100MB 硬限制 — 将来新权重 >100MB 必须走 Releases/LFS
3. Windows 专属代码 (tools/readline.py dummy、GBK 编码规避) 在 Linux 服务器无影响但需确认
4. 数据下载脚本共享后, 合作者可在服务器重建数据 — 已有幂等检查 + DATA_REGISTRY 防止重复下载
