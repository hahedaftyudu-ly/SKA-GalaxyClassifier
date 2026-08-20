"""
生成项目现状汇报幻灯片 PDF (16:9, 大字号)

与 docs/STATUS_SLIDES.html 内容一致, 用 fpdf2 渲染为大字号 PDF。
用法: python tools/generate_status_slides_pdf.py
输出: docs/STATUS_SLIDES.pdf
"""
import os
from fpdf import FPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "docs", "STATUS_SLIDES.pdf")

# 16:9 幻灯片尺寸 (mm)
# 注意: fpdf2 的 orientation='L' 会把 format 元组按竖版(宽,高)交换,
# 因此这里按 (高,宽) 传入才能得到 280x158 横版
W, H = 280, 158
MARGIN = 16
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

# 字号 (大字号版)
TITLE_SIZE = 30
H2_SIZE = 24
H3_SIZE = 18
BODY_SIZE = 16
TABLE_SIZE = 14
SMALL_SIZE = 13

NAVY = (26, 82, 118)
LIGHT = (110, 130, 150)
RED = (200, 60, 60)
GREEN = (70, 140, 90)
YELLOW = (180, 130, 20)
DARK = (40, 40, 40)


class Slides(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format=(H, W))
        self.add_font("CN", "", FONT_PATH)
        self.set_auto_page_break(False)

    def slide_header(self, title, page):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, W, 14, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("CN", "", 14)
        self.set_xy(MARGIN, 3)
        self.cell(0, 8, title, align="L")
        self.set_xy(W - MARGIN - 25, 3)
        self.cell(25, 8, f"{page}", align="R")
        self.set_y(MARGIN + 6)

    def bullet(self, text, color=DARK, size=BODY_SIZE, indent=0):
        self.set_text_color(*color)
        self.set_font("CN", "", size)
        self.set_x(MARGIN + indent)
        self.multi_cell(W - 2 * MARGIN - indent, size * 0.62,
                        "· " + text, align="L")
        self.ln(1.2)

    def h3(self, text):
        self.set_text_color(*YELLOW)
        self.set_font("CN", "", H3_SIZE)
        self.set_x(MARGIN)
        self.cell(W - 2 * MARGIN, H3_SIZE * 0.62, text)
        self.ln(H3_SIZE * 0.62 + 1)

    def table(self, headers, rows, col_widths, size=None):
        size = size or TABLE_SIZE
        self.set_font("CN", "", size)
        rh = size * 0.62 + 3
        self.set_fill_color(40, 50, 70)
        self.set_text_color(255, 255, 255)
        x0 = MARGIN
        for h, cw in zip(headers, col_widths):
            self.set_xy(x0, self.get_y())
            self.cell(cw, rh, h, border=1, align="C", fill=True)
            x0 += cw
        self.ln(rh)
        for i, row in enumerate(rows):
            self.set_fill_color(235, 238, 245) if i % 2 == 0 else self.set_fill_color(245, 247, 250)
            x0 = MARGIN
            for v, cw in zip(row, col_widths):
                self.set_xy(x0, self.get_y())
                color = DARK
                if "100%" in str(v) or "93.7" in str(v) or "82.9" in str(v):
                    color = GREEN
                elif "退化" in str(v) or "限流" in str(v) or "未解除" in str(v) or "停止" in str(v) or "不可达" in str(v) or "503" in str(v) or "零判别" in str(v):
                    color = RED
                self.set_text_color(*color)
                self.cell(cw, rh, str(v), border=1, align="C", fill=True)
                x0 += cw
            self.ln(rh)

    def table_wrap(self, headers, rows, col_widths):
        """支持自动换行的表格 — 手动按字符换行 + 整块矩形背景, 避免 multi_cell 绘制混乱"""
        self.set_font("CN", "", TABLE_SIZE)
        rh = TABLE_SIZE * 0.62 + 3
        x0 = MARGIN
        total_w = sum(col_widths)

        def wrap_lines(text, cw):
            """按字符宽度手动换行 (CJK 无空格, 不能依赖 word 模式)"""
            lines, cur = [], ""
            for ch in str(text):
                if cur and self.get_string_width(cur + ch) > cw - 4:
                    lines.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                lines.append(cur)
            return lines or [""]

        y = self.get_y()
        # 表头: 深色整块
        self.set_fill_color(40, 50, 70)
        self.rect(x0, y, total_w, rh, "F")
        self.set_text_color(255, 255, 255)
        xx = x0
        for h, cw in zip(headers, col_widths):
            self.set_xy(xx, y + 0.5)
            self.cell(cw, rh, h, align="C")
            xx += cw
        y += rh
        # 数据行
        for i, row in enumerate(rows):
            wrapped = [wrap_lines(v, cw) for v, cw in zip(row, col_widths)]
            row_h = max(len(w) for w in wrapped) * rh
            # 背景整块 + 细边框
            self.set_fill_color(235, 238, 245) if i % 2 == 0 else self.set_fill_color(245, 247, 250)
            self.rect(x0, y, total_w, row_h, "F")
            self.set_draw_color(150, 155, 170)
            self.rect(x0, y, total_w, row_h, "D")
            # 文本
            xx = x0
            for w, cw in zip(wrapped, col_widths):
                color = DARK
                joined = "".join(w)
                if any(k in joined for k in ("100%", "93.7", "82.9")):
                    color = GREEN
                elif any(k in joined for k in ("退化", "零判别")):
                    color = RED
                self.set_text_color(*color)
                for k, line in enumerate(w):
                    self.set_xy(xx + 1, y + 0.5 + k * rh)
                    self.cell(cw - 2, rh, line, align="L")
                xx += cw
            y += row_h
        self.set_y(y + 2)


def main():
    pdf = Slides()

    # ---- 1. 封面 ----
    pdf.add_page()
    pdf.set_y(H * 0.30)
    pdf.set_text_color(*DARK)
    pdf.set_font("CN", "", 40)
    pdf.set_x(MARGIN)
    pdf.cell(0, 18, "GalaxyClassifier 项目现状汇报")
    pdf.ln(22)
    pdf.set_font("CN", "", 20)
    pdf.set_text_color(*LIGHT)
    pdf.set_x(MARGIN)
    pdf.cell(0, 12, "射电源光学宿主识别 · 两阶段堆叠模型")
    pdf.ln(14)
    pdf.set_font("CN", "", 14)
    pdf.cell(0, 10, "2026-08-09")
    pdf.ln(10)
    pdf.cell(0, 10, "VLASS 射电 + PanSTARRS 光学 + WISE 红外 · 模型A（光学三分类）→ 模型B（宿主认证）")

    # ---- 2. 模型B独立验证 ----
    pdf.add_page()
    pdf.slide_header("一、模型 B 独立验证", 2)
    pdf.h3("历史验证（CDFS / Norris06 测试集）")
    pdf.table(["指标", "数值"], [
        ["准确率", "97.7%"],
        ["宿主召回率", "87.3%（62/71）"],
        ["非宿主特异性", "98.3%（1250/1272）"],
        ["宿主/非宿主概率分离", "0.646 vs 0.016（分离度极佳）"],
    ], [70, 125], size=12.5)
    pdf.ln(1.5)
    pdf.h3("本轮独立复验（2026-08-06，北天 RGZ）")
    pdf.bullet("数据：RGZ DR1 公民科学高置信标签（N_votes≥10、CL≥0.7），"
               "FIRST 北天选源，候选池 42658，抽样 10 源、7 源完成推理",
               size=12.5)
    pdf.bullet("方式：第三套独立样本复验（区别于 CDFS/Norris06 测试集），"
               "手动输入各源三分类概率（GALAXY/QSO/STAR）后经模型 B 推理，全链路本地完成",
               size=12.5)
    pdf.bullet("结果：7/7 全部正确识别（召回 100%）；host_prob 均值 0.611、"
               "范围 0.608~0.615，均高于 0.5 阈值",
               color=GREEN, size=12.5)
    pdf.bullet("局限：样本小且全为正样本（无负样本对照，无法评估精确率/误报率）；"
               "host_prob 窄带贴近阈值需校准；仅北天",
               size=12.5)

    # ---- 3. 动机 ----
    pdf.add_page()
    pdf.slide_header("二、模型 A 替换实验（XGBoost）——动机", 3)
    pdf.bullet("原版 CNN 模型 A 退化：7 个北天 RGZ 宿主全部输出 STAR≈0.99，逐源零判别力",
               color=RED)
    pdf.bullet("模型 A 训练切图已丢失，重训需全部重新下载，成本高")
    pdf.bullet("模型 A 在管线中仅提供三类概率 → 用光度学特征 + 树模型可完全替代，彻底摆脱图像依赖")
    pdf.bullet("方案：XGBoost 在 SDSSxWISE 全表（394 万行光谱标签）上训练三分类")

    # ---- 4. v1 结果 ----
    pdf.add_page()
    pdf.slide_header("三、XGBoost-A v1（WISE-only）结果", 4)
    pdf.h3("训练")
    pdf.bullet("抽样 60 万行（每类 20 万），8:2 划分；特征：W1、W2 星等 + W1-W2 颜色", size=15)
    pdf.bullet("训练耗时 8 秒（纯星表、无 GPU）；模型体积 1.6MB", color=GREEN, size=15)
    pdf.ln(1)
    pdf.h3("三分类验证（19.7 万行）—— 总准确率 82.9%")
    pdf.table(["真值 \\ 预测", "GALAXY", "QSO", "STAR"], [
        ["GALAXY", "80.0%", "3.3%", "16.7%"],
        ["QSO", "2.2%", "93.7%", "4.2%"],
        ["STAR", "21.5%", "3.6%", "74.9%"],
    ], [45, 40, 40, 40])
    pdf.ln(1)
    pdf.bullet("QSO 识别强（W1-W2 为 QSO 色选判据）；GALAXY-STAR 互混为 WISE 无光学波段的物理限制",
               size=15)
    pdf.bullet("特征重要性：W1-W2 颜色 0.56 / W1 星等 0.27 / W2 星等 0.17", size=15)
    pdf.bullet("实例：J165708 的 W1-W2=+0.76 → 判 QSO（0.97）；"
               "J170204 的 W1-W2=-0.03 → 判恒星（0.94）——模型学到的正是 WISE 色选判据",
               color=YELLOW, size=12.5)

    # ---- 5. 新旧模型对比 ----
    pdf.add_page()
    pdf.slide_header("四、新旧模型对比（CNN-A → XGBoost-A）", 5)
    pdf.table_wrap(["维度", "旧模型 CNN-A", "新模型 XGBoost-A"], [
        ["识别数据选取",
         "5 波段光学切图 + W1/W2 星等；重训需重新下载全部切图",
         "仅 3 个星表数值：W1/W2 星等 + W1-W2 颜色；零图像依赖"],
        ["识别能力",
         "7 源全部判 STAR≈0.99，逐源无区分",
         "逐源可解释，QSO/星系判定与 W1-W2 物理判据一致"],
        ["最终结果（三分类验证）",
         "射电选源上分类失效",
         "验证准确率 82.9%，QSO 召回 93.7%"],
    ], [36, 80, 80])
    pdf.ln(4)
    pdf.bullet("结论：新模型 A 分类可靠、输出可解释，数据需求从 5 波段切图降为 3 个星表数值",
               color=GREEN, size=BODY_SIZE)

    # ---- 6. v2 计划 ----
    pdf.add_page()
    pdf.slide_header("五、XGBoost-A v2（WISE + 光学特征）——进行中", 6)
    pdf.bullet("设计：加入 SDSS ugriz 光度（5 psfMag + 4 颜色 + 形态代理）")
    pdf.bullet("目标：拆掉 GALAXY-STAR 混淆，预期 82.9% 提升至 90% 以上")
    pdf.bullet("工具链已就绪：批量拉取脚本（断点续存）、v2 训练脚本、9 万源坐标表",
               color=GREEN)
    pdf.bullet("评估：沿用 v1 端到端流程，与 CNN 版、v1 三方对比")

    # ---- 7. 阻塞 ----
    pdf.add_page()
    pdf.slide_header("六、当前阻塞：SDSS 光度数据拉取被限流", 7)
    pdf.table(["数据源", "状态"], [
        ["官方 SDSS（dr16/dr18）", "限流 25+ 小时未解除（连接正常、查询静默返回空）"],
        ["China-VO CasJobs", "服务已停止"],
        ["China-VO skyserver 镜像", "无可用 API"],
        ["MAST PS1 星表", "网络不可达"],
        ["CDS VizieR（含国内镜像）", "503 / 服务失效"],
    ], [70, 125])
    pdf.ln(4)
    pdf.bullet("影响范围：仅 v2 训练数据获取；已完成的验证、推理管线、v1 模型均不受影响",
               size=BODY_SIZE)

    # ---- 8. 下一步 ----
    pdf.add_page()
    pdf.slide_header("七、下一步计划", 8)
    pdf.h3("短期（数据恢复后）")
    pdf.bullet("拉取 9 万源 SDSS 光度（脚本就绪、断点续存）→ 训练 v2 → 端到端三方对比", size=15)
    pdf.bullet("备选：用本地 PS 切图为评估源提取光学特征，端到端演示不依赖服务器", size=15)
    pdf.h3("长期（EMU 迁移）")
    pdf.bullet("南天光学巡天（DES/DELVE/SkyMapper）光度路线；XGBoost-A 结构天然适配星表迁移", size=15)
    pdf.bullet("射电预处理常数按 ASKAP/EMU 分辨率重新标定", size=15)

    pdf.output(OUT_PATH)
    print(f"PDF generated: {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB, {pdf.pages_count} pages)")


if __name__ == "__main__":
    main()
