# -*- coding: utf-8 -*-
"""
生成 EMU 模型B适用性测试汇报幻灯片 PDF (16:9, 大字号, 中文)

内容: 三种测试方法 (零样本三组对照 / XGBoost模型A / 微调射电分支) + 数据 + 结果 + 结论
文案为正式学术表述 (2026-08 修订版)。
用法: python tools/generate_emu_slides_pdf.py
输出: docs/EMU_TEST_SLIDES.pdf
"""
import os
from fpdf import FPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "docs", "EMU_TEST_SLIDES.pdf")

W, H = 280, 158
MARGIN = 16
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"

# 字号 (正式版: 正文 14, 表格 13, 适配内容较多的页面)
TITLE_SIZE = 32
H2_SIZE = 24
H3_SIZE = 18
BODY_SIZE = 14
TABLE_SIZE = 13
SMALL_SIZE = 12

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
        self.cell(25, 8, str(page), align="R")
        self.set_y(MARGIN + 6)

    def bullet(self, text, color=DARK, size=BODY_SIZE, indent=0):
        self.set_text_color(*color)
        self.set_font("CN", "", size)
        self.set_x(MARGIN + indent)
        self.multi_cell(W - 2 * MARGIN - indent, size * 0.58,
                        "· " + text, align="L")
        self.ln(0.8)

    def h3(self, text):
        self.set_text_color(*YELLOW)
        self.set_font("CN", "", H3_SIZE)
        self.set_x(MARGIN)
        self.cell(W - 2 * MARGIN, H3_SIZE * 0.55, text)
        self.ln(H3_SIZE * 0.55 + 0.5)

    def spacer(self, h=4):
        self.ln(h)

    def table_wrap(self, headers, rows, col_widths, highlight_good=(), highlight_bad=()):
        self.set_font("CN", "", TABLE_SIZE)
        rh = TABLE_SIZE * 0.55 + 2.5
        x0 = MARGIN
        total_w = sum(col_widths)

        def wrap_lines(text, cw):
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
        self.set_fill_color(40, 50, 70)
        self.rect(x0, y, total_w, rh, "F")
        self.set_text_color(255, 255, 255)
        xx = x0
        for h, cw in zip(headers, col_widths):
            self.set_xy(xx, y + 0.5)
            self.cell(cw, rh, h, align="C")
            xx += cw
        y += rh
        for i, row in enumerate(rows):
            wrapped = [wrap_lines(v, cw) for v, cw in zip(row, col_widths)]
            row_h = max(len(w) for w in wrapped) * rh
            self.set_fill_color(235, 238, 245) if i % 2 == 0 else self.set_fill_color(245, 247, 250)
            self.rect(x0, y, total_w, row_h, "F")
            self.set_draw_color(150, 155, 170)
            self.rect(x0, y, total_w, row_h, "D")
            xx = x0
            for w, cw in zip(wrapped, col_widths):
                joined = "".join(w)
                color = DARK
                if any(k in joined for k in highlight_good):
                    color = GREEN
                elif any(k in joined for k in highlight_bad):
                    color = RED
                self.set_text_color(*color)
                for k, line in enumerate(w):
                    self.set_xy(xx + 1, y + 0.5 + k * rh)
                    self.cell(cw - 2, rh, line, align="L")
                xx += cw
            y += row_h
        self.set_y(y + 3)


def main():
    pdf = Slides()
    page = 0

    # ---- 1. 封面 ----
    pdf.add_page()
    page += 1
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, W, H, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("CN", "", TITLE_SIZE)
    pdf.set_xy(MARGIN, 42)
    pdf.cell(W - 2 * MARGIN, 16, "模型B在EMU上的测试", align="C")
    pdf.set_font("CN", "", H2_SIZE)
    pdf.set_xy(MARGIN, 66)
    pdf.cell(W - 2 * MARGIN, 12, "三种做法的对比：不改模型 · 更换模型A · 微调射电分支", align="C")
    pdf.set_font("CN", "", 16)
    pdf.set_text_color(200, 215, 230)

    # ---- 2. 数据准备 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("数据准备", page)
    pdf.h3("数据构成")
    pdf.bullet("射电拼图：EMU_PS_IMAGE.taylor.0.fits，18 角秒圆波束，2 角秒/像素，约 4.5 万 × 3.4 万像素")
    pdf.bullet("分量星表：178,921 个射电分量，95% 有 WISE 对应体，60% 有 DES 光学匹配")
    pdf.bullet("宿主标签：DRAGNs 星表，3,557 个双瓣射电源经人工逐一认证，其中 3,181 个确认宿主")
    pdf.bullet("测试集：100 个源 = 50 个有宿主的 DRAGN + 50 个远处随机选取的分量")
    pdf.spacer(2)
    pdf.h3("标定修正（基于实测）")
    pdf.bullet("拼图为 2 角秒/像素，切图取 432 角秒（216 像素），与模型原有 CenterCrop(216) 对齐，无需重采样")
    pdf.bullet("位置特征缩放系数由 4 调整为 8（对应 2 角秒/像素 → 光学 0.25 角秒/像素）")

    # ---- 3. 做法一 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("做法一：模型参数不变，仅替换 A 概率输入", page)
    pdf.h3("方法")
    pdf.bullet("保持模型B权重不变，仅替换其 A 概率输入（三种来源），检验该输入对模型B输出的影响")
    pdf.table_wrap(
        ["组", "A 概率来源", "说明"],
        [
            ["raw", "CNN 模型A 真实输出", "南天上模型A退化，基本全判为恒星"],
            ["uniform", "[1/3, 1/3, 1/3]", "中性占位，不偏向任何类别"],
            ["prior", "[0.45, 0.45, 0.10]", "物理先验：射电选源不可能是恒星"],
        ],
        [32, 82, 136], highlight_bad=("退化",))
    pdf.spacer(2)
    pdf.h3("结果（100 个测试源）")
    pdf.table_wrap(
        ["指标", "raw", "uniform", "prior"],
        [
            ["宿主召回", "0/50", "0/50", "0/50"],
            ["分离度", "-0.018", "-0.018", "-0.018"],
            ["非宿主特异性", "0.980", "0.980", "0.980"],
        ],
        [90, 55, 55, 55], highlight_bad=("-0.018", "0/50"))
    pdf.spacer(3)
    pdf.h3("结论")
    pdf.bullet("零样本下 B 宿主召回为零：50 个宿主全部漏检", color=RED)
    pdf.bullet("三种来源输出几乎一致（差异 <0.0024）：B 对该输入不敏感，判断由图像分支决定", color=RED)
    pdf.bullet("修改 A 概率无法改善 B 的性能")

    # ---- 4. 做法二 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("做法二：以 XGBoost 构建替代模型A", page)
    pdf.h3("方法")
    pdf.bullet("以 XGBoost 构建替代模型A：仅使用 W1/W2 红外星等，无需图像输入")
    pdf.bullet("特征全部来自星表，无需额外下载图像，适用于南天场景")
    pdf.spacer(2)
    pdf.h3("结果")
    pdf.table_wrap(
        ["模型A", "宿主源分类", "非宿主源分类", "B 的输出"],
        [
            ["CNN 模型A", "恒星主导 (0.995)", "恒星主导", "分离度 -0.018"],
            ["XGBoost A", "恒星为主 (0.633)", "星系为主 (0.453)", "分离度 -0.018"],
        ],
        [52, 70, 70, 58], highlight_good=("0.453",), highlight_bad=("0.995",))
    pdf.spacer(3)
    pdf.h3("结论")
    pdf.bullet("XGBoost A 输出分布更合理：宿主与非宿主出现可分性，不再集中于恒星类", color=GREEN)
    pdf.bullet("但 B 的输出保持不变，表明其对 A 概率输入不敏感", color=RED)
    pdf.bullet("更换 A 无法改善 B，问题在于 B 的图像分支")

    # ---- 5. 做法三 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("做法三：微调模型B射电分支", page)
    pdf.h3("方法")
    pdf.table_wrap(
        ["项", "设置"],
        [
            ["训练数据", "1,187 个源：600 个 DRAGN 宿主 + 587 个远处分量（测试 100 源全部排除）"],
            ["冻结部分", "光学分支（eval 模式，批归一化使用运行统计）+ A 概率固定为均匀值"],
            ["训练部分", "射电分支 + 最后两层全连接；Adam，lr=3e-4，梯度裁剪，10 轮"],
            ["模型选择", "在 10% 验证集上按准确率选取最优权重，最终在留出的 100 个测试源上评估"],
        ],
        [38, 212])
    pdf.spacer(2)
    pdf.h3("训练过程（验证集分离度）")
    pdf.bullet("第 1 轮 0.03 → 第 5 轮 0.23 → 第 9 轮 0.475", color=GREEN)
    pdf.bullet("分离度持续上升，表明射电分支正在学习 EMU 图像特征")

    # ---- 6. 结果对比 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("结果：留出测试集（微调前后对比）", page)
    pdf.table_wrap(
        ["指标", "微调前", "微调后", "论文（北天）"],
        [
            ["准确率", "0.490", "0.760", "0.977"],
            ["宿主召回", "0/50 (0%)", "32/50 (64%)", "87.3%"],
            ["非宿主特异性", "0.980", "0.880", "98.3%"],
            ["宿主平均概率", "0.000", "0.588", "0.646"],
            ["非宿主平均概率", "0.018", "0.185", "0.016"],
            ["分离度", "-0.018", "+0.404", "0.630"],
        ],
        [95, 55, 55, 55],
        highlight_good=("0.760", "32/50", "+0.404", "0.588"),
        highlight_bad=("-0.018", "0/50"))
    pdf.spacer(4)
    pdf.h3("结果解读")
    pdf.bullet("在仅 1,200 个训练源、仅微调射电分支的条件下，分离度由 -0.018 升至 +0.404，宿主召回提升至 64%", color=GREEN)
    pdf.bullet("与论文水平仍有差距，但改进方向明确；剩余差距主要来自数据规模与方法细节")
    pdf.bullet("代价：误报增加，非宿主特异性由 98% 降至 88%")

    # ---- 7. 小结 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("小结：三步实验的结论", page)
    pdf.h3("模型B 输入构成")
    pdf.bullet("B 将射电特征（20 维）、光学特征（10 维）、A 概率（3 维）、位置特征（3 维）拼接后决策")
    pdf.bullet("A 概率仅占 3/36——前两种做法失效原因一致：该输入对整体决策影响有限")
    pdf.spacer(2)
    pdf.h3("三步结论")
    pdf.bullet("第一步：仅替换 A 概率 → 无效，B 对该输入不敏感", color=RED)
    pdf.bullet("第二步：更换 XGBoost A → A 输出改善，但 B 输出不变", color=RED)
    pdf.bullet("第三步：微调 B 射电分支 → 分离度翻正，宿主召回 0 → 64%", color=GREEN)
    pdf.spacer(2)
    pdf.h3("结论")
    pdf.bullet("模型B思路在 EMU 上可行；关键在占主导地位的图像分支，而非概率输入")

    # ---- 8. 局限与下一步 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("局限与下一步", page)
    pdf.h3("当前局限")
    pdf.bullet("光学暂用 DES grz 三波段近似（完整 5 波段需 Data Lab 账号）")
    pdf.bullet("训练规模 1,187 个源偏小；负样本可能混入真实宿主，标签存在噪声")
    pdf.bullet("仅训练 10 轮、无学习率调度；测试集规模 100 个源有限")
    pdf.spacer(2)
    pdf.h3("下一步")
    pdf.bullet("① 训练集扩展至全量：3,181 个宿主 + 数千负样本，预计分离度进一步提升")
    pdf.bullet("② 获取完整 grizY 光学数据，适配光学分支")
    pdf.bullet("③ 按形态（FR1/FR2/弯尾）分层评估，分析 18″ 分辨率下的退化差异")
    pdf.bullet("④ 基于 DRAGNs 的 CATWISE ID 反查宿主 WISE 星等，降低特征噪声")

    # ---- 9. 结论 ----
    pdf.add_page()
    page += 1
    pdf.slide_header("结论", page)
    pdf.set_fill_color(26, 82, 118)
    pdf.rect(MARGIN, 40, W - 2 * MARGIN, 58, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("CN", "", 24)
    pdf.set_xy(MARGIN, 50)
    pdf.multi_cell(W - 2 * MARGIN, 14,
                   "模型B在 EMU 数据上可行：\n"
                   "微调射电分支后，分离度由 -0.018 提升至 +0.404，宿主召回率达 64%",
                   align="C")
    pdf.set_font("CN", "", 16)
    pdf.set_text_color(200, 215, 230)
    pdf.set_xy(MARGIN, 116)
    pdf.multi_cell(W - 2 * MARGIN, 9,
                   "前两种做法（替换概率、更换模型A）无效，原因在于 A 概率在 B 的输入中占比很小，\n"
                   "需适配的是图像分支；剩余差距主要来自数据规模与方法细节，而非原理性问题。",
                   align="C")
    pdf.output(OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
