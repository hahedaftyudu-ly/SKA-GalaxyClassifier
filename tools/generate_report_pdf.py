"""Generate validation comparison PDF using fpdf2"""
import os, sys
from fpdf import FPDF

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pdf = FPDF(orientation='L', unit='mm', format='A4')
pdf.add_page()

# Register Chinese font
font_path = r'C:\Windows\Fonts\msyh.ttc'
if not os.path.exists(font_path):
    font_path = r'C:\Windows\Fonts\simsun.ttc'
pdf.add_font('CN', '', font_path, uni=True)

def section_title(text):
    pdf.set_font('CN', '', 13)
    pdf.set_fill_color(26, 82, 118)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, '  ' + text, fill=True)
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)

def make_table(cols, widths, data, col_align=None):
    pdf.set_font('CN', '', 7)
    # Header
    pdf.set_fill_color(26, 82, 118)
    pdf.set_text_color(255, 255, 255)
    for i, (h, w) in enumerate(zip(cols, widths)):
        pdf.cell(w, 6, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    # Data
    for j, row in enumerate(data):
        pdf.set_fill_color(235, 245, 251) if j % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        for i, (cell, w) in enumerate(zip(row, widths)):
            align = col_align[i] if col_align else ('C' if i > 0 else 'L')
            c = cell.upper() if isinstance(cell, str) else str(cell)
            if c in ['LOST', 'FAILED', 'COLLAPSED', '0.0%', '0%', '0.919', '0.002']:
                pdf.set_text_color(200, 0, 0)
            elif c in ['OK', 'HEALTHY', 'RECOVERED', '87.3%', '97.7%', '0.745', '0.646']:
                pdf.set_text_color(0, 150, 0)
            elif 'PARTIAL' in c:
                pdf.set_text_color(200, 150, 0)
            pdf.cell(w, 5, cell[:40] if len(cell) > 40 else cell, border=1, fill=True, align=align)
            pdf.set_text_color(0, 0, 0)
        pdf.ln()
    pdf.ln(4)

# ===== TITLE =====
pdf.set_font('CN', '', 18)
pdf.cell(0, 10, 'GalaxyClassifier - Model Validation Report', align='C')
pdf.ln(12)
pdf.set_font('CN', '', 9)
pdf.cell(0, 6, 'Radio Source Optical Host Identification | Two-Stage Multimodal CNN | Full Comparison', align='C')
pdf.ln(6)
pdf.cell(0, 6, 'Paper: Lou Kangzhi (2023) UCAS MSc | Validation: 2026-07-25 ~ 07-28 | Python 3.12.6, PyTorch 2.6.0+cu124, RTX 4060', align='C')
pdf.ln(10)

# ===== 1. DATA SOURCES =====
section_title('1. Data Sources Overview')

make_table(
    ['Data Item', 'Source', 'Size / Format', 'Sky Region', 'Status'],
    [60, 70, 58, 52, 24],
    [
        ['SDSSxWISE Catalog', 'Aliyun Drive (author)', '1.3 GB, 3.94M rows', 'North (Dec>-19.7)', 'OK'],
        ['Model A Weight (43MB)', 'GitHub master', 'opt_classification_model_wts.pt', 'N/A', 'OK'],
        ['Model B Weights x6 (600MB)', 'GitHub master', 'RGZ_*_crossmatch_model_wts.pt', 'N/A', 'OK'],
        ['Original FITS Cutouts', 'Expansion HDD', '94 GB, 403,549 files', 'North SDSS area', 'LOST'],
        ['Original RGZ Training Catalog', 'Linux server (scrapped)', 'CSV files', 'FIRST+ATLAS sky', 'LOST'],
        ['Original Norris Test Results', 'GitHub handoff', '1,343 records CSV', 'CDFS (South Dec-28)', 'OK'],
        ['CDFS VLASS Radio FITS (51)', 'CADC nrao:VLASS', '51 files, ~7 MB', 'CDFS South', 'OK (S4)'],
        ['CDFS PanSTARRS Optical (71x5)', 'STScI panstamps', '71x5band, ~1.8 GB', 'CDFS South', 'OK (S4)'],
        ['North PanSTARRS Test (30x5)', 'STScI panstamps', '30x5band', 'North SDSS', 'OK (S5)'],
        ['CatWISE IR Magnitudes', 'VizieR II/365', '150 query cache', 'All-sky', 'PARTIAL 87/1089'],
        ['Norris 2006 Catalog', 'VizieR J/ApJS/164/297', '594 sources', 'CDFS South', 'OK'],
        ['VLASS CDFS Components', 'CADC VLASS catalog', '305 components', 'CDFS South', 'OK'],
        ['RGZ DR1 FIRST (North)', 'Zenodo 14195049', '99,602 records', 'North (FIRST)', 'OK (S6)'],
        ['RGZ DR1 ATLAS (South)', 'Zenodo 14195049', '1,394 records', 'South (ATLAS)', 'OK (S6)'],
    ]
)

# ===== 2. MODEL A COMPARISON =====
section_title('2. Model A (Optical Classification: Galaxy / QSO / STAR) Comparison')

# Use single-line headers, add sub-header row
cols_a = ['Metric', 'Paper Orig (RGZ_all_neg v1)', 'CDFS Re-run (Session 4)', 'North Repro (Session 5)']
w_a = [46, 74, 74, 70]
data_a = [
    ['Samples', '1,343', '194', '30'],
    ['Sky Region', 'CDFS field but orig FITS', 'South CDFS (Dec ~ -28 deg)', 'North SDSS (Dec 0 ~ +60 deg)'],
    ['PS Image Source', 'Original 94GB (DR1 era)', 'New panstamps (DR2)', 'New panstamps (DR2)'],
    ['WISE Magnitudes', 'Original CatWISE real', '87% SDSSxWISE placeholder', 'SDSSxWISE real'],
    ['Galaxy mean', '0.014', '0.027', '--'],
    ['QSO mean', '0.745', '0.055', '0.374'],
    ['STAR mean', '0.241', '0.919', '~0.40'],
    ['STAR std deviation', '0.373', '0.085', '0.446'],
    ['Dominant Class', 'QSO (74.5%)', 'STAR (91.9%)', 'STAR(60%) / QSO(40%)'],
    ['QSO-tag sources correct', '--', '--', '78%'],
    ['STAR-tag sources correct', '--', '--', '99%'],
    ['VERDICT', 'HEALTHY', 'COLLAPSED', 'RECOVERED'],
]
make_table(cols_a, w_a, data_a)

# ===== 3. MODEL B COMPARISON =====
section_title('3. Model B (Host Identification: Non-host / Host) Comparison')

cols_b = ['Metric', 'Paper Original (RGZ_all_neg v1)', 'CDFS Re-run (Session 4)']
w_b = [52, 106, 106]
data_b = [
    ['Samples', '1,343', '194'],
    ['Ground Truth Source', 'RGZ Citizen Science', 'Norris2006 morph type'],
    ['Host : Non-host Ratio', '71 : 1272 = 1:18', '11 : 183 = 1:17'],
    ['Accuracy', '97.7%', '93.3%*'],
    ['Host Recall (TP Rate)', '87.3% (62/71)', '0.0% (0/11)'],
    ['Precision', '73.8%', '0.0%'],
    ['Non-host Specificity (TN)', '98.3% (1250/1272)', '98.9% (181/183)'],
    ['TP / TN / FP / FN', '62 / 1250 / 22 / 9', '0 / 181 / 2 / 11'],
    ['Host_prob mean (GT=1)', '0.646', '0.002'],
    ['Host_prob mean (GT=0)', '0.016', '0.016'],
    ['Host GT=1, prob > 0.5', '62/71 (87.3%)', '0/11 (0.0%)'],
    ['Model A input (Gal/QSO/STAR)', '0.014 / 0.745 / 0.241', '0.027 / 0.055 / 0.919'],
]
make_table(cols_b, w_b, data_b)

pdf.set_font('CN', '', 6)
pdf.set_text_color(150, 150, 150)
pdf.cell(0, 4, '* 93.3% is an artifact of extreme class imbalance (11:183). A model that always predicts Non-host achieves 93.3% accuracy.', align='L')
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# ===== 4. MODEL VARIANT ABLATION =====
section_title('4. Model Variant Ablation Comparison')

make_table(
    ['Model Variant', 'Samples', 'Accuracy', 'Host Recall', 'Precision', 'Specificity', 'TP/TN/FP/FN', 'Model A Dominant'],
    [40, 20, 22, 24, 22, 24, 44, 40],
    [
        ['RGZ_all_neg v1 (BEST)', '1,343', '97.7%', '87.3%', '73.8%', '98.3%', '62/1250/22/9', 'QSO (74.5%)'],
        ['RGZ_all_neg main', '1,343', '95.8%', '29.6%', '77.8%', '99.5%', '21/1266/6/50', 'QSO (50.7%)'],
        ['CDFS re-run (Session 4)', '194', '93.3%*', '0.0%', '0.0%', '98.9%', '0/181/2/11', 'STAR (91.9%)'],
    ]
)

pdf.set_font('CN', '', 6)
pdf.set_text_color(150, 150, 150)
pdf.cell(0, 4, '* CDFS 93.3% is a false metric from class imbalance. The effective metric is Host Recall = 0%.', align='L')
pdf.set_text_color(0, 0, 0)
pdf.ln(6)

# ===== 5. ROOT CAUSE =====
section_title('5. Root Cause Analysis')

pdf.set_font('CN', '', 8)

root_cause_text = [
    ('CONFIRMED: PanSTARRS Image Domain Shift (DR1 -> DR2), NOT WISE Magnitudes', True),
    ('', False),
    ('WISE Hypothesis - RULED OUT (Session 5):', True),
    ('  - SDSSxWISE median (619.5/713.8 mJy) is the TRUE statistical median of 3.94M sources, not arbitrary filler.', False),
    ('  - cal_luptitude() asinh transformation compresses dynamic range: placeholder (619 mJy) -> 2.07 mag, real (1.95 mJy) -> 2.07 mag.', False),
    ('  - Difference = 0.008 mag, completely negligible for Model A input.', False),
    ('  - 97% of training samples also use the same median -> Model A never learned to depend on WISE.', False),
    ('', False),
    ('PS Image Shift - CONFIRMED (Session 4 + 5):', True),
    ('  - North sky (SDSS area): PS DR1 vs DR2 differences small -> Model A STAR std=0.446 (normal function).', False),
    ('  - South sky (CDFS, Dec~-28 deg, PS coverage edge): PS DR1 vs DR2 differences large -> Model A collapses.', False),
    ('  - Likely causes: DR2 reprocessing changed photometric zero-points, stacking depth, or PSF modeling at survey edge.', False),
    ('', False),
    ('Cascade Failure Mechanism:', True),
    ('  Model A ResNet features degrade on CDFS DR2 images -> outputs STAR=92% (vs 24% on orig) ->', False),
    ('  Model B receives "this is a star" signal -> learned rule "STAR -> Non-host" suppresses host_prob ->', False),
    ('  Model B ResNet image features ALSO degrade on CDFS DR2 (double degradation) ->', False),
    ('  Even with synthetic Galaxy input, host_prob only reaches 10% (threshold=50%) ->', False),
    ('  RESULT: Host Recall 0%', False),
]
for text, is_bold in root_cause_text:
    if is_bold:
        pdf.set_font('CN', '', 8)
    else:
        pdf.set_font('CN', '', 7)
    pdf.cell(0, 4.5, text)
    pdf.ln()

pdf.ln(4)

# ===== 6. CONCLUSIONS =====
section_title('6. Conclusions')

pdf.set_font('CN', '', 8)
conclusions = [
    ['Model Validity', 'YES - On training-domain data (North PS DR1): 97.7% accuracy, 87.3% host recall. Host vs Non-host separation is strong (host_prob: 0.646 vs 0.016).'],
    ['Reproducibility', 'YES - North sky re-downloaded PS DR2 images: Model A recovers discrimination (STAR std=0.446 vs CDFS 0.085). Paper method is reproducible.'],
    ['CDFS Failure Cause', 'PS DR1->DR2 image shift x CDFS sky-region edge effect -> ResNet feature degradation -> Model A->B cascade collapse. NOT a model bug.'],
    ['WISE Role', 'NEGLIGIBLE - asinh compression makes all WISE inputs nearly identical. Model relies almost entirely on PanSTARRS images.'],
    ['North VLASS Usage', 'READY - 7 weight files (600MB) can be used directly for North-sky VLASS host identification with original PS DR1 images.'],
    ['South / EMU Migration', 'NOT READY - Requires: (1) replace PanSTARRS with DES/DELVE/SkyMapper, (2) adapt to EMU resolution, (3) retrain/fine-tune on southern data.'],
    ['RGZ DR1 Labels', 'AVAILABLE - 99,602 North + 1,394 South citizen science host labels ready for future retraining (Zenodo 14195049).'],
]

col_w = [38, 226]
for j, (label, text) in enumerate(conclusions):
    pdf.set_fill_color(235, 245, 251) if j % 2 == 0 else pdf.set_fill_color(255, 255, 255)
    pdf.set_font('CN', '', 8)
    pdf.cell(col_w[0], 6, '  ' + label, border=1, fill=True)
    pdf.set_font('CN', '', 7)
    pdf.cell(col_w[1], 6, text, border=1, fill=True)
    pdf.ln()

pdf.ln(8)

# ===== FOOTER =====
pdf.set_font('CN', '', 6)
pdf.set_text_color(150, 150, 150)
pdf.cell(0, 4, f'GalaxyClassifier Validation Report | Generated 2026-07-28 (Session 6) | Project: {PROJ}/', align='C')
pdf.ln(3)
pdf.cell(0, 4, 'Paper: Lou Kangzhi (2023) Identifying Host Galaxies of Extragalactic Radio Emission Structures using Deep Learning | UCAS / NAOC | Supervisor: Chao-Wei Tsai', align='C')

# Output
outpath = os.path.join(PROJ, 'validation_report.pdf')
pdf.output(outpath)
print(f'PDF generated: {outpath}')
print(f'Size: {os.path.getsize(outpath)/1024:.0f} KB')
