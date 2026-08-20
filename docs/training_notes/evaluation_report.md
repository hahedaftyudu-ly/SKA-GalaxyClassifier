# CDFS Independent Validation Report

> Generated: 2026-07-26 12:17
> Model: Two-stage CNN (ResNet18 + WISE → Crossmatch)
> Test set: Norris2006 VLASS components via CDFS field (N=194)
> Environment: Python 3.12, PyTorch 2.6, RTX 4060 4GB
>
> 验证口径: 模型A仅光学输入(PS 5波段+WISE), 本报告即**光学+南天**组合;
> 射电(VLASS)数据仅用于模型B。

---

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | **93.3%** (181/194) |
| Precision | 0.0% |
| Recall (Sensitivity) | 0.0% |
| F1 Score | 0.0% |

## 2. Confusion Matrix

| | Predicted Non-host | Predicted Host |
|---|---|---|
| **Actual Non-host** | 181 (TN) | 2 (FP) |
| **Actual Host** | 11 (FN) | 0 (TP) |

- True Negative Rate (Specificity): 98.9%
- True Positive Rate (Recall): 0.0%

## 3. Stratified by Norris2006 Morphological Type

| Norris Type | Samples | Correct | Accuracy |
|-------------|---------|---------|----------|
| Unclassified | 18 | 16 | 88.9% |
| AGN/QSO | 110 | 102 | 92.7% |
| Galaxy/ELG | 33 | 32 | 97.0% |
| ULIRG | 4 | 4 | 100.0% |
| Composite | 11 | 10 | 90.9% |

## 4. Model A (Optical Classification) Output

| Class | Mean Probability |
|-------|-----------------|
| Galaxy | 2.6% |
| QSO | 5.5% |
| Star | 91.9% |

## 5. Data Status

- Radio (VLASS) FITS: 51/71 components downloaded
- Optical (PanSTARRS) FITS: 71/71 components with 5-band grizy
- WISE magnitudes: real CatWISE values for ~87 sources, SDSSxWISE median placeholder for rest
- Norris2006 crossmatch: 45/71 matched at <10 arcsec

## 6. Comparison with Original Paper

| Aspect | Original (Lou 2023) | This Validation |
|--------|---------------------|-----------------|
| Test set | RGZ + Norris06 (full) | CDFS Norris06 subset |
| Samples | ~1343 | 194 |
| Accuracy | ~93% (reported) | 93.3% |
| Data | Original FITS (94GB) | Public CADC + PanSTARRS |
| WISE mags | CatWISE full | Partial CatWISE + placeholder |