# 🏆 AzbukaBoard — Cyrillic Handwriting Leaderboard

Метрики:
- **CER ↓** — Character Error Rate (меньше = лучше)
- **WER ↓** — Word Error Rate (меньше = лучше)
- **ACC ↑** — Accuracy (больше = лучше)

## Dataset: orig_cyrillic

| Avg Rank | Model | CER ↓ | WER ↓ | ACC ↑ | Trained on |
|------|--------|--------|--------|--------|--------|
| 1.0 | manuscript/trba_base_g1 | 0.0640 | 0.2928 | 0.6839 | Yes |
| 2.0 | manuscript/trba_lite_g1 | 0.0695 | 0.2974 | 0.6801 | Yes |
| 3.0 | easyocr/cyrillic_g1 | 0.7956 | 1.1551 | 0.0147 | No |
| 4.0 | easyocr/cyrillic_g2 | 1.9392 | 2.0337 | 0.0076 | No |

---

## Dataset: school_notebooks_RU

| Avg Rank | Model | CER ↓ | WER ↓ | ACC ↑ | Trained on |
|------|--------|--------|--------|--------|--------|
| 1.3 | manuscript/trba_lite_g1 | 0.0512 | 0.2071 | 0.8017 | Yes |
| 1.6 | manuscript/trba_base_g1 | 0.0508 | 0.2091 | 0.7985 | Yes |
| 3.0 | easyocr/cyrillic_g1 | 2.8655 | 1.7300 | 0.0198 | No |
| 4.0 | easyocr/cyrillic_g2 | 3.0184 | 2.1341 | 0.0177 | No |

---

## Dataset: DonkeySmallOCR

| Avg Rank | Model | CER ↓ | WER ↓ | ACC ↑ | Trained on |
|------|--------|--------|--------|--------|--------|
| 1.0 | manuscript/trba_base_g1 | 0.1017 | 0.4161 | 0.6169 | No |
| 2.6 | manuscript/trba_lite_g1 | 0.1155 | 0.4434 | 0.5812 | No |
| 3.0 | easyocr/cyrillic_g1 | 0.4816 | 0.5733 | 0.6113 | No |
| 4.0 | easyocr/cyrillic_g2 | 2.1286 | 1.2016 | 0.6446 | No |

---

## Dataset: YeniseiGovReports-HWR

| Avg Rank | Model                   | CER ↓   | WER ↓   | ACC ↑  | Trained on |
|----------|--------------------------|---------|---------|--------|------------|
| 1.0      | manuscript/trba_base_g1 | 0.0284  | 0.1009  | 0.9054 | Yes        |
| 2.0      | manuscript/trba_lite_g1 | 0.0359  | 0.1306  | 0.8756 | Yes        |
| 3.33     | easyocr/cyrillic_g1     | 3.2052  | 1.4752  | 0.0233 | No         |
| 3.67     | easyocr/cyrillic_g2     | 4.1480  | 2.1698  | 0.0441 | No         |

---

## Dataset: YeniseiGovReports-PRT

| Avg Rank | Model                   | CER ↓   | WER ↓   | ACC ↑  | Trained on |
|----------|--------------------------|---------|---------|--------|------------|
| 1.0      | manuscript/trba_base_g1 | 0.0111  | 0.0392  | 0.9649 | Yes        |
| 2.0      | manuscript/trba_lite_g1 | 0.0157  | 0.0557  | 0.9500 | Yes        |
| 3.33     | easyocr/cyrillic_g1     | 1.8229  | 1.0918  | 0.3274 | No         |
| 3.67     | easyocr/cyrillic_g2     | 4.0555  | 1.2718  | 0.5085 | No         |

---