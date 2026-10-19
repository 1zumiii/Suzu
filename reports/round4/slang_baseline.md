# 阶段 A：网络用语压下去率基线

日期 2026-10-19。脚本 tools/slang_eval.py；词表、评测用例和词分半文件放在私有数据目录，不入库（本报告只含汇总数字）。

口径：纯模型（无词频融合），候选分为各字对数概率之和。压下去率 = 词库首位的网络用语里，模型在不足一半的上文下把它排第一的比例；抬升数 = 词库非首位的网络用语里，模型在至少一半的上文下排第一的个数；分差 = 网络用语分减最强竞争词分的中位数（词 × 上文）。

复现结论：8 种通用上文下 Suzu-2 为 39.8%、Suzu-2.1 为 24.2%，抬升数 23 与 47，分差中位数 -0.56 与 +0.75，与 Mac 实测（40%、24%；23、47；-0.57、+0.74）一致，偏差在 3 个百分点内。

评测词按词分半：对整张词表（1258 个词）按类别分层，类别内按 md5 排序后交替分到训练种子半（631）与评测半（627）；428 个有竞争词的评测词中，评测半 207 个、种子半 221 个。评测半的词不得进入任何生成提示与训练语料。18 种上文 = 8 种通用上文加 10 种聊天、弹幕、评论风格上文。


## 8 contexts / all

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 289 | 115 | 39.8% | 139 | 23 | -0.56 |
| Suzu-2.1 | 289 | 70 | 24.2% | 139 | 47 | +0.75 |

## 8 contexts / eval half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 140 | 55 | 39.3% | 67 | 14 | -0.45 |
| Suzu-2.1 | 140 | 36 | 25.7% | 67 | 22 | +0.88 |

## 8 contexts / seed half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 149 | 60 | 40.3% | 72 | 9 | -0.62 |
| Suzu-2.1 | 149 | 34 | 22.8% | 72 | 25 | +0.66 |

## 18 contexts / all

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 289 | 122 | 42.2% | 139 | 25 | -0.50 |
| Suzu-2.1 | 289 | 71 | 24.6% | 139 | 44 | +0.86 |

## 18 contexts / eval half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 140 | 59 | 42.1% | 67 | 14 | -0.44 |
| Suzu-2.1 | 140 | 36 | 25.7% | 67 | 23 | +0.92 |

## 18 contexts / seed half

| model | lead terms | suppressed | rate | non-lead | lifted | gap median |
|---|---|---|---|---|---|---|
| Suzu-2 | 149 | 63 | 42.3% | 72 | 11 | -0.58 |
| Suzu-2.1 | 149 | 35 | 23.5% | 72 | 21 | +0.80 |
