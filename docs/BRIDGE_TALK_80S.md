# 模块 B 补充讲稿：双语查询通道（80 秒）

对应 `docs/HANDOFF.md` 模块 B（视觉表示、微调与双语查询），接在第 4 页“组合外观”之后。
作者：zimengzhu1225-creator ｜ 分支 `feat/zh-bridge` ｜ 完整报告 `docs/CROSS_LINGUAL_BRIDGE_REPORT.md`

## 英文口播稿（约 170 词，正常语速 80 秒）

Page 4 also carries the bilingual query problem. Our encoder is strong in English: on our frozen
protocol, the same Chinese descriptions score 0.842, while human-written English reaches 0.915.
A Chinese question therefore loses accuracy before retrieval even starts — not because the model
cannot see, but because the query never enters the space the image tower was trained in.

My contribution is a translation bridge. The Chinese query is translated into English, specialist
terminology is repaired against a short glossary, and the English channel is fused with the Chinese
one at a weight selected on the development split and then frozen.

That lifts Chinese nDCG@10 from 0.842 to 0.872, recovering about forty percent of the Chinese-English
gap. The ablation is the interesting part: swapping in a larger translation model changes nothing —
every point comes from terminology repair. The bottleneck is specialist vocabulary, not general
translation quality.

We also mapped where it breaks. On negation and surface-state conditions — “no painted decoration”,
“plain body” — the bridge hurts, because translation blurs those constraints. So this is a route for
specialist terms, not a general-purpose fix.

## 讲述提示（中文，不用念出来）

* 三个数字必须一起说：0.842 / 0.915 / 0.872。只说涨了多少会被问“跟谁的基线比”。
* **口径警告**：本页数字来自本工作的 60 图冻结评测集（`eval/visual_test`），与模块 B 页面上
  全库口径的 0.6385 / 0.7219 **不是同一套评测**。被问到时如实说“这是不同评测集上的结果，
  不能直接并排比较”，千万不要混着报。
* 不声称显著：p=0.134。正确说法是“方向在两个集合一致、幅度稳定”，不是“显著优于基线”。
* 最后那句“not a general-purpose fix”是主动示弱，评委反而会认为你懂边界，别删。

## 追问备用答案

| 会被问 | 回答 |
|---|---|
| 为什么不直接训练跨语言模型？ | 训练数据是馆藏英文著录，中文监督缺失；桥接不需要新训练数据，只改查询侧。 |
| 为什么不用更大的翻译模型？ | 测过 NLLB 与 opus 两档，单独换模型无收益（0.8518 vs 0.8543），收益全在术语纠错。 |
| 术语纠错怎么做？ | 译后按术语表补回被 MT 丢失的专业词（器型、纹样、窑口），不在原文上扩写，避免概念重复。 |
| 和组员 gloss 词表什么关系？ | 同一机制的两种实现（他 180 条静态词表，我整句神经翻译），所以叠加只有 +0.006。 |
| 为什么多条件子集上反而变差？ | 那批查询含否定条件，“无彩绘/素面”翻译后语义被冲淡；后续可让否定子句走不翻译通道。 |
