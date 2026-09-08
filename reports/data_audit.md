# InsPLAD 本地数据审计

原始数据不修改。默认隔离 46 个歧义文件名对应的 92 个训练图像记录和 339 个框。

| 类别 | 原始训练框 | 清理后训练框 | 验证框 |
|---|---:|---:|---:|
| yoke | 1343 | 1321 | 318 |
| yoke suspension | 5270 | 5210 | 1250 |
| spacer | 72 | 68 | 22 |
| stockbridge damper | 5699 | 5608 | 1254 |
| lightning rod shackle | 170 | 164 | 25 |
| lightning rod suspension | 618 | 606 | 92 |
| polymer insulator | 2389 | 2350 | 855 |
| glass insulator | 2015 | 2015 | 963 |
| tower id plate | 198 | 196 | 44 |
| vari-grip | 846 | 827 | 162 |
| polymer insulator lower shackle | 1460 | 1432 | 382 |
| polymer insulator upper shackle | 1315 | 1301 | 377 |
| polymer insulator tower shackle | 47 | 42 | 10 |
| glass insulator big shackle | 110 | 110 | 149 |
| glass insulator small shackle | 128 | 128 | 135 |
| glass insulator tower shackle | 98 | 98 | 97 |
| spiral damper | 831 | 805 | 189 |
| sphere | 26 | 15 | 0 |

- 完成全部图像文件存在性、图像头尺寸/结构、框有限性和边界检查。
- 训练/验证文件名无交集；文件名前缀也无交集，但前缀不等同于已验证的杆塔身份，未据此声称消除所有泄漏。
- 验证集 sphere 无标注；COCO AP 不包含该类别，逐类结果为空。
- 这里没有独立 test split。
- 默认排除协议会改变训练集，全部对照需采用同一协议。

详细 JSON：`../data/insplad/audit.json`；冲突清单：`../data/insplad/train_conflicts.json`。
