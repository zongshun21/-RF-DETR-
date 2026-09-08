# 基于 RF-DETR-S 的 InsPLAD 目标检测

本工程提供可复现的 RF-DETR-S / InsPLAD 检测基线，支持 640 和 960 输入、
COCO 整体与逐类评估、预测可视化、数据审计、单卡训练和双卡 DDP。工程固定
使用 RF-DETR 1.10.0 正式版源码，目前属于基线和后续创新研究框架，没有宣称
已经提出新网络结构。

## 已完成结果

两组模型均使用单张 RTX 4090 24 GB、BF16、EMA、seed 42 训练 150 epoch。
以下结果由本工程对最佳权重进行完整验证集独立 COCO 评估得到：

| 模型 | 输入 | 训练 batch | AP50:95 | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|---:|
| RF-DETR-S | 640 | 32 | 0.7432 | 0.9044 | 0.7623 | 0.8556 |
| RF-DETR-S | 960 | 16 | 0.7541 | 0.9147 | 0.7934 | 0.8633 |

这是单次训练的验证集结果，不是独立测试集结果，也不能代替多个随机种子的
统计检验。完整配置、训练指标历史和逐类 AP 位于 `release_metadata/`。
表格中的 batch 是已完成训练实际使用的历史值；当前 640 起始配置按你的要求
默认为 batch 16，适合先在 24 GB 单卡上测试显存。

## 安装

```bash
git clone https://github.com/zongshun21/-RF-DETR-.git
cd -RF-DETR-
bash scripts/setup.sh
source .venv/bin/activate
python -m pip check
```

下载 RF-DETR-S 官方 COCO 预训练权重：

```bash
mkdir -p weights
curl -fL -C - \
  https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth \
  -o weights/rf-detr-small.pth
echo 'fb37061c1af7bace359c91b723a8d5c1  weights/rf-detr-small.pth' | md5sum -c -
```

## 数据处理

仓库包含 `datasets/InsPLAD-det/annotations/` 下的 COCO 标注，不包含 JPG 原图。
从 [InsPLAD 官方项目](https://github.com/andreluizbvs/InsPLAD) 下载目标检测图像。
数据使用 CC BY-NC-SA 4.0，禁止商业使用。

原始目录结构：

```text
/path/to/InsPLAD-det/
├── annotations/instances_train.json
├── annotations/instances_val.json
├── train/*.jpg
└── val/*.jpg
```

执行检查和转换：

```bash
python prepare_data.py \
  --source /path/to/InsPLAD-det \
  --output data/insplad \
  --verify-images
```

脚本不会修改或复制原图，而是创建符号链接。原始训练 JSON 有 7,981 个图像记录、
22,635 个框，其中 46 个文件名分别关联两套不同 image_id 和框标注。默认协议会
隔离这 92 个歧义记录，最终使用 7,889 个训练记录、22,296 个框。完整冲突清单
在 `datasets/InsPLAD-det/audit/train_conflicts.json`。验证集没有 `sphere` 实例，
因此该类 AP 为空值，不记成 0。详见 `docs/DATASET.md`。

## 训练

先检查最终解析参数：

```bash
python train.py --config configs/small_640.yaml --dry-run
```

分别运行 640 和 960：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/small_640.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/small_960.yaml
```

终端会显示 tqdm 进度条。输出目录已存在时，程序会自动新建带时间戳的目录，
不会覆盖旧实验。也可以从命令行临时修改参数：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/small_960.yaml \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --epochs 150 \
  --output outputs/small_960_bs4
```

有效 batch = 每卡 batch × 梯度累积 × GPU 数。发生显存不足时降低实际 batch，
再相应提高梯度累积。双卡、续训和输出文件说明见 `docs/TRAINING.md`。

## 下载与评估训练权重

```bash
bash scripts/download_release_weights.sh
```

最佳权重以 GitHub 可接受的分块形式保存在 `model_weights/chunks/`。脚本将它们
重组到 `weights/releases/`，并自动进行 SHA256 校验。评估 640 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint weights/releases/rfdetr_s_insplad_640_best.pth \
  --dataset data/insplad \
  --output outputs/eval_640
```

输出包括 `metrics.json`、`per_class.csv` 和 `predictions.coco.json`。预测图片：

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --checkpoint weights/releases/rfdetr_s_insplad_960_best.pth \
  --image /path/to/image.jpg \
  --output outputs/prediction.jpg
```

网络入口、位置编码恢复、背景标签处理和评估预处理说明见
`docs/IMPLEMENTATION.md`；权重说明见 `docs/WEIGHTS.md`。
