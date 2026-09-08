# RF-DETR / InsPLAD 公开发布设计

## 目标

将 `/home/wzs/dataset/wxh` 中已经验证的 RF-DETR / InsPLAD 检测工程发布到公开仓库 `zongshun21/-RF-DETR-`，使其他研究者能够理解数据处理方式、重建环境、复现 640/960 训练、加载权重并运行 COCO 评估。

## 发布边界

Git 仓库包含：

- 项目训练、评估、预测及数据准备脚本；
- RF-DETR 1.10.0 官方源码发行包及其 Apache-2.0 许可证和来源校验记录；
- 512、640、960 和 smoke 配置；
- InsPLAD 训练集和验证集 COCO JSON 标注；
- 46 个冲突文件名的排除清单、数据审计结果及少量冲突预览；
- 完整使用说明、实现说明、训练说明和发布权重清单；
- 固定依赖及自动化检查。

Git 仓库不包含：

- InsPLAD 原始 JPG 图像；
- Python 虚拟环境、下载缓存和运行日志；
- 24 GB 的全部训练输出及每 10 epoch 的完整 Lightning 检查点；
- 官方 COCO 预训练权重副本。

## 权重发布

正式训练的两个最佳推理权重作为 GitHub Release 附件发布：

- 640：`outputs/small_640_bs16/checkpoint_best_total.pth`；
- 960：`outputs/small_960_20260907_045435/checkpoint_best_total.pth`。

Release 同时附带对应的 `training_config.json`、`run_manifest.json`、`metrics.csv`、逐类指标（若可由已有输出获得）和 SHA256 清单。轻量推理权重约 123–126 MB，超过 GitHub 普通 Git 单文件限制，因此不放入 Git 历史。权重必须能够脱离原训练目录加载，并保留实际分辨率和位置编码元数据。

## 数据发布

公开原始 `instances_train.json` 和 `instances_val.json`，并发布脚本生成的清理协议。README 明确说明：训练 JSON 有 46 个文件名分别映射到两套不同 image_id 和标注；默认训练会隔离相关 92 个图像记录，得到 7,889 个训练记录和 22,296 个框。原始标注与清理记录同时保留，不静默改写原标注。

README 不提供原图镜像，要求用户从 InsPLAD 官方来源取得图片后按指定目录放置，并运行 `prepare_data.py --verify-images`。发布前检查标注不含本机绝对路径、隐私字段或凭据。

## 文档结构

根 README 提供项目定位、结果边界、目录结构、安装、数据准备、单卡/双卡训练、不同分辨率配置、断点续训、评估、预测、常见错误和引用。单独文档说明：

- `docs/IMPLEMENTATION.md`：网络入口、适配层及 RF-DETR 1.10 兼容修复；
- `docs/DATASET.md`：数据目录、标注统计、冲突处理和可复现数据协议；
- `docs/TRAINING.md`：训练参数、显存说明、输出文件和实验管理；
- `docs/WEIGHTS.md`：Release 文件、校验值、模型加载与适用范围。

所有本地链接和命令以仓库根目录为基准，不依赖 `/home/wzs/dataset` 绝对路径，默认数据目录允许通过命令行覆盖。

## 发布验证

发布前执行：

1. Python 编译、单元测试和依赖一致性检查；
2. 标注结构、图片文件名及敏感信息扫描；
3. 640/960 权重 SHA256 和独立加载检查；
4. Git 忽略规则和大文件扫描，确保普通 Git 对象不超过 GitHub 限制；
5. 在干净临时检出中检查文档链接、配置 dry-run 和测试；
6. 推送 `main` 后核对远端提交；
7. 创建 GitHub Release、上传权重和元数据，并核对附件大小与下载地址。

## 失败处理

Git 推送或 Release 上传若缺少 GitHub 身份验证，不改写仓库历史，也不把令牌写入文件或命令输出。已完成的本地提交保持可审阅，待用户完成认证后继续。Release 附件若上传中断，重新上传对应附件并按 SHA256 核对，不重复创建含混版本。
