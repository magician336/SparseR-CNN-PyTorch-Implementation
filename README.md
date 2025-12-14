# SparseR-CNN (PyTorch Implementation)

本仓库提供 SparseR-CNN 的两种使用方式：
- 轻量版（Lite）：纯 PyTorch 运行与训练，最小依赖、便于快速上手。
- Detectron2 版（D2）：贴近原版的完整流程与配置，适合更丰富的功能与扩展。

本文档基于当前文件结构给出两种方式的独立运行指南与常见问题说明，不改变现有脚本入口。

## 环境准备

- Python 3.7+
- PyTorch（建议 GPU，加速训练与推理）
- 依赖安装：

```powershell
pip install -r requirements.txt
```

主要依赖：`torch`, `torchvision`, `Pillow`, `opencv-python`, `numpy`, `scipy`。
如需使用 Detectron2 版，请另外安装 `detectron2` 及其依赖。

---

## Lite 版本（纯 PyTorch）

Lite 版本使用自定义的 `SparseRCNNLite` 与合成数据集进行训练与推理，不依赖 Detectron2。

### 数据合成

```powershell
python data_synth.py `
  --make-data `
  --out data `
  --num-train 2000 `
  --num-val 200 `
  --img-size 800 `
  --max-rects 5
```

生成的合成数据位于 `data/` 目录，供训练与测试使用。

### 训练（Lite）

- 纯 PyTorch 训练脚本：`lite/train_lite.py`

示例：
```powershell
python train_lite.py `
  --data-root data `
  --output-dir outputs `
  --epochs 5 `
  --batch-size 2 `
  --lr 5e-4 `
  --num-classes 1 `
  --num-proposals 50 `
  --num-workers 2
```

训练完成后，会在 `outputs/model_final.pth` 保存权重（字典包含键 `model`），与 Lite 推理脚本兼容。

### 推理（Lite）

- 根目录 Lite 推理入口：`infer_sparse_rcnn.py`

示例：
```powershell
python infer_lite.py `
  --image sample.jpg `
  --output outputs/infer_vis_lite.jpg `
  --score-thr 0.6 `
  --weights outputs/model_final.pth `
  --num-classes 1
```

常见问题：
- 若输出图像与输入一模一样，多为“检测结果为空”。可尝试：
  - 降低阈值：`--score-thr 0.1`
  - 使用合成数据中的图像或包含矩形的测试图像
  - 确认训练与推理的 `--num-classes` 一致
- 若加载权重报尺寸不匹配，确认训练时的类别数与推理一致。

---

## Detectron2 版本（D2）

D2 版提供完整的配置与训练流程，适合更丰富的功能与评估。确保在安装有 Detectron2 的环境中运行。

### 训练（D2）

```powershell
python train_sparse_d2.py `
  --config-file projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml `
  --data-root data `
  --output-dir outputs `
  --batch-size 2 `
  --lr 2.5e-4 `
  --max-iter 2000 `
  --num-workers 2 `
  --min-size-train 800 `
  --num-proposals 50 `
  --num-classes 1 `
  --eval-period 1000
```

### 评估（D2）

```powershell
python train_sparse_d2.py `
  --config-file projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml `
  --data-root data `
  --output-dir outputs `
  --num-classes 1 `
  --num-proposals 50 `
  --eval-only `
  --weights outputs/model_final.pth
```

### 推理（D2）

```powershell
python infer_sparse_d2.py `
  --config-file projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml `
  --image sample.jpg `
  --output outputs/infer_vis_d2.jpg `
  --weights outputs/model_final.pth `
  --num-classes 1 `
  --num-proposals 50 `
  --score-thr 0.6
```

---

## 脚本说明与权重兼容

- Lite 推理脚本已适配由训练脚本保存的 Detectron2 风格权重（包含 `model` 键），也可直接加载 `state_dict`。
- 使用 Lite 时，务必保持 `--num-classes` 与训练一致；否则会出现分类头维度不匹配。
- 若在 D2 训练过程中出现学习率里程碑（`SOLVER.STEPS`）与 `MAX_ITER` 冲突，请调整配置文件中的里程碑小于 `MAX_ITER`。

---

## 目录结构（关键项）

- `sparsercnn_lite/`：Lite 模型实现（核心模块）
- `data_synth.py`：合成数据生成脚本
- `train_lite.py`：纯 PyTorch 训练
- `infer_sparse_rcnn.py`：Lite 推理入口（根目录）
- `infer_sparse_d2.py`：Detectron2 推理入口（根目录）
- `train_sparse_d2.py`：Detectron2 训练入口（根目录）
- `config_lite.yaml`：Lite 默认配置
- `requirements.txt`：依赖列表

---

## 反馈与贡献

欢迎提交 Issue 或 Pull Request 改进本项目。若需进一步区分或精简脚本入口，我可以协助整理更一致的命令行参数与模块抽象。
