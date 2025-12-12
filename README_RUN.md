# SparseR-CNN 运行指南

本指南汇总项目的两种运行路径：
- Detectron2 项目版 Sparse R-CNN（训练/评估/推理，贴近原版）
- 轻量版（lite）（最小依赖演示）

所有命令以 Windows PowerShell 5.1 为例。

## 一、Detectron2 项目版 Sparse R-CNN

- 入口脚本：`personal/train_sparse_d2.py`、`personal/infer_sparse_d2.py`
- 配置文件：`projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml`（已为 4060 6GB 约束）
- 依赖：`torch`、`torchvision`、`fvcore`、`iopath`、`yacs`、`termcolor`、`pycocotools`、Detectron2

### 1) 数据准备（单类示例）
```powershell
python personal\data_synth.py --make-data --out personal\data --num-train 2000 --num-val 200 --img-size 800 --max-rects 5
```

### 2) 训练（含周期评估）
```powershell
python personal\train_sparse_d2.py `
  --config-file projects\SparseRCNN\configs\sparsercnn.res50.lite6g.yaml `
  --data-root personal\data `
  --output-dir personal\outputs `
  --batch-size 2 `
  --lr 2.5e-4 `
  --max-iter 2000 `
  --num-workers 2 `
  --min-size-train 800 `
  --num-proposals 50 `
  --num-classes 1 `
  --eval-period 1000
```
输出：`personal\outputs\model_final.pth` 等检查点。

### 3) 仅评估（eval-only）
```powershell
python personal\train_sparse_d2.py `
  --config-file projects\SparseRCNN\configs\sparsercnn.res50.lite6g.yaml `
  --data-root personal\data `
  --output-dir personal\outputs `
  --num-classes 1 `
  --num-proposals 50 `
  --eval-only `
  --weights personal\outputs\model_final.pth
```

### 4) 推理（项目版）
```powershell
python personal\infer_sparse_d2.py `
  --config-file projects\SparseRCNN\configs\sparsercnn.res50.lite6g.yaml `
  --image personal\sample.jpg `
  --output personal\outputs\infer_vis_d2.jpg `
  --weights personal\outputs\model_final.pth `
  --num-classes 1 `
  --num-proposals 50 `
  --score-thr 0.6
```

### 常见问题
- 使用源码目录导入 `detectron2` 会缺少编译扩展 `_C`，请在安装有 Detectron2 的 Python 环境中运行。
- PIL 插值常量已统一为 `Image.BILINEAR`，避免旧版常量缺失。
- 单类配置需 `--num-classes 1` 且数据 `category_id=0`（数据合成脚本已修复为零起始）。

## 二、轻量版（lite）

- 入口脚本：`personal/infer.py`、`personal/train_sparse_rcnn.py`
- 配置：`personal/config_lite.yaml`（自包含）
- 依赖：`torch`、`torchvision`、`opencv-python`、`Pillow`（可选 `scipy`）

### 1) 最小推理
未提供权重时会回退到 Detectron2 模型库（Faster R-CNN）进行演示：
```powershell
python personal\infer.py `
  --image personal\sample.jpg `
  --output personal\outputs\infer_vis_lite.jpg `
  --score-thr 0.6
```
使用本地配置：
```powershell
python personal\infer.py `
  --config personal\config_lite.yaml `
  --image personal\sample.jpg `
  --output personal\outputs\infer_vis_lite.jpg `
  --score-thr 0.6
```

### 2) 最小训练
```powershell
python personal\train_sparse_rcnn.py `
  --data-root personal\data `
  --output-dir personal\outputs `
  --num-classes 1 `
  --num-proposals 50 `
  --max-iter 1000
```
若出现设备不一致错误（CPU/GPU），请确保 `targets` 与模型在同一设备；脚本已处理在训练前移动到对应 device。

## 目录速览
- `personal/infer_sparse_d2.py`：Detectron2 项目版推理
- `personal/train_sparse_d2.py`：Detectron2 项目版训练/评估
- `projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml`：6GB 显存约束配置
- `personal/data_synth.py`：合成数据与 COCO 导出（零起始标签）
- `personal/infer.py`：轻量版推理（含模型库回退）
- `personal/train_sparse_rcnn.py`：轻量版训练
- `personal/config_lite.yaml`：轻量版配置（自包含）

如需我协助直接运行训练或评估，请告知使用的环境或让我执行上面的命令。