# SparseR-CNN (PyTorch Implementation)

这是一个PyTorch实现的SparseR-CNN目标检测模型，支持Detectron2集成和轻量版（lite）两种运行模式。

## 特性

- **Detectron2集成版**：贴近原版SparseR-CNN，支持完整训练、评估和推理流程。
- **轻量版（lite）**：最小依赖演示版本，适合快速测试和学习。
- 支持单类和多类目标检测。
- 包含数据合成脚本，用于生成训练数据。

## 安装

### 环境要求

- Python 3.7+
- PyTorch (根据你的CUDA版本安装，参考[PyTorch官网](https://pytorch.org/get-started/locally/))

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- torch
- torchvision
- Pillow
- opencv-python
- numpy
- detectron2
- scipy

对于Detectron2版本，还需要安装Detectron2及其依赖（fvcore, iopath, yacs, termcolor, pycocotools）。

## 使用

### 数据准备

使用数据合成脚本生成训练数据：

```powershell
python data_synth.py --make-data --out data --num-train 2000 --num-val 200 --img-size 800 --max-rects 5
```

### Detectron2版本

#### 训练

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

#### 评估

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

#### 推理

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

### 轻量版（lite）

#### 推理
##### 使用默认权重

```powershell
python infer.py `
  --config config_lite.yaml `
  --image sample.jpg `
  --output outputs/infer_vis_lite.jpg `
  --score-thr 0.6
```

##### 指定加载权重
```powershell
python infer.py --config config_lite.yaml --image sample.jpg --output outputs/infer_vis_lite.jpg --score-thr 0.6 --model-weights outputs/model_final.pth
```

#### 训练

```powershell
python train_sparse_rcnn.py `
  --data-root data `
  --output-dir outputs `
  --num-classes 1 `
  --num-proposals 50 `
  --max-iter 1000
```

## 项目结构

- `sparsercnn_lite/` - 轻量版模型实现
- `data_synth.py` - 数据合成脚本
- `train_sparse_d2.py` - Detectron2版本训练脚本
- `infer_sparse_d2.py` - Detectron2版本推理脚本
- `train_sparse_rcnn.py` - 轻量版训练脚本
- `infer.py` - 轻量版推理脚本
- `config_lite.yaml` - 轻量版配置文件
- `requirements.txt` - 依赖列表

## 常见问题

- 确保在安装有Detectron2的Python环境中运行Detectron2版本。
- 单类检测时使用`--num-classes 1`，数据类别ID从0开始。
- 若遇到设备不一致错误，确保模型和数据在同一设备上。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。