# SparseR-CNN Lite 版本运行指南

## 简介

SparseR-CNN Lite 是 PyTorch 实现的轻量版 SparseR-CNN 目标检测模型，专注于最小依赖和快速演示。不依赖 Detectron2，支持单类和多类目标检测。

## 环境要求

- Python 3.7+
- PyTorch (推荐 CUDA 版本以支持 GPU)
- 其他依赖：Pillow, OpenCV, NumPy, SciPy

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/magician336/SparseR-CNN-PyTorch-Implementation.git
   cd SparseR-CNN-PyTorch-Implementation
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

使用内置数据合成脚本生成训练数据：

```bash
python data_synth.py --make-data --out data --num-train 2000 --num-val 200 --img-size 800 --max-rects 5
```

- `--num-train`: 训练样本数量
- `--num-val`: 验证样本数量
- `--img-size`: 图像尺寸
- `--max-rects`: 每张图像最大矩形数量

这会在 data 目录下生成 NPZ 格式的数据文件。

## 训练

使用 lite 训练脚本：

```bash
python train_sparse_rcnn.py --data-root data --output-dir outputs --num-classes 1 --num-proposals 50 --max-iter 1000
```

参数说明：
- `--data-root`: 数据根目录 (默认: data)
- `--output-dir`: 输出目录 (默认: outputs)
- `--num-classes`: 类别数量 (不包括背景，默认: 1)
- `--num-proposals`: 提议数量 (默认: 50)
- `--max-iter`: 最大迭代次数 (默认: 1000)
- `--batch-size`: 批大小 (默认: 2)
- `--lr`: 学习率 (默认: 0.0005)

训练完成后，模型权重保存在 model_final.pth。

## 推理

使用 lite 推理脚本：

```bash
python infer_sparse_rcnn.py --image sample.jpg --output outputs/infer_vis_lite.jpg --score-thr 0.6 --weights outputs/model_final.pth --num-classes 1
```

参数说明：
- `--image`: 输入图像路径
- `--output`: 输出可视化图像路径
- `--score-thr`: 检测得分阈值 (默认: 0.5)
- `--weights`: 模型权重路径 (可选，默认使用随机权重)
- `--num-classes`: 类别数量 (默认: 1)
- `--device`: 设备 (默认: cuda 如果可用，否则 cpu)

输出图像会在检测框上绘制边界框和标签。

## 注意事项

1. **类别数量**: 训练和推理时 `--num-classes` 必须一致。单类检测设为 1，多类相应调整。

2. **权重格式**: 训练保存的权重为 Detectron2 格式，推理脚本会自动处理。

3. **图像格式**: 输入图像应为 RGB 格式的 JPG/PNG。

4. **性能**: Lite 版本在 CPU 上也可运行，但 GPU 推荐以提高速度。

5. **调试**: 如果推理无检测结果，尝试降低 `--score-thr` 或检查图像是否包含目标对象。

6. **数据**: 如果使用自定义数据，确保格式与合成数据一致 (NPZ 文件包含 'image' 和 'annotations')。

## 常见问题

- **训练时内存不足**: 降低 `--batch-size` 或 `--img-size`。
- **推理无结果**: 检查图像是否包含矩形，或降低阈值。
- **权重加载失败**: 确保权重文件路径正确，且与 `--num-classes` 匹配。

如有问题，请检查终端输出或提交 Issue。