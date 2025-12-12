# 运行指南（区分 Lite 与 Detectron2 版本）

## Lite 版本

- 训练：
```powershell
python lite/train.py `
  --data-root data `
  --output-dir outputs `
  --num-classes 1 `
  --num-proposals 50 `
  --max-iter 1000
```

- 推理：
```powershell
python lite/infer.py `
  --image sample.jpg `
  --output outputs/infer_vis_lite.jpg `
  --score-thr 0.6 `
  --weights outputs/model_final.pth `
  --num-classes 1
```

## Detectron2 版本

- 训练：
```powershell
python d2/train.py `
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

- 推理：
```powershell
python d2/infer.py `
  --config-file projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml `
  --image sample.jpg `
  --output outputs/infer_vis_d2.jpg `
  --weights outputs/model_final.pth `
  --num-classes 1 `
  --num-proposals 50 `
  --score-thr 0.6
```

## 注意
- Lite 与 Detectron2 版本互不依赖，分别位于 `lite/` 与 `d2/` 文件夹。
- 训练得到的权重默认保存在 `outputs/model_final.pth`；Lite 推理脚本已适配 Detectron2 格式权重。
- 保持 `--num-classes` 与训练一致，避免权重加载时维度不匹配。
