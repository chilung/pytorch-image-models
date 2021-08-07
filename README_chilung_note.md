
### Vision Trenaformer
1. How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers
2. When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations

### Train Command
./distributed_train.sh 4 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC/ --model vit_tiny_r_s16_p8_224

### Inference Command
python inference.py ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --model vit_tiny_r_s16_p8_224 --pretrained
python inference.py ../Dataset/imagenet/ILSVRC/Data/CLS-LOC --model vit_tiny_r_s16_p8_224 --pretrained --num-gpu 4 --workers 8

# Establish Vision Transformer Baseline
## Ref 1
Training data-efficient image transformers & distillation through attention

Top-1 Acc.
```
ViT-B/16: ImageNet top-1 77.9%
ViT-B/32: ImageNet top-1 73.4%
ViT-L/16: ImageNet top-1 76.5%
ViT-L/32: ImageNet top-1 71.2%
```
Train Hyper-Parameter for ViT-B
```
Epochs 300, Batch size 64, Optimizer AdamW, learning rate 0.003, Learning rate decay cosine, Weight decay 0.3, Warmup epochs 5, Label smoothing 0.1, Dropout 0.1, Stoch. Depth x, Repeated Aug x, Gradient Clip. x, Rand Augment x, Mixup prob. x, Cutmix prob. x, Erasing prob. x
./distributed_train.sh 4 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC/ --model vit_base_patch16_224 --epochs 300 --batch-size 64 --opt 'AdamW' --lr 0.003 --sched 'cosine' --weight-decay 0.3 --warmup-epochs 5 --smoothing 0.1 --drop 0.1 --workers 4
```
