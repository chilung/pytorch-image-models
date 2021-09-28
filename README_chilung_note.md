# Install local timm version
pip install -e .
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
Epochs 300, Batch size 4096, Optimizer AdamW, learning rate 0.003, Learning rate decay cosine, Weight decay 0.3, Warmup epochs 3.4, Label smoothing x, Dropout 0.1, Stoch. Depth x, Repeated Aug x, Gradient Clip. V, Rand Augment x, Mixup prob. x, Cutmix prob. x, Erasing prob. x
```
## Ref 2
DeepViT Towards Deeper Vision Transformer

Top-1 Acc.
```
ViT-B/12: ImageNet top-1 77.6%
ViT-B/16: ImageNet top-1 78.9%
ViT-B/24: ImageNet top-1 79.4%
ViT-B/32: ImageNet top-1 79.3%
```
Train Hyper-Parameter for ViT
```
pochs 300, Batch size 256, Optimizer AdamW, learning rate 0.0005, Learning rate decay cosine, Warmup epochs 3, Rand Augment V, Mixup prob. V, 12 heads
rand augment and mixup: ref: Resnest: Split-attention networks
```
## My Experiments
### vit_base_patch16_224
Epochs 300, Batch size 64, Optimizer AdamW, learning rate 0.003, Learning rate decay cosine, Weight decay 0.3, Warmup epochs 5, Label smoothing 0.1, Dropout 0.1, Stoch. Depth x, Repeated Aug x, Gradient Clip. x, Rand Augment x, Mixup prob. x, Cutmix prob. x, Erasing prob. x
```
./distributed_train.sh 4 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC/ --model vit_base_patch16_224 --epochs 300 --batch-size 64 --opt 'AdamW' --lr 0.003 --sched 'cosine' --weight-decay 0.3 --warmup-epochs 5 --smoothing 0.1 --drop 0.1 --workers 4
```
result file:

# The feature map in ViT tends to become identical in deeper layers
## Evidence
Check the similarity of feature maps between two consecutive layers while in training phase.
## model class DiverViT for this experiment
### source code modification
1. source code path: ../timm/models/divervit.py
2. supported configuration: diver_vit_base_patch16_224
### training command
./distributed_train.sh 4 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC/ --model divervit_base_patch16_224 --epochs 300 --batch-size 4 --opt 'AdamW' --lr 0.003 --sched 'cosine' --weight-decay 0.3 --warmup-epochs 5 --smoothing 0.1 --drop 0.1 --workers 1

# Add LMDB support

