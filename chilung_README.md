
### Vision Trenaformer
1. How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers
2. When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations

### Train Command
./distributed_train.sh 4 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC/ --model vit_tiny_r_s16_p8_224

### Eval Command
./distributed_train.sh 1 ../Dataset/imagenet/ILSVRC/Data/CLS-LOC -c --model vit_tiny_r_s16_p8_224 --pretrained  --eval-only
