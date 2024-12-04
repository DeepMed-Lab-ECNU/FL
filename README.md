# Multi-stage Multi-granularity Focus-tuned Learning Paradigm for Medical HSI Segmentation(MICCAI 2024)

by Haichuan Dong, Runjie Zhou, Boxiang Yun, Huihui Zhou, Benyan Zhang, Qingli Li, and Yan Wang*

## Introduction
(MICCAI 2024) Official code for "[Multi-stage Multi-granularity Focus-tuned Learning Paradigm for Medical HSI Segmentation](https://papers.miccai.org/miccai-2024/paper/0621_paper.pdf)".

### main-stream

![TrainPic](https://github.com/user-attachments/assets/15156111-ccd7-48c9-8441-ac85b6be63b8)

### pretraining-stage

![PretrainPic](https://github.com/user-attachments/assets/cfd0b2cf-a50e-4aa4-8411-19ba49dcac8b)



## Requirements
Experiments are conducted by using PyTorch 1.11.0 on an NVIDIA GeForce RTX 4090 GPU.



## Usage

The official dataset can be found at [MDC](http://bio-hsi.ecnu.edu.cn/). 


To pretrain a model,
```
python QSQL.py \
  -a resnet34 \
  --lr 0.03 \
  --batch-size 32
```


To train a model,
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 \
Train_FL.py  -r ../MDC_dataset \
-b 8 \
-spe_c 60 \
-hw 256 256 \
-name 
```

To test a model,
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 \
Evaluate_FL.py  -r ../MDC_dataset \
-b 8 \
-spe_c 60 \
-hw 256 256 \
-name DFS3R_MHSI_Fintune_MDC \
--pretrained_model ./bileseg-checkpoint/ModelName
```
## Citation
If you find these projects useful, please consider citing:

```bibtex
@inproceedings{dong2024multi,
  title={Multi-stage Multi-granularity Focus-Tuned Learning Paradigm for Medical HSI Segmentation},
  author={Dong, Haichuan and Zhou, Runjie and Yun, Boxiang and Zhou, Huihui and Zhang, Benyan and Li, Qingli and Wang, Yan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={456--466},
  year={2024},
  organization={Springer}
}
```
## Acknowledgements
Some modules in our code were inspired by [Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). We appreciate the effort of these authors to provide open-source code for the community. Hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at '10212140417@stu.ecnu.edu.cn'
