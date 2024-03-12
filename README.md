<!-- <div align="center">    -->
  
# Official implementation of the WACV-2024 paper: [Frequency Attention Knowledge Distillation](https://openaccess.thecvf.com/content/WACV2024/papers/Pham_Frequency_Attention_for_Knowledge_Distillation_WACV_2024_paper.pdf)
</div>


## Installation

Prerequires:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0
- other packages like wandb, tensorboard, etc.

Install the package:

```
pip3 install -r requirements.txt
```

## Dataset
For ImageNet dataset, download data to train and val folder

## Training

### Training on CIFAR-100

<!-- - Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`. -->

  ```bash
  python3 tools/train.py --cfg configs/cifar100/FAM_KD/res56_res20.yaml

  # you can also change settings at command line
  python3 tools/train.py --cfg configs/cifar100/FAM_KD/res56_res20.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`
  ```bash
  # train ResNet18 with FAM-KD
  python3 tools/train.py --cfg configs/imagenet/r34_r18/fam_kd.yaml
  ```

<!-- ## Evaluation
    
      ```bash
      # evaluate the trained model
      python3 tools/eval.py --cfg configs/cifar100/FAM_KD/res56_res20.yaml
      ``` -->

### Note
- Current pytorch version does not support multi-gpu for training with `torch.fft` module. Therefor we should specific GPU number when training (e.g, CUDA_VISIBLE_DEVICES=0). We will update the code to support multi-gpu training soon.
## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{pham2024frequency,
  title={Frequency Attention for Knowledge Distillation},
  author={Pham, Cuong and Nguyen, Van-Anh and Le, Trung and Phung, Dinh and Carneiro, Gustavo and Do, Thanh-Toan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2277--2286},
  year={2024}
}
```
## Related resources

## Acknowledgments

This work based on resources below. A huge thank you to the original authors and community for their contributions to the open-source community.

- [Decoupled Knowledge Distillation](https://github.com/megvii-research/mdistiller.git)
- [Distilling Knowledge via Knowledge Review](https://github.com/dvlab-research/ReviewKD)
- [Stand-Alone Self-Attention in Vision Models](https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py)
- [Fourier Neural Operator](https://github.com/neuraloperator/neuraloperator.git)

