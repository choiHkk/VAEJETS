## Introduction
1. FastSpeech2, HiFi-GAN 오픈 소스를 활용하여 VAEJETS(End-To-End)를 간단 구현하고 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
3. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
4. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
5. preprocessing 단계에서는 학습에 필요한 transcript와 stats 정도만 추출하는 과정만 포함되어 있습니다.
6. 그 외의 다른 preprocessing 과정은 필요하지 않습니다.
7. 본 레포지토리는 JETS 모델에 VAE를 접목하며 개인적인 실험을 목적으로 만들어졌음을 알려드립니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/VAEJETS/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/VAEJETS/data/dataset`

## Docker build
1. `cd /path/to/the/VAEJETS`
2. `docker build --tag VAEJETS:latest .`

## Training
1. `nvidia-docker run -it --name 'VAEJETS' -v /path/to/VAEJETS:/home/work/VAEJETS --ipc=host --privileged VAEJETS:latest`
2. `cd /home/work/VAEJETS`
5. `ln -s /home/work/VAEJETS/data/dataset/kss`
6. `python preprocess.py ./config/kss/preprocess.yaml`
7. `python train.py -p ./config/kss/preprocess.yaml -s ./config/kss/model.yaml -g ./config/kss/config_v1.json -t ./config/kss/train.yaml`
8. arguments
  * -p : preprocess config path
  * -s : synthesizer config path
  * -g : generator config path
  * -t : train config path
9. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses


## Tensorboard Stats


## Reference
1. [VAENAR-TTS: Variational Auto-Encoder based Non-AutoRegressive Text-to-Speech Synthesis](https://arxiv.org/pdf/2107.03298v1.pdf)
2. [Comprehensive-E2E-TTS](https://github.com/keonlee9420/Comprehensive-E2E-TTS)
3. [VITS](https://github.com/jaywalnut310/vits)
