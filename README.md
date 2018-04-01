<p align="center"><img width="40%" src="PNG/logo3.PNG" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of [Text2Colors](아카이브주소). Text2Colors is capable of producing plausible colors (or color palette) given variable length of text input, and colorize a grayscale image based on the colors.

<p align="center"><img width="100%" src="PNG/main.PNG" /></p>

&nbsp;

## Paper
[Text2Colors: Guiding Image Colorization through Text-Driven Palette Generation](아카이브주소) <br/>
[Wonwoong Cho*<sup>1</sup>](https://github.com/wonwoongJo), [Hyojin Bahng*<sup>1</sup>](https://github.com/hjbahng), [David K. Park*<sup>1</sup>](https://github.com/heykeetae), [Seungjoo Yoo*<sup>1</sup>](https://github.com/sjooyoo), Ziming Wu<sup>2</sup>, [Xiaojuan Ma<sup>2</sup>](https://www.cse.ust.hk/~mxj/), and [Jaegul Choo<sup>1</sup>](https://sites.google.com/site/jaegulchoo/)<br/>
***These authors contributed equally and are presented in random order.**<br/>

&nbsp;
&nbsp;

## Model Description
### Text-to-Palette Generation Networks (TPN) and Palette-Based Colorization Networks (PCN)
<p align="center"><img width="100%" src="PNG/model1.PNG" /></p>
Overview of our Text2Colors architecture. During training, generator <b>G<sub>0</sub></b> learns to produce a color palette (<b>y hat</b>) given a set of conditional variables (<b>c hat</b>) processed from input text <b>x</b> = {x<sub>1</sub>, ···,  x<sub>T</sub>}. Generator <b>G<sub>1</sub></b> learns to predict a colorized output of a grayscale image (<b>L</b>) given a palette (<b>p</b>) extracted from the ground truth image. At test time, the trained generators <b>G<sub>0</sub></b> and <b>G<sub>1</sub></b> are used to produce a color palette from given text, and then colorize a grayscale image reflecting the generated palette.


<p align="center"><img width="100%" src="PNG/model2.PNG" /></p>


The model architecture of a generator <b>G<sub>0</sub></b> that produces the t-th color in the palette given an input text <b>x</b> = {x<sub>1</sub>, ···,  x<sub>T</sub>}. Note that randomness is added to each hidden state vector h in the sequence before it is passed to the generator

&nbsp;
&nbsp;

## Palette-and-Text (PAT) dataset

We open our manually curated dataset named Palette-and-Text(PAT). PAT contains 10,183 text and five-color palette pairs, where the set of five colors in a palette is associated with its corresponding text description as shown in Figs. 2(b)-(d). The text description is made up of 4,312 unique words. The words vary with respect to their relationships with colors; some words are direct color words (e.g. pink, blue, etc.) while others evoke a particular set of colors (e.g. autumn or vibrant).
<p align="center"><img width="100%" src="PNG/pat.PNG" /></p>
(a) the number of data items with respect to their text lengths. On the right are examples that show diverse textpalette pairs in PAT. Those text descriptions matching with their palettes include (b) direct color names, (c) texts with a relatively low level of semantic relations to colors, (d) those with a high-level semantic context. </br>

&nbsp;


## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.1](http://pytorch.org/)
* [Scikit-Image 0.13.1](http://scikit-image.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/awesome-davian/Text2Colors.git
$ cd Text2Colors/
```

#### 2. Dataset
##### (i) unzip the Palette-and-Text (PAT) dataset
```bash
$ bash download.sh
```

##### (ii) RaFD dataset
Because <b>RaFD</b> is not a public dataset, you must first request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). Then, you need to create the folder structure as decribed [here.](https://github.com/yunjey/StarGAN/blob/master/png/RaFD.md)

#### 3. Train StarGAN
##### (i) Training with CelebA

```bash
$ python main.py --mode='train' --dataset='CelebA' --c_dim=5 --image_size=128 \
                 --sample_path='stargan_celebA/samples' --log_path='stargan_celebA/logs' \
                 --model_save_path='stargan_celebA/models' --result_path='stargan_celebA/results'
```
##### (ii) Training with RaFD

```bash
$ python main.py --mode='train' --dataset='RaFD' --c_dim=8 --image_size=128 \
                 --num_epochs=200 --num_epochs_decay=100 --sample_step=200 --model_save_step=200 \
                 --sample_path='stargan_rafd/samples' --log_path='stargan_rafd/logs' \
                 --model_save_path='stargan_rafd/models' --result_path='stargan_rafd/results'
```

##### (iii) Training with CelebA+RaFD

```bash
$ python main.py --mode='train' --dataset='Both' --image_size=256 --num_iters=200000 --num_iters_decay=100000 \
                 --sample_path='stargan_both/samples' --log_path='stargan_both/logs' \
                 --model_save_path='stargan_both/models' --result_path='stargan_both/results'
```

#### 4. Test StarGAN
##### (i) Facial attribute transfer on CelebA
```bash
$ python main.py --mode='test' --dataset='CelebA' --c_dim=5 --image_size=128 --test_model='20_1000' \
                 --sample_path='stargan_celebA/samples' --log_path='stargan_celebA/logs' \
                 --model_save_path='stargan_celebA/models' --result_path='stargan_celebA/results'
```

##### (ii) Facial expression synthesis on RaFD
```bash
$ python main.py --mode='test' --dataset='RaFD' --c_dim=8 --image_size=128 \
                 --test_model='200_200' --rafd_image_path='data/RaFD/test' \
                 --sample_path='stargan_rafd/samples' --log_path='stargan_rafd/logs' \
                 --model_save_path='stargan_rafd/models' --result_path='stargan_rafd/results'
```

##### (iii) Facial expression synthesis on CelebA
```bash
$ python main.py --mode='test' --dataset='Both' --image_size=256 --test_model='200000' \
                 --sample_path='stargan_both/samples' --log_path='stargan_both/logs' \
                 --model_save_path='stargan_both/models' --result_path='stargan_both/results'
```

&nbsp;

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1711.09020).
```
@article{choi2017stargan,
 title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},    
 author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
 journal= {arXiv preprint arXiv:1711.09020},
 Year = {2017}
}
```
&nbsp;

## Acknowledgement
This work was mainly done while the first author did a research internship at <b>Clova AI Research, NAVER (CLAIR)</b>. We also thank all the researchers at CLAIR, especially Donghyun Kwak, for insightful discussions.


**For the use of PAT dataset for your research, please cite our [paper](아카이브주소).**
```
아카이브레퍼런스
```
&nbsp;

