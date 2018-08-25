<p align="center"><img width="40%" src="PNG/logo3.png" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of [Text2Colors](https://arxiv.org/abs/1804.04128). Text2Colors is capable of producing plausible colors (or color palette) given variable length of text input, and colorize a grayscale image based on the generated color palettes.

<p align="center"><img width="100%" src="PNG/main.PNG" /></p>

&nbsp;

## Paper
[Coloring with Words: Guiding Image Colorization through Text-based Palette Generation](https://arxiv.org/abs/1804.04128) <br/>
[Hyojin Bahng*<sup>1</sup>](https://github.com/hjbahng), [Seungjoo Yoo*<sup>1</sup>](https://github.com/sjooyoo), [Wonwoong Cho*<sup>1</sup>](https://github.com/wonwoongJo), [David Keetae Park<sup>1</sup>](https://github.com/heykeetae), [Ziming Wu<sup>2</sup>](https://github.com/jimmy-ng), [Xiaojuan Ma<sup>2</sup>](https://www.cse.ust.hk/~mxj/), and [Jaegul Choo<sup>1</sup>](https://sites.google.com/site/jaegulchoo/)<br/>
***These authors contributed equally.**<br/>
&nbsp;&nbsp; **<sup>1</sup>Korea University &nbsp;&nbsp;&nbsp; <sup>2</sup>Hong Kong University of Science and Technology**

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
Statistics and samples of PAT dataset: (a) the number of data items with respect to their text lengths. On the right are examples that show diverse textpalette pairs in PAT. Those text descriptions matching with their palettes include (b) direct color names, (c) texts with a relatively low level of semantic relations to colors, (d) those with a high-level semantic context. </br>
 </br>

**For the use of PAT dataset for your research, please cite our [paper](https://arxiv.org/abs/1804.04128).**

&nbsp;

## Results
<p align="center"><img width="100%" src="PNG/test_palettes.png" /></p>
<p align="center"><img width="100%" src="PNG/qualitative.png" /></p>

## Prerequisites
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 0.4.1](http://pytorch.org/)
* [Scikit-Image 0.13.1](http://scikit-image.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/awesome-davian/Text2Colors.git
$ cd Text2Colors/
```

#### 2. Dataset & Libraries install
```bash
$ bash install_pre.sh
```

#### 3. Train 
##### (i) Training Text-to-Palette Generation Networks (TPN) with PAT data

```bash
$ python main.py --mode train_TPN
```

##### (ii) Training Palette-Based Colorization Networks (PCN) with CUB-200-2011* data

```bash
$ python main.py --mode train_PCN
```
*Wah, Catherine, et al. "The caltech-ucsd birds-200-2011 dataset." (2011).

#### 4. Test
##### (i) Testing TPN

```bash
$ python main.py --mode test_TPN
```

##### (ii) Testing Text2Colors

```bash
$ python main.py --mode test_text2colors
```

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1804.04128).

&nbsp;



