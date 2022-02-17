# DeepFM

Deep Factorization Machine Model for [CRT](https://en.wikipedia.org/wiki/Click-through_rate) prediccion.

### References

* [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (Paper)](https://arxiv.org/pdf/1703.04247.pdf)
* [Factorization Machines](https://d2l.ai/chapter_recommender-systems/fm.html)
* [Sistemas de Recomendación (Parte 1): Filtros Colaborativos | Clase 22 | Aprendizaje Profundo 2021](https://www.youtube.com/watch?v=YAvX3BBh7U4)
* [Factorization Machine models in PyTorch](https://github.com/rixwew/pytorch-fm)


## Notebooks

* [Use example](https://github.com/adrianmarino/deep-fm/blob/master/notebooks/rs.ipynb)
* [rs-check-how-works](https://github.com/adrianmarino/deep-fm/blob/master/notebooks/rs-check-how-works.ipynb)
 

## Requisites

* [anaconda](https://www.anaconda.com/products/individual) / [minconda](https://docs.conda.io/en/latest/miniconda.html)
* pytorch-common
  * [Github repo](https://github.com/adrianmarino/pytorch-common/tree/master)
  * [Pypi repo](https://pypi.org/project/pytorch-common/)

## Getting starter

**Step 1**: Clone repo.

```bash
$ git clone https://github.com/adrianmarino/deep-fm.git
$ cd deep-fm
```

**Step 2**: Create environment.

```bash
$ cd dfm
$ conda env create -f environment.yml
```

**Step 3**: Enable project environment.

```bash
$ conda activate deep-fm
```

**Step 3**: Run regression tests.

```bash
$ cd deep-fm
$ pytest
```

## Training

```bash
$ python bin/train_model.py
```

```bash
$ python bin/train_model.py --help

Usage: train_model.py [OPTIONS]

Options:
  --device TEXT                   Device used to functions and optimize model.
                                  Values: gpu(default) or cpu.
  --cuda-process-memory-fraction FLOAT
                                  Setup max memory used per CUDA process.
                                  Percentage expressed between 0 and
                                  1(default: 0.5).
  --dataset TEXT                  Select movie lens dataset type. Values:
                                  1m(default), 20m.
  --cv-n-folds INTEGER            cross validation n folds(default: 10).
  --train-percent FLOAT           Observations percent to used on training
                                  process(default: 0.7).
  --help                          Show this message and exit.
```
