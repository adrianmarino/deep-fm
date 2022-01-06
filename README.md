# DFM

Deep Factorization Machine Model.

### References

* [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)
* [Factorization Machines](https://d2l.ai/chapter_recommender-systems/fm.html)
* [Sistemas de Recomendación (Parte 1): Filtros Colaborativos | Clase 22 | Aprendizaje Profundo 2021](https://www.youtube.com/watch?v=YAvX3BBh7U4)
* [Factorization Machine models in PyTorch](https://github.com/rixwew/pytorch-fm)

## Requisites

* [git](https://git-scm.com/downloads)
* [anaconda](https://www.anaconda.com/products/individual) / [minconda](https://docs.conda.io/en/latest/miniconda.html)

## Getting starter

**Step 1**: Clone repo.

```bash
$ git clone https://github.com/adrianmarino/dfm.git
$ cd dfm
```

**Step 2**: Create environment:

```bash
$ cd dfm
$ conda env create -f environment.yml
```

**Paso 3**: Enable project environment:

```bash
$ conda activate dfm
```

**Paso 3**: Start regression test:

```bash
$ cd dfm
$ pytest
```

## Training

```bash
$ python bin/train_model.py
INFO ./datasets/ml-1m/ratings.dat dataset loaded!
INFO Start training...
INFO {'epoch': 2, 'train_loss': 20.14286157488823, 'time': '0:00:00.87', 'val_loss': 0.6246903538703918, 'val_auc': 0.7874902948711655, 'lr': 0.001}
INFO {'epoch': 3, 'train_loss': 19.685476034879684, 'time': '0:00:01.11', 'val_loss': 0.5780770182609558, 'val_auc': 0.790851011262817, 'lr': 0.001}
INFO {'epoch': 4, 'train_loss': 19.354053854942322, 'time': '0:00:01.10', 'val_loss': 0.5723821520805359, 'val_auc': 0.7925530963733936, 'lr': 0.001}
INFO {'epoch': 5, 'train_loss': 19.014766484498978, 'time': '0:00:01.04', 'val_loss': 0.5685867667198181, 'val_auc': 0.7937161306097898, 'lr': 0.001}
INFO {'epoch': 6, 'train_loss': 18.78430986404419, 'time': '0:00:01.03', 'val_loss': 0.5611663460731506, 'val_auc': 0.7949465228856205, 'lr': 0.001}
INFO {'epoch': 7, 'train_loss': 18.601320549845695, 'time': '0:00:00.90', 'val_loss': 0.5584127306938171, 'val_auc': 0.7958149356504789, 'lr': 0.001}
INFO {'epoch': 8, 'train_loss': 18.43570476770401, 'time': '0:00:01.09', 'val_loss': 0.5547337532043457, 'val_auc': 0.7968336349602996, 'lr': 0.001}
INFO {'epoch': 9, 'train_loss': 18.26313306391239, 'time': '0:00:01.09', 'val_loss': 0.5517903566360474, 'val_auc': 0.7977368827880491, 'lr': 0.001}
INFO {'epoch': 10, 'train_loss': 18.07542097568512, 'time': '0:00:01.11', 'val_loss': 0.5494707822799683, 'val_auc': 0.7988593627683911, 'lr': 0.001}
INFO {'epoch': 11, 'train_loss': 17.8567902892828, 'time': '0:00:01.09', 'val_loss': 0.5475335717201233, 'val_auc': 0.7996708183008361, 'lr': 0.001}
INFO {'epoch': 12, 'train_loss': 17.649917647242546, 'time': '0:00:01.11', 'val_loss': 0.5478389859199524, 'val_auc': 0.8004633910633032, 'lr': 0.001}
INFO {'epoch': 13, 'train_loss': 17.386514708399773, 'time': '0:00:00.88', 'val_loss': 0.549045205116272, 'val_auc': 0.8007445247997682, 'lr': 0.001}
INFO {'epoch': 14, 'train_loss': 17.1092601493001, 'time': '0:00:00.89', 'val_loss': 0.5484861731529236, 'val_auc': 0.8007977743257161, 'lr': 0.001}
INFO {'epoch': 15, 'train_loss': 16.799456879496574, 'time': '0:00:00.90', 'val_loss': 0.5505458116531372, 'val_auc': 0.8002330459560807, 'lr': 0.001}
```
