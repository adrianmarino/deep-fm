# DeepFM

Deep Factorization Machine Model for [CRT](https://en.wikipedia.org/wiki/Click-through_rate) prediccion.

### References

* [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)
* [Factorization Machines](https://d2l.ai/chapter_recommender-systems/fm.html)
* [Sistemas de Recomendación (Parte 1): Filtros Colaborativos | Clase 22 | Aprendizaje Profundo 2021](https://www.youtube.com/watch?v=YAvX3BBh7U4)
* [Factorization Machine models in PyTorch](https://github.com/rixwew/pytorch-fm)


## notebooks

* [rs](https://github.com/adrianmarino/dfm/blob/master/notebooks/rs.ipynb)
* [rs-check-how-works](https://github.com/adrianmarino/dfm/blob/master/notebooks/rs-check-how-works.ipynb)


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

**Paso 3**: Run regression tests:

```bash
$ cd dfm
$ pytest
```

## Training

```bash
$ python bin/train_model.py

2022-01-06 21:47:54,145 MainProcess root INFO ../datasets/ml-1m/ratings.dat dataset loaded! Shape: (1000209, 2)
2022-01-06 21:47:54,195 MainProcess root INFO Start training...
2022-01-06 21:48:05,794 MainProcess root INFO {'time': '0:00:05.55', 'epoch': 2, 'train_loss': 96.89652383327484, 'val_loss': 0.5426889061927795, 'val_auc': 0.7907098209219573, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:11,705 MainProcess root INFO {'time': '0:00:05.85', 'epoch': 3, 'train_loss': 95.11382699012756, 'val_loss': 0.5383059978485107, 'val_auc': 0.793302029966962, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:17,625 MainProcess root INFO {'time': '0:00:05.86', 'epoch': 4, 'train_loss': 94.11347880959511, 'val_loss': 0.5369410514831543, 'val_auc': 0.7942879591591123, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:23,489 MainProcess root INFO {'time': '0:00:05.81', 'epoch': 5, 'train_loss': 93.46087232232094, 'val_loss': 0.5363039970397949, 'val_auc': 0.7951937879488736, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:29,368 MainProcess root INFO {'time': '0:00:05.82', 'epoch': 6, 'train_loss': 93.00318217277527, 'val_loss': 0.5356779098510742, 'val_auc': 0.7960232098935927, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:35,262 MainProcess root INFO {'time': '0:00:05.84', 'epoch': 7, 'train_loss': 92.49271914362907, 'val_loss': 0.5344399809837341, 'val_auc': 0.7974643544928881, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:41,098 MainProcess root INFO {'time': '0:00:05.78', 'epoch': 8, 'train_loss': 92.00186142325401, 'val_loss': 0.5327858328819275, 'val_auc': 0.7991877111217052, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:46,807 MainProcess root INFO {'time': '0:00:05.65', 'epoch': 9, 'train_loss': 91.28615865111351, 'val_loss': 0.5307636260986328, 'val_auc': 0.8012521971089442, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:52,555 MainProcess root INFO {'time': '0:00:05.69', 'epoch': 10, 'train_loss': 90.70660012960434, 'val_loss': 0.5298435091972351, 'val_auc': 0.8023009990353678, 'patience': 0, 'lr': 0.001}
2022-01-06 21:48:58,549 MainProcess root INFO {'time': '0:00:05.94', 'epoch': 11, 'train_loss': 90.11680521070957, 'val_loss': 0.5296309590339661, 'val_auc': 0.8028243791208586, 'patience': 0, 'lr': 0.001}
2022-01-06 21:49:04,130 MainProcess root INFO {'time': '0:00:05.53', 'epoch': 12, 'train_loss': 89.67735868692398, 'val_loss': 0.5292838215827942, 'val_auc': 0.803515671412783, 'patience': 0, 'lr': 0.001}
2022-01-06 21:49:09,796 MainProcess root INFO {'time': '0:00:05.61', 'epoch': 13, 'train_loss': 89.14242053031921, 'val_loss': 0.5299152731895447, 'val_auc': 0.8033158798000799, 'patience': 0, 'lr': 0.001}
2022-01-06 21:49:15,485 MainProcess root INFO {'time': '0:00:05.63', 'epoch': 14, 'train_loss': 88.7641935646534, 'val_loss': 0.529975414276123, 'val_auc': 0.8031260300705301, 'patience': 1, 'lr': 0.001}
2022-01-06 21:49:21,178 MainProcess root INFO {'time': '0:00:05.64', 'epoch': 15, 'train_loss': 86.67957392334938, 'val_loss': 0.5314153432846069, 'val_auc': 0.8038199164535775, 'patience': 2, 'lr': 0.0001}
2022-01-06 21:49:26,810 MainProcess root INFO {'time': '0:00:05.58', 'epoch': 16, 'train_loss': 86.42091783881187, 'val_loss': 0.5308202505111694, 'val_auc': 0.8041260298209563, 'patience': 0, 'lr': 0.0001}
2022-01-06 21:49:32,572 MainProcess root INFO {'time': '0:00:05.70', 'epoch': 17, 'train_loss': 86.14583252370358, 'val_loss': 0.5306746959686279, 'val_auc': 0.8043197501507591, 'patience': 0, 'lr': 0.0001}
2022-01-06 21:49:38,524 MainProcess root INFO {'time': '0:00:05.89', 'epoch': 18, 'train_loss': 86.02148641645908, 'val_loss': 0.530491828918457, 'val_auc': 0.804388646639788, 'patience': 0, 'lr': 0.0001}
2022-01-06 21:49:44,189 MainProcess root INFO {'time': '0:00:05.61', 'epoch': 19, 'train_loss': 85.73756909370422, 'val_loss': 0.5308253169059753, 'val_auc': 0.8043685140269367, 'patience': 0, 'lr': 0.0001}
2022-01-06 21:49:49,848 MainProcess root INFO {'time': '0:00:05.60', 'epoch': 20, 'train_loss': 85.4438521116972, 'val_loss': 0.530732274055481, 'val_auc': 0.8043741050732824, 'patience': 1, 'lr': 1e-05}
2022-01-06 21:49:55,721 MainProcess root INFO {'time': '0:00:05.82', 'epoch': 21, 'train_loss': 85.36180830001831, 'val_loss': 0.5304549932479858, 'val_auc': 0.8043753010959659, 'patience': 0, 'lr': 1e-05}
2022-01-06 21:50:01,637 MainProcess root INFO {'time': '0:00:05.86', 'epoch': 22, 'train_loss': 85.30105184018612, 'val_loss': 0.530529797077179, 'val_auc': 0.8043737839912317, 'patience': 0, 'lr': 1.0000000000000002e-06}
2022-01-06 21:50:07,537 MainProcess root INFO {'time': '0:00:05.84', 'epoch': 23, 'train_loss': 85.29549261927605, 'val_loss': 0.5304415225982666, 'val_auc': 0.8043764119043915, 'patience': 1, 'lr': 1.0000000000000002e-06}
2022-01-06 21:50:13,374 MainProcess root INFO {'time': '0:00:05.78', 'epoch': 24, 'train_loss': 85.33127601444721, 'val_loss': 0.5307154059410095, 'val_auc': 0.8043711127776818, 'patience': 0, 'lr': 1.0000000000000002e-07}
2022-01-06 21:50:19,263 MainProcess root INFO {'time': '0:00:05.83', 'epoch': 25, 'train_loss': 85.27048416435719, 'val_loss': 0.5306497812271118, 'val_auc': 0.8043681938540804, 'patience': 1, 'lr': 1.0000000000000002e-07}
2022-01-06 21:50:25,187 MainProcess root INFO {'time': '0:00:05.87', 'epoch': 26, 'train_loss': 85.38470436632633, 'val_loss': 0.5306235551834106, 'val_auc': 0.8043720136758188, 'patience': 2, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:30,993 MainProcess root INFO {'time': '0:00:05.75', 'epoch': 27, 'train_loss': 85.31247788667679, 'val_loss': 0.5305320024490356, 'val_auc': 0.8043726945943375, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:36,766 MainProcess root INFO {'time': '0:00:05.71', 'epoch': 28, 'train_loss': 85.27476288378239, 'val_loss': 0.530694305896759, 'val_auc': 0.8043694422690995, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:42,446 MainProcess root INFO {'time': '0:00:05.62', 'epoch': 29, 'train_loss': 85.30920684337616, 'val_loss': 0.5306165814399719, 'val_auc': 0.8043697788983766, 'patience': 1, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:48,079 MainProcess root INFO {'time': '0:00:05.58', 'epoch': 30, 'train_loss': 85.32866214215755, 'val_loss': 0.5304827094078064, 'val_auc': 0.8043711494636814, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:53,664 MainProcess root INFO {'time': '0:00:05.53', 'epoch': 31, 'train_loss': 85.37240533530712, 'val_loss': 0.530687689781189, 'val_auc': 0.8043674143561441, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:50:59,534 MainProcess root INFO {'time': '0:00:05.81', 'epoch': 32, 'train_loss': 85.35424067080021, 'val_loss': 0.5305352210998535, 'val_auc': 0.8043738569995529, 'patience': 1, 'lr': 1.0000000000000004e-08}
2022-01-06 21:51:05,463 MainProcess root INFO {'time': '0:00:05.87', 'epoch': 33, 'train_loss': 85.29677991569042, 'val_loss': 0.5306407809257507, 'val_auc': 0.8043702334501852, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:51:11,470 MainProcess root INFO {'time': '0:00:05.95', 'epoch': 34, 'train_loss': 85.2883033901453, 'val_loss': 0.5304749608039856, 'val_auc': 0.804376263705682, 'patience': 1, 'lr': 1.0000000000000004e-08}
2022-01-06 21:51:17,435 MainProcess root INFO {'time': '0:00:05.91', 'epoch': 35, 'train_loss': 85.33005565404892, 'val_loss': 0.5304862260818481, 'val_auc': 0.8043752025847377, 'patience': 0, 'lr': 1.0000000000000004e-08}
2022-01-06 21:51:23,262 MainProcess root INFO {'time': '0:00:05.77', 'epoch': 36, 'train_loss': 85.27811680734158, 'val_loss': 0.5305851101875305, 'val_auc': 0.8043733597837905, 'patience': 1, 'lr': 1.0000000000000004e-08}
2022-01-06 21:51:28,985 MainProcess root INFO {'time': '0:00:05.67', 'epoch': 37, 'train_loss': 85.32134871184826, 'val_loss': 0.5305461883544922, 'val_auc': 0.8043721797174711, 'patience': 2, 'lr': 1.0000000000000004e-08}
```
