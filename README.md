# RefrTomo

This library contains all the required building blocks to peform 2D turning ray tomography and associated examples codes.

## Project structure
This repository is organized as follows:

* :open_file_folder: **refrtomo**: python library containing routines for 2D turning ray tomography;
* :open_file_folder: **data**: folder containing data
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);


## Notebooks
The following notebooks are provided:

- :orange_book: ``Rayshooting_straight.ipynb``: notebook testing raytracer and tomographic matrix engine with straight rays;
- :orange_book: ``Rayshooting.ipynb``: notebook testing raytracer and tomographic matrix engine with turning rays;
- :orange_book: ``Rayshooting_Stryde.ipynb``: notebook testing raytracer and tomographic matrix engine with small refraction dataset acquired with Stryde sensors in KAUST;
- :orange_book: ``RefrTomo.ipynb``: notebook performing refraction tomographic of a portion of synthetic Marmousi model;
- :orange_book: ``RefrTomo_minimal.ipynb``: notebook performing refraction tomographic of a portion of synthetic Marmousi model (minimal number of steps);
- :orange_book: ``FirstArrivalPicking_Stryde.ipynb``: notebook performing automated first arrival picking of a small refraction dataset acquired with Stryde sensors in KAUST;
- :orange_book: ``RefrTomo_Stryde.ipynb``: notebook performing refraction tomographic of a small refraction dataset acquired with Stryde sensors in KAUST.
- :orange_book: ``RefrTomo_Strydeminimal.ipynb``: notebook performing refraction tomographic of a small refraction dataset acquired with Stryde sensors in KAUST (minimal number of steps).




## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate refrtomo
```


## Improvements :point_up:

- [ ] Create routine that takes two rays reaching the surface and recursively interpolates a ray in the middle until a given distance from a surface point of interest (to be used to take the two closest rays on either side of a receiver, which are however above the allowed threshold, and find one in the middle below this threshold);

