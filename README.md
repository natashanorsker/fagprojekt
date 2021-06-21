# Background

This is a repository containing our project for the course 02466 Project work - Bachelor of Artificial Intelligence and Data in Spring 2021 at The Technical University of Denmark.

The task was to create a jewellery recommendation engine that can recommend new pieces of jewellery based on a query image. The project was made in collaboration with Pandora jewellery and the consulting firm eCapacity.


# Motivation

Say for example that you want to buy a piece of jewellery similiar to the one below but you don't know how to describe it or what exactly to search for. Instead you can search by just taking an image of the necklace and use that image to seach with.

<p align="center">
<img src=Figures/test2.jpeg width=200px alt=necklace>
</p>

# Dataset

The data set obtained for this project is images of around 1700 products scraped from the various pandora regions sites. A few examples are shown here.

<p align="center">
<img src=survey\Questionnaire_imgs\180919CZ.jpg width=96px alt=ring> <img src=survey\Questionnaire_imgs\781811.jpg width=96px alt=charms> <img src=survey\Questionnaire_imgs\396240CZ.jpg width=96px alt=necklace>
</p>

# Results

3D PCA of embedding space generated by the triplet model. It shows that the model was able to capture semantic meaning.
<p align="center">
<img src=Figures/3dpca.png width=400px>
</p>

Shown here spherized
<p align="center">
<img src=Figures/3dsphere.gif width=400px>
</p>


## Examples of recommendations
### Triplet model
<p align="center">
<img src=Figures/triplet_recommendations.png width=400px>
</p>

### Variational autoencoder
<p align="center">
<img src=Figures/vae_recommendations.png width=400px>
</p>

In Human experiments, the recommendations from the variational autoencoder were preferred. 

# How to use

The autoencoder is contained in the autoencoder folder, the triplet network is contained in the deepRanking folder and this is where the most work has been done.

To get model files run the ``train_vae.py`` and ``deepRanking/deep_ranking_gridsearch.py`` or ``deepRanking/production_model.py`` to get a single usable model.

# Necessary files
The dataset is not stored in this repo but should be located like this. It will be downloaded by first running `image_scaping.py`, then `data_loader.py`.

```
fagprojekt
└───data
```
When pulling, remember to `git lfs pull` to get the detectron2 model file, otherwise cropping is not possible.

An additional file `masterdata.csv` is also necessary to place in the data_code folder but unfortunately we cannot share it here. 

# Dependencies

All code in the repository is tested to work with python 3.8 / 3.9 and and pytorch 1.7 / 1.8.1, Tensorflow 2.5 (although the autoencoder seems to require Tensforflow 2.3 to train)
```
pytorch
detectron2
pillow
opencv-python
numpy
matplotlib
sklearn
scipy
```
For a guide to install detectron check the official docs (we used v0.4)
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# Experiments

The following outlines the scripts you need to run to reproduce the results in the report

| **Script** | **Figure** |
|--------|--------|
|   `deepRanking/plots.py`   |    Fig. 4, distribution of dataclasses    |
| `deepRanking/plots.py`| Fig. 5 PCA plot of data set |
|`autoencoder/plot_vae.py`| Fig. 8 Visualisation of autoencoder vs variationer autoencoder | 
|`detectron2segment/Detectron2_trainer.ipynb`| Fig. 20, image segmentation |
|`deepRanking/runs`| Fig. 23/24/34 set tensorboard logdir to this folder and screengrab the relevant screens.|
|`deepRanking/mAP.py`| Tab. 7/8, note that it takes approx 1 hour to run. manunally change to categories to search for|
|`deepRanking/cmc_recall_plot.py`| Fig. 25/26, uses the supplied npz files optained from running `deepRanking/mAP.py`|
|`survey/survey_analysis.R`| Fig. 27/28/29 statistical comparisons of survey results|
|`survey/survey_analysis.R`| Tab. 11/12/13/14 tables comparisons of survey results|
|``triplet_recommendations.py``|Fig. 31 recommendations from triplet ranking model|
|``autoencoder_vae/recommendations.py``|Fig. 32 recommendations from variational autoencoder ranking model|
