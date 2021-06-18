# Background

This is a repository containing our project for the course 02466 Project work - Bachelor of Artificial Intelligence and Data in Spring 2021 at The Technical University of Denmark.

The task was to create a jewellery recommendation engine that can recommend new piecs of jewellery based on a query image.


# Motivation

Say for example that you want to buy a piece of jewellery similiar to the one below but you don't know how to describe it or what exactly to search for. Instead you can search by just taking an image of the necklace and use that image to seach with.


<img src=Figures/test2.jpeg width=200px alt=necklace>


# Data set

Most training images are similiar to this one

<img src=Figures/test3.jpg width=200px alt=ring>



# How to use
Run the script ```main-run.py```
Specify an image path. The following will be output
![image](Figures/nearest.png)

# Necessary files
The dataset is not stored in this repo but should be located like this. It will be downloaded by first running `image_scaping.py`, then `data_loader.py`.

```
fagprojekt
└───data
```
when pulling, remember to `git lfs pull` to get the detectron2 model file, otherwise cropping is not possible

# Dependencies

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
For a guide to install detectron check the official docs
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# Experiments

The following outlines the scripts you need to run to reproduce the results in the report

| **Script** | **Figure** |
|--------|--------|
|       |        |
