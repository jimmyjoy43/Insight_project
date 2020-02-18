# Inspo-book
Data Science project developed at Insight, January 2020

## Project Description
Inspo-book is at it's core a clothing detection and recommendation web application. It takes the user input of an image, 
identifies the item of clothing in the image, and extracts a feature vector representing this item's attributes like 
color and texture using a convolutional neural network. Inspo-book them compares similar feature vectors from a set of 
images scrapped from Reddit to recommend the three outfits that have an item closely matching the user's input. In this manner, 
Inspo-book inspires the user to build new outfits around items of clothing the user already has in their closet. 

![inspo](inspo.gif)

## Repo Structure

+ `inspo`: contains the Streamlit python file to run inspo-book 
+ `notebooks`: miscellaneous ipython notebooks
+ `web scrapers`: web scrapers for Reddit using BeautifulSoup, PushShift and the Python Reddit API

## Getting Started

#### Requisites
The code uses python 3.7, and Keras with Tensorflow backend. Training was performed on an AWS *ml.c5.4xlarge* SageMaker notebook instance.

#### Installation

#### Clone, setup conda environment
Clone this repo with:
```
git clone https://github.com/jimmy-io/Insight_project.git
cd inspo/
```

Simply setup the required libraries with 
```
pip3 install -r requirements.txt
```
## Usage
The script doing the work is app.py (for features extracted using InceptionV3 network) or app2.py (for features extracted using AlexNet). The web app can be executed with:

```
streamlit run app2.py
```
And follow the external URL Streamlit provides. 

## License
Unless explicitly stated at the top of a file, all code is licensed under the MIT license.
Modanet data set is released under the Creative Commons Attribution-NonCommercial license 4.0 by Shuai Zheng, Fan Yang, M. Hadi Kiapour, Robinson Piramuthu. ModaNet: A Large-Scale Street Fashion Dataset with Polygon Annotations. ACM Multimedia, 2018
The iMaterialist data set can be obtained from: https://github.com/visipedia/imat_fashion_comp
