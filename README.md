# Inspo-book
Data Science project developed at Insight, January 2020

## Project Description
Inspo-book is at it's core a clothing detection and recommendation web application. It takes the user input of an image, 
identifies the item of clothing in the image, and extracts a feature vector representing this item's attributes like 
color and texture using a convolutional neural network. Inspo-book them compares similar feature vectors from a set of 
images scrapped from Reddit to recommend the three outfits that have an item closely matching the user's input. In this manner, 
Inspo-book inspires the user to build new outfits around items of clothing the user already has in their closet. 



## Repo Structure

+ `data`: an example set of images
+ `notebooks`: miscellaneous ipython notebooks
+ `web scrapers`: web scrapers for Reddit using BeautifulSoup, PushShift and the Python Reddit API


## License
Unless explicitly stated at the top of a file, all code is licensed under the MIT license.
