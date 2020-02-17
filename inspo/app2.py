import streamlit as st
st.title('inspo-Book')
st.header('Upload an item of clothing to find matching looks')
st.subheader('Jimmy Joy')
st.subheader('Insight Data Science, Los Angeles')

from PIL import *
import cv2

###########################################################
### uploading the image ###
###########################################################

upload = st.file_uploader('Upload an picture of the item you are trying to match')
if upload is not None:
    image1 = Image.open(upload)
    st.image(image1, caption='Uploaded Image.', width = 200)

##### User input of number of similar outfits to find

number = st.number_input('How many similar outfits would you like to see?', value = 0, step = 1)



########################################################
### importing libraries 
########################################################

import pandas as pd
import csv
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
import os
import pandas as pd
from keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
import matplotlib.gridspec as gridspec

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn import preprocessing



###########################################################
### TF model
###########################################################
### User image identification and extraction 


# import keras
import keras
import keras_maskrcnn
import keras_retinanet 

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import shutil 
import numpy as np
import time
import json



# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


######################################################

### User image identification and extraction 

# adjust this to point to your downloaded/trained model
model_path = os.path.join('/home/ec2-user/inspo/', 'resnet50_modanet.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {1: 'bag', 2: 'belt', 3: 'boots', 4: 'footwear', 5: 'outer', 6: 'dress', 7: 'sunglasses', 8: 'pants', 9: 'top', 10: 'shorts', 11: 'skirt', 12: 'headwear', 13: 'scarf/tie'}

#######################################################

def cloth(input_imagefile):
    # load image
    image = read_image_bgr(input_imagefile)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    boxes  = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks  = outputs[-1][0]

    # correct for image scale
    boxes /= scale

    masks_dic={}
    boxes_dic={}
    labels_dic={}
    counter=0

    # visualize detections
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < 0.5:
            break
    

        color = label_color(label)
    
        b = box.astype(int)
        draw_box(draw, b, color=color)
    
        mask = mask[:, :, label]
        draw_mask(draw, b, mask, color=label_color(label))
    
        masks_dic[str(counter)]=mask
        boxes_dic[str(counter)]=box
        labels_dic[str(counter)]=label
        counter+=1
    image = read_image_bgr(input_imagefile)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # visualize detections

    items_dic={}
    counter=0
    
    for box, mask2, label2 in zip(boxes_dic.values(), masks_dic.values(), labels_dic.values()):
        b = box.astype(int)

        # resize to fit the box
        mask2 = mask2.astype(np.float32)
        mask2 = cv2.resize(mask2, (b[2] - b[0], b[3] - b[1]))

        # binarize the mask1
        mask2 = (mask2 > 0.5).astype(np.uint8)

        # draw the mask2 in the image
        mask2_image = np.zeros((draw.shape[0], draw.shape[1]), np.uint8)
        mask2_image[b[1]:b[3], b[0]:b[2]] = mask2
        mask2 = mask2_image

        mask2 = (np.stack([mask2] * 3, axis = 2))*draw
        
        items_dic[str(counter)] = mask2
        counter+=1
    
#        newfileneame=input_imagefile.split("/")[4].split('.')[0]
   #     plt.ioff()
  #      plt.figure(figsize=(15, 15))
 #       plt.axis('off')
#        plt.imshow(mask2)
        #plt.savefig('/home/ec2-user/SageMaker/'+str(newfileneame)+'-masked-'+str(label2)+'.jpg',bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close('all')
        
    

    return mask2, label2


###################################################################


mask1, label = cloth(upload)


#########################################################################


from PIL import Image
testimg = Image.fromarray(mask1, 'RGB')
testimg = testimg.resize((224,224))

#####################################################################

# loading model file for AlexNEt
an_model = load_model("/home/ec2-user/inspo/an-model.h5")

######################################################################3

#Loading the weights for AlexNet
import tensorflow as tf

an_model.layers[0].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'conv2d/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'conv2d/bias')])
an_model.layers[3].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'conv2d_1/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'conv2d_1/bias')])
an_model.layers[6].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'conv2d_2/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'conv2d_2/bias')])
an_model.layers[8].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'conv2d_3/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'conv2d_3/bias')])
an_model.layers[10].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'conv2d_4/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'conv2d_4/bias')])
an_model.layers[13].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'dense/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'dense/bias')])
an_model.layers[16].set_weights([tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                            'dense_1/kernel'), 
                    tf.train.load_variable('/home/ec2-user/inspo/model.ckpt-5000', 
                                           'dense_1/bias')])

######################################################################

# Extracting features from user test using AlexNet 
INV3_feature_dic = {}
INV3_feature_list=[]


img_data = image.img_to_array(testimg)
img_data = np.expand_dims(img_data, axis=0)
#img_data = preprocess_input(img_data)
INV3_feature = an_model.predict(img_data)
feature_np = np.array(INV3_feature)
testfeature = feature_np.flatten()

###################################################################3

features =pd.read_csv('/home/ec2-user/inspo/an_features.csv', index_col='names')

####################################################################

testfeature = testfeature.reshape(1, -1)
testfeature = np.append(testfeature, label)
testfeature = np.append(testfeature, label)
testfeature = pd.Series(testfeature, index = features.columns, name = 'test')

####################################################################

features_withtest = features.append(testfeature)

### replace test label name with name instead of number 

features_withtest['label names'][features_withtest['label names'] == features_withtest['label names']['test'] ] = labels_to_names[features_withtest['label names']['test']+1]

### Scaling

scaler = preprocessing.StandardScaler()
features_withtest.loc[:,'0':'4095'] = scaler.fit_transform(features_withtest.loc[:,'0':'4095'])

sample_df = features_withtest.loc[features_withtest['labels'] == label]

from sklearn.metrics.pairwise import cosine_similarity
cos_val = cosine_similarity(sample_df.iloc[:, :2048], sample_df.iloc[-1:, :2048])

cos_val=cos_val.reshape(-1,)

sample_df['cosine'] = pd.Series(cos_val, index=sample_df.index)

sample_df_sort=sample_df.sort_values(by=['cosine'], ascending = False)

#########################################################################

fig5 = plt.figure(constrained_layout=True,figsize=(24,24))

spec5 = gridspec.GridSpec(ncols=3, nrows=1+number//3)
spec5.update(wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 
counter=1
for row in range(1+number//3):
    for col in range(3):
        if counter < 1+number:
            ax = fig5.add_subplot(spec5[row, col])
            ax.imshow(mpimg.imread('/home/ec2-user/inspo/processedimages/'+str(sample_df_sort.index[counter]).split('-')[0]+'.jpg'))
            ax.axis('off')
            
            counter+=1

st.pyplot()

######################################################################
### cosine similarity 
#####################################################################


st.pyplot()
