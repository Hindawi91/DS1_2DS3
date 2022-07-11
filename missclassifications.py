import keras
import cv2
import tensorflow as tf
import numpy as np
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
from glob import glob                                                           
import cv2 
import pandas as pd
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
import timeit
import datetime

# In[6]:

def find_missclassifications (y_ture, y_pred):
    missclassifieds = []
    for i, y in enumerate(y_pred):
        if y_pred [i] != y_true[i]:
            missclassifieds.append(im_files[i])
    return missclassifieds
             
All_Models = []
All_Metrics = []
All_CMs = []

expirements = ["exp1"]
Files_Names = "90k+110k"
GAN_Models = [190000]
# expirements = ["exp1"]
for exp in GAN_Models:

    models_names = glob(f'*.hdf5*')
    

    categories = ['ONB_BIC','CHF']
    images = []
    y_true = []
    
    print(f"getting data for {exp} ......")
    
    for j,category in enumerate (categories): 
        im_files = glob(f'./brats_syn_256_lambda0.1/test_results_{exp}/{category}/*.j*')

        for i,im_file in enumerate (im_files):
            
            if category == 'CHF':
                y_true.append(0)
            elif category == 'ONB_BIC':
                y_true.append(1)

            

            img1 = image.load_img(im_file,target_size=(128, 72))
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 /= 255.
            images.append(img1)
            


    for model_name in models_names:
        
        begin_time = datetime.datetime.now()
        
        model = keras.models.load_model(f"./{model_name}")
        
        print (f"predicting using model {model_name} on results from GAN{exp}.......")
        
        imagesNP = np.vstack(images)
        y_pred = model.predict_classes(imagesNP)
        y_pred_prob = model.predict_proba(imagesNP)[:,1]
        print("predicted",y_pred[10])
        print("prediction probs", y_pred_prob[10])
        
        missclassifides = find_missclassifications (y_true, y_pred)
        
        print(missclassifides)
        
        


