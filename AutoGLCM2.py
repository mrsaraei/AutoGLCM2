# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoGLCM2: GLCM-Based Automated Features Extraction for Machine Learning Models")
print("Creator: Mohammad Reza Saraei")
print("Contact: mrsaraei@yahoo.com")
print("Supervisor: Dr. Sebelan Danishver")
print("Created Date: May 29, 2022")
print("") 

# print("------------------------------------------------------")
# print("---------------- Initial Description -----------------")
# print("------------------------------------------------------")
# print("")

# First Method: 
# skimage.feature.greycomatrix(image, distances, angles, levels = None, symmetric = False, normed = False)
# Distances: List of pixel pair distance offsets
# Angles: List of pixel pair angles in radians

# Second Method:
# skimage.feature.greycoprops(P, prop)
# Prop: Computing the property of the GLCM: (‘Contrast’, ‘Dissimilarity’, ‘Homogeneity’, ‘Energy’, ‘Correlation’, ‘ASM’)

print("------------------------------------------------------")
print("------------------ Import Libraries ------------------")
print("------------------------------------------------------")
print("")

# Import Libraries for Python
import os
import cv2
import glob
import numpy as np
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("---------------- Pixel Data Ingestion ----------------")
print("------------------------------------------------------")
print("")

# Import Images From Folders 
ImagePath = "Images2/"

print(os.listdir(ImagePath))
print("")

print("------------------------------------------------------")
print("---------------- Image Preprocessing -----------------")
print("------------------------------------------------------")
print("")

# Creating Empty List
images = []
target = []

for target_path in glob.glob(ImagePath + '//**/*', recursive = True):
    lable = target_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(target_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (128, 128)) 
        gray = color.rgb2gray(img)
        img = img_as_ubyte(gray)
        plt.figure()
        plt.imshow(img, cmap = 'gray')
        images.append(img)
        target.append(lable)
        print(img_path)

# Convert List to Array
images = np.array(images)
target = np.array(target)

print("")
print('Resized Images Shape:', images.shape)
print("")

print("------------------------------------------------------")
print("------------------- GLCM Function --------------------")
print("------------------------------------------------------")
print("")

# Creating GLCM Matrix
def FE(dataset):
    ImageDF = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[image, :, :]
        
        GLCM = greycomatrix(img, [1], [0])        
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr
        GLCM_asm = greycoprops(GLCM, 'contrast')[0]
        df['ASM'] = GLCM_asm
        
        GLCM2 = greycomatrix(img, [3], [0])        
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2       
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2
        GLCM_asm2 = greycoprops(GLCM2, 'contrast')[0]
        df['ASM2'] = GLCM_asm2
        
        GLCM3 = greycomatrix(img, [5], [0])        
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3       
        GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3
        GLCM_asm3 = greycoprops(GLCM3, 'contrast')[0]
        df['ASM3'] = GLCM_asm3
        
        GLCM4 = greycomatrix(img, [0], [np.pi/2])        
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4     
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4       
        GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        GLCM_asm4 = greycoprops(GLCM4, 'contrast')[0]
        df['ASM4'] = GLCM_asm4
            
        GLCM5 = greycomatrix(img, [0], [np.pi/4])        
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5     
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5       
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5      
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        GLCM_asm5 = greycoprops(GLCM5, 'contrast')[0]
        df['ASM5'] = GLCM_asm5
        
        ImageDF = ImageDF.append(df)
    return ImageDF

print("------------------------------------------------------")
print("------------------- GLCM Propertis -------------------")
print("------------------------------------------------------")
print("")

# Extracting Features from All Images        
ImageFeatures = FE(images)
ImageFeatures['Diagnosis'] = target
print(ImageFeatures)        
print("")

print("------------------------------------------------------")
print("-------------------- Save Output ---------------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(ImageFeatures).to_csv('AutoGLCM.csv', index = False)

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")


