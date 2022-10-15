from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import pickle


# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import os
import cv2
#import matplotlib.pyplot as plt
import math
import pandas as pd
categories=['1221','1509','1637','1718','PB 4','PB 5','PR 11','PB 1121',
            'PR 113','PR 114','PR 116','PR 121','PR 122','PR 123','PR 124','PR 127','PR 133','PR 111',
            'Shabnam rice','Sharbati rice','Sugandha rice']

def rotate(im3):
    rotated=im3.copy()
    sb=0
    (h, w) = im3.shape
    (cX, cY) = (w // 2, h // 2)
    for i in range(im3.shape[1]):
        if im3[0,i]==0:
            sb+=1
    if sb>10:
        M = cv2.getRotationMatrix2D((cX, cY), 38, 1.0)
        rotated = cv2.warpAffine(im3, M, (w, h))
        #plt.imshow(rotated,cmap='gray')
    else:
        for i in range(im3.shape[1]-1,-1,-1):
            if im3[0,i]==0:
                sb+=1
            if sb>10:
                M = cv2.getRotationMatrix2D((cX, cY), -18, 1.0)
                rotated = cv2.warpAffine(im3, M, (w, h))
                #plt.imshow(rotated,cmap='gray')
    return rotated
def getboundedRect(im):  
    n=[]
    flag=0
    left,top,right,bottom=0,0,0,0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]!=0:
                left,top=i+1,j+1
                flag=1
                break
        if flag==1:
            break
    for j in range(top,im.shape[1]):
        if im[left,j]==0:
            right=j-1
            break
    for i in range(left,im.shape[0]):
        if im[i,top]==0:
            bottom=i-1
            break
    im1=im[left+1:bottom,top+1:right]
    #op=cv2.fastNlMeansDenoising(im1,None,20,7,21)
    return im1
def preprocess(img):
    img=getboundedRect(img)
    
    return img
def displayimagematrix(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            print(im[i,j],end=" ")
        print()
def extractmomentfeatures(im):
    moments=cv2.moments(im)  # returns a dictionary
    feat=[]
    for i,j in moments.items():
        feat.append(j)
    return feat
def extractHuMoments(im):
    mfeat=cv2.moments(im)
    humoments=cv2.HuMoments(mfeat)  # & Humoments
    mfeat=humoments.flatten()
    for i in range(len(mfeat)):
        humoments[i]=-1*np.sign(humoments[i])*np.log10(np.abs(humoments[i]))
    return humoments.flatten()
def enhance(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]<50:
                im[i,j]=0
    return im
def getmin(c):
    if len(c)!=0:
        return min(c)
    else:
        return 0
def getexactimage(im):
    c=[]
    b=0
    startrow=startcol=0
    endrow=im.shape[0]-1
    endcol=im.shape[1]-1
    for i in range(im.shape[0]):
        b=0
        for j in range(im.shape[1]//2):
            if im[i,j]==0:
                b+=1
        c.append(b)
    startcol=getmin(c)
    c=[]
    for i in range(im.shape[0]):
        b=0;
        for j in range(im.shape[1]-1,im.shape[1]//2,-1):
            if im[i,j]==0:
                b+=1
        c.append(b)
    endcol=im.shape[1]-getmin(c)-1
    c=[]
    for j in range(im.shape[1]):
        b=0
        for i in range(im.shape[0]//2):
            if im[i,j]==0:
                b+=1
        c.append(b)
        startrow=getmin(c)

    c=[]
    for j in range(im.shape[1]):
        b=0
        for i in range(im.shape[0]-1,im.shape[0]//2,-1):
            if im[i,j]==0:
                b+=1
        c.append(b)
    endrow=im.shape[0]-getmin(c)-1
    return im[startrow:endrow+1,startcol:endcol+1]
def getfeatures(path):
    feature=[]
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    im=getboundedRect(im)
    im=rotate(im)
    im=enhance(im)
    im=getexactimage(im)
    #moments=extractmomentfeatures(im)
    humoments=extractHuMoments(im)
    area=im.shape[0]*im.shape[1]
    Perimeter=2*(im.shape[0]+im.shape[1])
    if im.shape[1]!=0:
        aspectRatio=im.shape[0]/im.shape[1]
    else:
        aspectRatio=.5

    ret, thresh = cv2.threshold(im, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    #img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
    areas=[]
    perimeters=[]
    hullarea=[]
    for cnt in contours:
        areas.append(cv2.contourArea(cnt,False))
        perimeters.append(cv2.arcLength(cnt,True))
        hull=cv2.convexHull(cnt)
        hullarea.append(cv2.contourArea(hull))
        ExtremeLeftmostPoint = tuple(cnt[cnt[:,:,0].argmin()][0])
        ExtremeRightmostPoint = tuple(cnt[cnt[:,:,0].argmax()][0])
        ExtremeTopmostPoint = tuple(cnt[cnt[:,:,1].argmin()][0])
        ExtremeBottommostPoint = tuple(cnt[cnt[:,:,1].argmax()][0])
    if len(areas)!=0:
        MinContourArea=min(areas)
        MaxContourArea=max(areas)
        extent=MaxContourArea/area
    if len(perimeters)!=0:
        MinContourLength=min(perimeters)
        MaxContourLength=max(perimeters)
    if len(hullarea)!=0:
        HullArea=max(hullarea)

    if HullArea!=0:
        Solidity=float(area)/HullArea
    EquivalentDiameter=math.sqrt(4*area/math.pi)
    feature=[area,Perimeter,aspectRatio,MinContourArea,MaxContourArea,MinContourLength,MaxContourLength,extent,HullArea,Solidity,EquivalentDiameter]
    #for i in moments:
        #feature.append(i)
    for i in humoments:
        feature.append(i)
    return feature

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models\\ricemomentmodel.pickle'

# Load your trained model

model = pickle.load(open(MODEL_PATH,'rb'))
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
##model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def model_predict(img_path, model):
    feat=getfeatures(img_path)

    # Preprocessing the image


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    preds=model.predict([feat])
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = categories[int(preds[0])] 
        return pred_class
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.run()