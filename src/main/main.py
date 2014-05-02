'''
Created on May 1, 2014

@author: Jatin
'''
import os
import numpy as np
import sys

dataPath = '/Jatin/Brivas/gaze/data'
projectPath = '/Jatin/workspace/eyePupil'
leftEyePath=os.path.join(dataPath,'left')
rightEyePath=os.path.join(dataPath,'right')
sys.path.append(os.path.join(projectPath,'src'))

from support import utility
import skimage.io as io
from skimage import data
import mahotas
import itertools

from skimage.transform import resize
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import cPickle


def writePathToSamples(parentDir):
    '''write paths to samples for each labels. The .txt files are generated
    for each block and it is stored at 
    projectPath/baseNameOfparentDir/block_<number>.txt
    Each .txt file contains paths to the samples of that block for
    all the users'''
    userlist = utility.getSubdirectories(parentDir)
    print userlist

    baseName = os.path.basename(parentDir)
    data_dir = os.path.join(projectPath,baseName)
    utility.checkDirectory(data_dir)
#    blocks = ["{:02d}".format(x) for x in range(1,10)]
    for i in range(1,10):
        with open(os.path.join(data_dir,'block_'+str(i)+'.txt'),'w') as f:
            for user in userlist:
                blockPath = os.path.join(parentDir,user,'block_'+str(i))
                if os.path.isdir(blockPath):
                    img_list = os.listdir(blockPath)
                    for img in img_list:
                        img_path = os.path.join(blockPath,img)
                        if not os.path.isfile(img_path) or (img_path.split('.')[-1] != "jpg" 
                                                            and img_path.split('.')[-1] != "png"): 
                            continue
                        f.write("%s\n"%img_path)

def loadData(parentDir):
    '''From the parentDir get the dir name where .txt files are stored.
    This function will return data and the labels associated with it'''
    baseName = os.path.basename(parentDir)
    data_dir = os.path.join(projectPath,baseName)
    files = os.listdir(data_dir)
    files = [f for f in files if f.split('.')[-1]=='txt']
    data = []
    for f in files:
        label = f.split('.')[0]
        filePath = os.path.join(data_dir,f)
        with open(filePath,'r') as r:
            for image in r:
                data.append([image.strip(),label])
    return data

def extract_features(img_data):
    features = []
    labels = []
    #This part is to convert our problem into 3 class problem
    all_labels = []
    for i in range(1,10):
        all_labels.append('block_'+str(i))
    for sample in img_data:
        im = sample[0]
        label = sample[1]
        if label in all_labels[0:3]:
            label = 'left'
        elif label in all_labels[3:6]:
            label = 'center'
        elif label in all_labels[6:9]:
            label = 'right'
        else : print 'label not found'
        feature_vector = get_lbp_feature(im)
        features.append(feature_vector)
        labels.append(label)
    print len(features)
    return (features,labels)

def get_lbp_feature(im):
    img = io.imread(im,as_grey=True)
    scaled_img = resize(img,(96,112))
    patches = utility.extract_patches(scaled_img)
    descriptor = []
    for patch in patches:
        '''Extract LBP descriptor'''
        lbp_descriptor = mahotas.features.lbp(patch,2,9)
        lbp_descriptor = np.divide(lbp_descriptor,lbp_descriptor.sum())
        descriptor.append(lbp_descriptor)
    feature = np.concatenate(np.array(descriptor))
    return feature.tolist()

def classify(feature,labels,model='model'):
    print "classifying data"
    modelDir = os.path.join(projectPath,'model')
    utility.checkDirectory(modelDir)
    dataTrain,dataTest,labelsTrain,labelsTest = train_test_split(feature, 
                                                                        labels, test_size=0.20, 
                                                                        random_state=42)
    print len(dataTrain)
    print len(labelsTrain)
    print len(dataTest)
    print len(labelsTest)
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(dataTrain, labelsTrain)
        # save the classifier
    scores = cross_val_score(clf, dataTrain, labelsTrain)
    print scores.mean()
    raw_input()
    with open(os.path.join(modelDir,model+'.pkl'), 'wb') as fid:
        cPickle.dump(clf, fid)    
    # load it again
    with open(os.path.join(modelDir,model+'.pkl'), 'rb') as fid:
        clf_loaded = cPickle.load(fid)   
    predicted_label = clf_loaded.predict(dataTest)
    true_predictions = 0
    for i in range(len(labelsTest)):
        print 'Given Label = '+str(labelsTest[i])
        print 'predicted Label2 =' + str(predicted_label[i])
        if (labelsTest[i]==predicted_label[i]):
            true_predictions = true_predictions +1
    accuracy = float(true_predictions)/float(len(labelsTest))
    print 'accuracy is :', str(accuracy)

if __name__ == '__main__':
    writePathToSamples(leftEyePath)
    writePathToSamples(rightEyePath)
    leftData = loadData(leftEyePath)
    rightData = loadData(rightEyePath)
    leftFeatures, leftLabels = extract_features(leftData)
    rightFeatures, rightLabels = extract_features(rightData)
    classify(leftFeatures,leftLabels,'left')
    classify(rightFeatures,rightLabels,'right')
    
    
    