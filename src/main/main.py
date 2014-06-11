'''
Created on May 1, 2014

@author: Jatin
'''
import os
import numpy as np
import sys
import time
projectPath = '/Jatin/workspace/eyePupil'
dataPath = os.path.join(projectPath,'data')
leftEyePath=os.path.join(dataPath,'left')
rightEyePath=os.path.join(dataPath,'right')
sys.path.append(os.path.join(projectPath,'src'))

from support import utility
import skimage.io as io
from skimage import data
import mahotas
import itertools
from sklearn import svm, grid_search
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate

from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import cPickle
from sklearn.metrics import classification_report

modulePath = '/Jatin/workspace/eyeGaze/src'
sys.path.append(modulePath)
import cv2
import brivasmodule
import numpy

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

    for sample in img_data:
        im = sample[0]
        label = sample[1]
        label = utility.change_label(label,3)
        feature_vector = get_lbp_feature(im)
        features.append(feature_vector)
        labels.append(label)
    print 'No of features '+str(len(features))
    return (features,labels)

def get_lbp_feature(img,isImgpath = 'True'):
    #print im
    if isImgpath:
        img = io.imread(img,as_grey=True)
    #print img.shape
    #raw_input()
    scaled_img = resize(img,(48,112))
    patches = utility.extract_patches(scaled_img)
    descriptor = []
    for patch in patches:
        '''Extract LBP descriptor'''
        lbp_descriptor = mahotas.features.lbp(patch,2,9)
        lbp_descriptor = np.divide(lbp_descriptor,lbp_descriptor.sum())
        descriptor.append(lbp_descriptor)
    feature = np.concatenate(np.array(descriptor))
    return feature.tolist()

def classify_svm(feature,labels,model='left'):
    print "---------SVM Classifier-------------"
    modelDir = os.path.join(projectPath,model)
    utility.checkDirectory(modelDir)
    dataTrain,dataTest,labelsTrain,labelsTest = train_test_split(feature, 
                                                                        labels, test_size=0.20, 
                                                                        random_state=42)
    param_grid = [
                  {'C': [1, 10, 100, 1000], 'gamma': [1,0.1,0.001, 0.0001],'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'gamma': [1,0.1,0.001, 0.0001], 'kernel': ['rbf']},
                  ]
    svc = svm.SVC()
    clf = grid_search.GridSearchCV(estimator=svc, param_grid=param_grid,cv=5,n_jobs=-2)
    print "Training SVM classifier for grid of C and gamma values to select best parameter\n"
    start = time.time()
    clf.fit(dataTrain,labelsTrain)
    end = time.time()
    elapsed = end - start
    print("Time take  : %f seconds"%elapsed)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labelsTest, clf.predict(dataTest)
    print(classification_report(y_true, y_pred))
    print()
    with open(os.path.join(modelDir,model+'_svm.pkl'), 'wb') as fid:
        cPickle.dump(clf, fid) 

def classify_rfc(feature,labels,model='left'):
    print "---------Random Forest Classifier-------------"

    modelDir = os.path.join(projectPath,model)
    utility.checkDirectory(modelDir)
    dataTrain,dataTest,labelsTrain,labelsTest = train_test_split(feature, 
                                                                        labels, test_size=0.20, 
                                                                        random_state=42)
    param_grid = [
              {'n_estimators': [1, 10, 100, 1000], 'max_features': [10,50,100, 400]},
              ]
    rfc = RandomForestClassifier(n_estimators=10)
    clf = grid_search.GridSearchCV(estimator=rfc, param_grid=param_grid,cv=5,n_jobs=-2)
    print "Classification for "+model+" eye\n"
    print "Training RFC classifier for grid of C and gamma values to select best parameter\n"
    start = time.time()
    clf.fit(dataTrain,labelsTrain)
    end = time.time()
    elapsed = end - start
    print("Time take  : %f seconds"%elapsed)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labelsTest, clf.predict(dataTest)
    print(classification_report(y_true, y_pred))
    print()
    with open(os.path.join(modelDir,model+'_rfc.pkl'), 'wb') as fid:
        cPickle.dump(clf, fid)


#     
# def classify(feature,labels,model='model'):
#     print "classifying data"
#     modelDir = os.path.join(projectPath,model)
#     utility.checkDirectory(modelDir)
#     dataTrain,dataTest,labelsTrain,labelsTest = train_test_split(feature, 
#                                                                         labels, test_size=0.20, 
#                                                                         random_state=42)
#     clf = RandomForestClassifier(n_estimators=10)
#     clf = clf.fit(dataTrain, labelsTrain)
#         # save the classifier
#     scores = cross_val_score(clf, dataTrain, labelsTrain)
#     print scores.mean()
#     with open(os.path.join(modelDir,model+'.pkl'), 'wb') as fid:
#         cPickle.dump(clf, fid)    
#     # load it again
#     with open(os.path.join(modelDir,model+'.pkl'), 'rb') as fid:
#         clf_loaded = cPickle.load(fid)   
#     predicted_label = clf_loaded.predict(dataTest)
#     true_predictions = 0
#     for i in range(len(labelsTest)):
#         print 'Given Label = '+str(labelsTest[i])
#         print 'predicted Label2 =' + str(predicted_label[i])
#         if (labelsTest[i]==predicted_label[i]):
#             true_predictions = true_predictions +1
#     accuracy = float(true_predictions)/float(len(labelsTest))
#     print 'accuracy is :', str(accuracy)

def loadImage():
    image = '/Users/Jatin/Copy/zion/DSC_0412.JPG'
    frame = cv2.imread(image)
    newFrame = cv2.resize(frame,(200,100))
#cv2.imshow('jazz',frame)
    left = numpy.array([])
    right = numpy.array([])
    x = brivasmodule.detect(newFrame,image)
    

def train():
    writePathToSamples(leftEyePath)
    writePathToSamples(rightEyePath)
    leftData = loadData(leftEyePath)
    rightData = loadData(rightEyePath)
    leftFeatures, leftLabels = extract_features(leftData)
    rightFeatures, rightLabels = extract_features(rightData)
    #classify_svm(leftFeatures,leftLabels,'left')
    print "-----------------------Classification for Left eye ---------------------------\n"
    classify_svm(leftFeatures, leftLabels, 'left')
    #classify_rfc(leftFeatures,leftLabels,'left')
    print "-----------------------Classification for Right eye ---------------------------\n"
    classify_svm(rightFeatures,rightLabels,'right')
    #classify_rfc(rightFeatures,rightLabels,'right')

def test():
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        print 'camera is ready to capture'
    else:
        print 'can not open camera'
        exit()
    rval,frame = vc.read()
    cv2.namedWindow("camera",cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow("Eyes",cv2.CV_WINDOW_AUTOSIZE)
    
    leftModelDir = os.path.join(projectPath,'left')
    rightModelDir = os.path.join(projectPath,'right')

    with open(os.path.join(leftModelDir,'left_svm.pkl'), 'rb') as fid:
        leftClf = cPickle.load(fid)
    
    with open(os.path.join(rightModelDir,'right_svm.pkl'), 'rb') as fid:
        rightClf = cPickle.load(fid) 
 
    while True:
        rval, frame = vc.read()
        if frame is not None:
            cv2.imshow("camera",frame)
            eyes = brivasmodule.detect(frame,"None")
            
            if eyes is not None:
                cv2.imshow("Eyes",eyes)
                leftEye = eyes[0:48,:]
                rightEye = eyes[49:,:]
                leftEyeFeature = get_lbp_feature(leftEye,False)
                rightEyeFeature = get_lbp_feature(rightEye, False)
                print leftClf.predict(leftEyeFeature)
                print rightClf.predict(rightEyeFeature)
        else:
            print "no frame"
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
def addArtificialData():
    print "here"
    baseName = os.path.basename(leftEyePath)
    print baseName
    data_dir = os.path.join(projectPath,baseName)
    print data_dir
    files = os.listdir(data_dir)
    files = [f for f in files if f.split('.')[-1]=='txt']
    print files
    data = []
    for f in files:
        label = f.split('.')[0]
        filePath = os.path.join(data_dir,f)
        with open(filePath,'r') as r:
            for image in r:
                data.append(image.strip())
    #print data
    for f in data:
        parentDir =  os.path.dirname(f)
        image_name = f.split('/')[-1].split('.')[0]
        scale_image_name = os.path.join(parentDir,image_name+'_s.jpg')
        roate_image_name = os.path.join(parentDir,image_name+'_r.jpg')
        print image_name
        img = io.imread(f,as_grey=True)
        scale_image = rescale(img,0.9)
        rotated_image = rotate(img,5,resize=False)
        print img.shape
        print scale_image.shape
        print rotated_image.shape
        io.imsave(scale_image_name,scale_image)
        io.imsave(roate_image_name,rotated_image)
        raw_input()


if __name__ == '__main__':
    #addArtificialData()
    #train()
    test()
    #loadImage()
    
    
    
    