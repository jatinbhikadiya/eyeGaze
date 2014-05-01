'''
Created on May 1, 2014

@author: Jatin
'''
import os
from support import utility

dataPath = '/Jatin/Brivas/gaze/data'
projectPath = '/Jatin/workspace/eyePupil'
leftEyePath=os.path.join(dataPath,'left')
rightEyePath=os.path.join(dataPath,'right')

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
    data = []
    for f in files:
        label = f.split('.')[0]
        filePath = os.path.join(data_dir,f)
        with open(filePath,'r') as r:
            for image in r:
                data.append([image.strip(),label])
    return data

def classify(data):
    for sample in data:
        print sample[0]
        
    
if __name__ == '__main__':
    #writePathToSamples(leftEyePath)
    #writePathToSamples(rightEyePath)
    leftData = loadData(leftEyePath)
    rightData = loadData(rightEyePath)
    classify(leftData)
    