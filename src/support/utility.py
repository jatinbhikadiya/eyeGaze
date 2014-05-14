'''
Created on May 1, 2014

@author: Jatin
'''
import os

def checkDirectory(dir_path):
    '''
        Checks to see if the directory exists. If it does not exist then the directory
        is created with full permissions for user and group.
        @param dir_path: full path to the directory
        '''
    if not os.path.isdir(dir_path):
        os.umask(0) # reset the umask back to 0
        os.makedirs(dir_path,0770)

def getSubdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
    
def extract_patches(image):
    block_size = 16
    patches=[]
    width = image.shape[1]
    #print width
    height = image.shape[0]
    #print height
    grid_x = int(width/16.0)
    grid_y = int(height/16.0)
    for i in range(grid_y):
        for j in range(grid_x):
            x=j*block_size
            #print x
            y=i*block_size
            patch = image[int(y):int(y+16),int(x):int(x+16)]
            #print patch.shape
            patches.append(patch)
        #else :
            #print p.x
    return patches

def change_label(label,no_of_labels = 3):
    all_labels = []
    for i in range(1,10):
        all_labels.append('block_'+str(i))
    if no_of_labels==3:
        if label in all_labels[0:3]:
            label = 'left'
        elif label in all_labels[3:6]:
            label = 'center'
        elif label in all_labels[6:9]:
            label = 'right'
        else : print 'label not found'    
    if no_of_labels==5:
        if label in all_labels[0:2]:
            label = 'left_most'
        elif label in all_labels[2:4]:
            label = 'left'
        elif label in all_labels[4:5]:
            label = 'center'
        elif label in all_labels[5:7]:
            label = 'right'
        elif label in all_labels[7:9]:
            label = 'right_most'
        else : print 'label not found'
    return label

        