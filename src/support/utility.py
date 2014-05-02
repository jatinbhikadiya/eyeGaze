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