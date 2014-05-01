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