import glob
import os  
import shutil  
from pathlib import Path
import ntpath
import cv2
import numpy as np
import png
import cv2
import shutil 
import random
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from datetime import datetime
import argparse
import sys

def mainParser(args=sys.argv[1:]):

    # Parser definition
    parser = argparse.ArgumentParser(description="Parses command.")

    # Parser Options
    parser.add_argument("-r", "--sourcePath", help="Path to the test datasets")
    parser.add_argument("-t", "--targetPath", help="Path to the model outputs")
    parser.add_argument("-p", "--patchSize", type=int, default=256, help="Set patch size")

    options = parser.parse_args(args)
    return options

def extractFileName(path, withoutExtension = None):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)

    if withoutExtension:
        return tail.split(".")[-2] or ntpath.basename(head).split(".")[-2]

    return tail or ntpath.basename(head)


def createDir(path):
    # Create a directory to save processed samples
    Path(path).mkdir(parents=True, exist_ok=True)
    return True

def imageList(path, multiDir = False, imageExtension =['*.jpg', '*.png', '*.jpeg', '*.tif', '*.npy']):
    #types = () # the tuple of file types
    imageList = []
    for ext in imageExtension:

        if multiDir == True:
            imageList.extend(glob.glob(path+"*/"+ext))
        else:
            imageList.extend(glob.glob(path+ext))
        
        imageList
    return imageList


def imread_uint16_png(image_path, alignratio_path):
    """ This function loads a uint16 png image from the specified path and restore its original image range with
    the ratio stored in the specified alignratio.npy respective path.


    Args:
        image_path (str): Path to the uint16 png image
        alignratio_path (str): Path to the alignratio.npy file corresponding to the image

    Returns:
        np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    #print("Read Max: ", align_ratio)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio

def imwrite_uint16_png(image_path, image, alignratio_path):
    """ This function writes the hdr image as a uint16 png and stores its related align_ratio value in the specified paths.

        Args:
            image_path (str): Write path to the uint16 png image (needs to finish in .png, e.g. 0000.png)
            image (np.ndarray): HDR image in float format.
            alignratio_path (str): Write path to the align_ratio value (needs to finish in .npy, e.g. 0000_alignratio.npy)

        Returns:
            np.ndarray (np.float32, (h,w,3)): Returns the RGB HDR image specified in image_path.

    """
    align_ratio = (2 ** 16 - 1) / image.max()
    #print("Write Max: ",align_ratio)
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR)






class visual16to8:
    def __init__(self, rootDir, targetDir, patchSize = 256 ):
        self.targetDir = targetDir
        self.rootDir = rootDir
        
        self.targetDir = targetDir
        createDir(self.targetDir)
        self.rootDir = rootDir

        #gamma=2.24

        self.patchSize = patchSize
    def patchExtract (self):
        images = imageList(self.rootDir, multiDir=True)
        print(len(images))
        countT = 0
        for im in images:
            if '.png' in im:
                # Path Defination
                gtPath = im
                alignPath = im.replace(".png", '_alignratio.npy')
                # Read Images
                imageGT16 = imread_uint16_png(gtPath, alignPath)
                #file_name = os.path.splitext(im)[0]
                (filepath, tempfilename) = os.path.split(im)
                (filename, extension) = os.path.splitext(tempfilename)
                #targetCounter = "{:05d}".format(file_name) + "_"
                #print(filename)
                cv2.imwrite(self.targetDir+filename + "_gt8.png", cv2.cvtColor(imageGT16, cv2.COLOR_RGB2BGR) *255 )
                countT += 1
                if countT % 100 == 0:
                    print(countT)

                
   
    def __call__(self):
        self.patchExtract()
        #self.modelEvaluation()

if __name__ == "__main__":

    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()

    if options.sourcePath:
        sourcePath = options.sourcePath
    if options.targetPath:
        targetPath = options.targetPath

    if options.patchSize:
        patchSize = options.patchSize
    else:
        patchSize = ""
    
    
    visual8 = visual16to8(sourcePath, targetPath, patchSize)
    visual8()