import random
import sys
from PIL import Image
import cv2
import numpy as np
from math import inf
import time

# DECLARATIONS  #

img72 = cv2.imread('image072.png')
img72_padding = cv2.imread('image072_padding.png')  #READ IMAGES
img92 = cv2.imread('image092.png')

height, width, channels = img72.shape

# POUR GRAY IMAGES 

newGrayImg72 = cv2.cvtColor(img72_padding, cv2.COLOR_RGB2YCrCb)[:, :, 0]
grayImg72 = cv2.cvtColor(img72, cv2.COLOR_RGB2YCrCb)[:, :, 0]      #converts the input image from the RGB color space to the YCrCb color space.
grayImg92 = cv2.cvtColor(img92, cv2.COLOR_RGB2YCrCb)[:, :, 0]


#DEBUT#

Nv_Image = np.zeros((height, width), dtype=np.uint8) #pour les entiers de 8 bits

boxSize = 16
blocks92 = []
blocks72 = []


# LES FONCTIONS DE CALCUL #

#FONCTION DE CALCUL MSE 

def MSE(bloc1, bloc2):
    block1, block2 = np.array(bloc1), np.array(bloc2)
    return np.square(np.subtract(block1, block2)).mean()    # MSE FORMULA APPLICATION 

#FONCTION DE RECHERCHE DICHOTOMIQUE 

def dichotomicSearch(bloc, val_i, val_j, mse_minimum):                #valeurs i,j points
    global y, x, resultBloc
    stepToVosin = 32
    Firsti = val_i - stepToVosin - 8 + 64              #############
    Firstj = val_j - stepToVosin - 8 + 64
    while stepToVosin >= 1:
       
        fin = stepToVosin * 2 + boxSize
        for i in range(int(Firsti), int(Firsti) + int(fin), int(stepToVosin)):
            for j in range(int(Firstj), int(Firstj) + int(fin), int(stepToVosin)):
                resultBloc = newGrayImg72[i:i + boxSize, j:j + boxSize]
                loss = MSE(bloc, resultBloc)

                if loss < mse_minimum:
                    mse_minimum = loss
                    x = i        #le i nouveau 
                    y = j        #le j nouveau
                
        stepToVosin /= 2
        Firsti = x
        Firstj = y
    return x, y, mse_minimum, resultBloc

# L'ENCADREMENT DU MEME COULEUR

def ENCADRER(img1, blocks1, img2, blocks2):    #TO ADD RECTANGLES ON THE IMAGE 
                 
    for i in range(len(blocks1)):
        cv2.rectangle(img1, (blocks1[i][0], blocks1[i][1]),
                      (blocks1[i][0] + boxSize, blocks1[i][1] + boxSize), (0, 255, 0), 2)

    for i in range(len(blocks2)):
        cv2.rectangle(img2, (blocks2[i][0], blocks2[i][1]),
                      (blocks2[i][0] + boxSize, blocks2[i][1] + boxSize), (0, 0, 255), 2)


        

#  MAIN  #

if __name__ == '__main__':
    start_time = time.time()
    # PARCOURS MAIN #
    for i in range(0, height - boxSize, boxSize):  
        for j in range(0, width - boxSize, boxSize):  
            bloc92 = grayImg92[i:i + boxSize, j:j + boxSize]
            mse_minimum = inf
            x, y, mse_minimum, bloc = dichotomicSearch(bloc92, i + 8, j + 8, mse_minimum)  
            if mse_minimum > 60:
               
                blocRes = bloc92 - bloc
                Nv_Image[i:i + boxSize, j:j + boxSize] = blocRes
                blocks92.append((y, x))
                blocks72.append((j, i))

    ENCADRER(img92, blocks92, img72, blocks72)
    time = time.time() - start_time
    print(f"{time} seconds")
    imgResu = Image.fromarray(Nv_Image)
    imgResu.save('Resultat.png')
    cv2.imwrite('image072withRedRect.png', img72)  # cv2.show() works
    cv2.imwrite('image092withGreenRect.png', img92)
    sys.exit()