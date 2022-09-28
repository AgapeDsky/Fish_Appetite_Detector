import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np

img = cv.imread('Foto/bank_foto/ikan_kenyang/ikan_kenyang_001.jpg')

# cv.waitKey(0)

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(image=imgray, threshold1=100, threshold2=200) 

cv.imshow('hehe',img)
cv.imshow('duar',imgray)
cv.waitKey(0)