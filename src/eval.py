import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#IMAGE SIZE THAT IS PASSED SHOULD BE EQUAL

def dist1(img1,img2):
    return np.sum(np.abs(img1-img2))

def dist2(img1,img2):
    return np.sqrt(np.sum((img1-img2)**2))

def psnr(img1,img2):
    mse = np.mean((img1-img2)**2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

lst = os.listdir('test2/')
lst.sort()

lst_final = []
lst_l1 = []
lst_l2 = []
lst_psnr = []

for i in range(0,len(lst),2):
    img = cv2.imread('test2/'+lst[i])
    img_ori = cv2.imread('test2/'+lst[i+1])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_ori = cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
    dis1 = dist1(img,img_ori)/(256*256)
    dis2 = dist2(img,img_ori)/(256*256)
    ps = psnr(img,img_ori)
    lst_l1.append(dis1)
    lst_l2.append(dis2)
    lst_psnr.append(ps)
    lst_final.append([dis1,dis2,ps])
"""
with open('eval.txt', 'w') as f:
    for item in lst_final:
        f.write("%s\n" % item)
"""
dist1_mean = np.mean(lst_l1)
dist2_mean = np.mean(lst_l2)
psnr_mean = np.mean(lst_psnr)
