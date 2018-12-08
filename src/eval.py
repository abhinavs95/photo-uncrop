import numpy as np
import cv2
from skimage.measure import compare_mse, compare_psnr

l1_mean = np.zeros((5440))
l2_mean = np.zeros((5440))
p_mean = np.zeros((5440))

for i in range(5440):
	if i%1000==0:
		print(i)
	img_res = cv2.imread('results-run3-full/post-res/img_'+str(i)+'.post.jpg')
	img_ori = cv2.imread('results-run3-full/color-correct/img_'+str(i)+'.ori.jpg')
	psnr = compare_psnr(img_ori, img_res)
	l2 = compare_mse(img_res, img_ori)
	img_res = np.asarray(img_res, dtype=np.float32)
	img_ori = np.asarray(img_ori, dtype=np.float32)
	l1 = np.mean(np.abs(img_res-img_ori))
	p_mean[i] = psnr
	l1_mean[i] = l1
	l2_mean[i] = l2



print(np.mean(l1_mean)/255.0*100.0, np.mean(l2_mean)/(255.0*255.0)*100.0, np.mean(p_mean))