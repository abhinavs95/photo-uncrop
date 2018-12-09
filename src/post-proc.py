import cv2
import numpy as np

center = (127,127)
for i in range(5440):
	print(i)
	img_res = cv2.imread('color-correct/img_'+str(i)+'.jpg')
	img_ori = cv2.imread('color-correct/img_'+str(i)+'.ori.jpg')
	img_ovr = img_res.copy()
	img_ovr[40:216,40:216] = img_ori[40:216,40:216]
	mask = np.zeros_like(img_res)
	mask[40:216,40:216,:] = 1
	img_crop = img_ori*mask
	img_out = cv2.seamlessClone(img_crop,img_res,255*mask,center,cv2.NORMAL_CLONE)
	cv2.imwrite('post-res/img_'+str(i)+'.ovr.jpg',img_ovr)
	cv2.imwrite('post-res/img_'+str(i)+'.crop.jpg',img_crop)
	cv2.imwrite('post-res/img_'+str(i)+'.post.jpg',img_out)