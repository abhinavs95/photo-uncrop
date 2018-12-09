import cv2
import os

d = os.listdir('results-run3-full/')
for i in d:
	if i[-1]=='g':
		print(i)
		a = cv2.imread('results-run3-full/'+i)
		b = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
		cv2.imwrite('results-run3-full/color-correct/'+i,b)