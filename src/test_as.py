import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model_as import *
from util_as import *


batch_size = 64

# size of overlap of output patch
overlap_size = 7
# size of output patch
hiding_size = 256
# size of border
border_size = 40


testset_path  = '../data/places_testset.pickle'
result_path= '../results/test/'
pretrained_model_path = '../models/places/model-50'
testset = pd.read_pickle( testset_path )
is_train = tf.placeholder( tf.bool )

# input image placeholder
images_tf = tf.placeholder( tf.float32, [batch_size, 256, 256, 3], name="images")
images_hiding = tf.placeholder( tf.float32, [batch_size, hiding_size, hiding_size, 3], name='images_hiding')


# load model
model = Model()

# main network
bn1, bn2, bn3, bn4, bn5, bn6, debn6, debn5, debn4, debn3, debn2, reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=100)
tf.initialize_all_variables().run()
saver.restore( sess, pretrained_model_path )

ii = 0
for start,end in zip(
        range(0, len(testset), batch_size),
        range(batch_size, len(testset), batch_size)):

    test_image_paths = testset[start:end]['image_path'].values
    test_images_ori = map(lambda x: load_image(x), test_image_paths)

    test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
    test_images, test_crops, xs,ys = zip(*test_images_crop)

    reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn6_val,debn5_val,debn4_val, debn3_val, debn2_val = sess.run(
            [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn6,debn5,debn4, debn3, debn2],
            feed_dict={
                images_tf: test_images,
                images_hiding: test_crops,
                is_train: False
                })

    for rec_val, img, img_ori in zip(reconstruction_vals, test_images, test_images_ori):
        rec_hid = (255. * (rec_val+1)/2.).astype(int)
        rec_con = (255. * (img+1)/2.).astype(int)
        img_ori = (255. * (img_ori+1)/2.).astype(int)

        rec_hid[border_size : hiding_size - border_size, border_size : hiding_size - border_size] = rec_con[border_size : hiding_size - border_size, border_size : hiding_size - border_size]
        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.jpg'), rec_hid)
        cv2.imwrite(os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), img_ori)
        ii += 1





