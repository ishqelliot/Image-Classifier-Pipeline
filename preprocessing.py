from os import listdir
from os.path import isfile, join
import numpy
import cv2
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_SIZE = 224
mypath=['C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\frogs\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\horses\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\doctors\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\men\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\women\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\angry faces\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\buildings\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\shirts\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\mountains\\',
'C:\\Users\\Kanishk\\Documents\\GitHub\\tensorflow_img_cls\\classes\\money\\']

#Images gathered are of different sizes so below funciton converts images to 224 above mentioned size.
def tf_resize_images(X_img_file_paths,mypath):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), 
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#color conversioin is need because opencv reads images to bgr format 
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            cv2.imwrite(str(mypath)+'resize'+str(index)+'.jpg',resized_img)
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

# Below function produces each image at scaling of 90%, 75% and 60% of original image.
def central_scale_images(X_imgs, scales,mypath):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            #cv2.imwrite(str(mypath)+'scaled'+str(index)+'.jpg',scaled_imgs)
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    for index,data in enumerate(X_scale_data):
      cv2.imwrite(str(mypath)+'scale'+str(index)+'.jpg',data)
    return X_scale_data

#Function for rotating image by 90 degree	
def rotate_images(X_imgs,mypath):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype = np.float32)
    for index,data in enumerate(X_rotate):
      cv2.imwrite(str(mypath)+'rotate'+str(index)+'.jpg',data)
    return X_rotate

#Function for flipping images
def flip_images(X_imgs,mypath):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    for index,data in enumerate(X_flip):
      cv2.imwrite(str(mypath)+'flip'+str(index)+'.jpg',data)
    return X_flip

#Function for adding noise to images
def add_salt_pepper_noise(X_imgs,mypath):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for index,X_img in enumerate(X_imgs_copy):
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
        cv2.imwrite(str(mypath)+'saltting'+str(index)+'.jpg',X_img)
    return X_imgs_copy
  
#function calling
for i in range(10):
    onlyfiles = [ f for f in listdir(mypath[i]) if isfile(join(mypath[i],f)) ]
    images = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = join(mypath[i],onlyfiles[n])
    X_data=tf_resize_images(images,mypath[i])
    scaled_imgs = central_scale_images(X_data, [0.90, 0.75, 0.60],mypath[i])
    rotated_imgs = rotate_images(X_data,mypath[i])
    flipped_images = flip_images(X_data,mypath[i])
    salt_pepper_noise_imgs = add_salt_pepper_noise(X_data,mypath[i])

    
