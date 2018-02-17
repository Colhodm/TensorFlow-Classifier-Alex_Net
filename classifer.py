import tensorflow as tf
dir(tf.contrib)
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
#chkp.print_tensors_in_checkpoint_file("model_epoch10.ckpt", tensor_name='', all_tensors=True)
import os
import cv2
import numpy as np
from alexnet import AlexNet
# initialize input place holder to specific batch_size, e.g. 1 if you want to classify image by image  output_file = open("TESTING123.txt",'w')
output_file = open("TESTING123.txt",'w')
output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
batch_size = 1
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
keep_prob = tf.constant(1., dtype=tf.float32)
imagenet_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'test')
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
imgs = []
groundtruth = []
#importedgraph = tf.train.import_meta_graph("model_epoch10.ckpt.meta")
temp_files = [image_dir +"/"+ f for f in os.listdir(image_dir) if f.endswith('.jpg')]
for f in temp_files:
	if not f.endswith('.jpg'):
		continue
	else:   
		imgs.append(tf.read_file(f))
for f in temp_files:
	if f.endswith('.jpg') and f[-20:].find('image') != -1:
		groundtruth.append(0)
	elif f.endswith('.jpg') and f[-20:].find('image') == -1:
		groundtruth.append(1)
	else:
		continue
# Initialize model
model = AlexNet(x, keep_prob, 2 , [])

# Link variable to model output
score = model.fc8
softmax = tf.nn.softmax(score)
label = tf.placeholder(tf.float32,None,name='label')
prediction_list = tf.placeholder(tf.float32,None,name='prediction_list')
accuracy = tf.metrics.accuracy(label,prediction_list)
auc = tf.metrics.auc(label,prediction_list)
#confusion = tf.confusion_matrix(label,prediction_list) 
# create saver instance
saver = tf.train.Saver()
predictions = []
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, 'model_epoch50.ckpt')

    for i,image in enumerate(imgs):
    	img_decoded = tf.image.decode_jpeg(image, channels=3)
	img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized,imagenet_mean)
	img_bgr = img_centered[:, :, ::-1]
        # Reshape as needed to feed into model
        img = tf.reshape(img_bgr,(1,227,227,3))
        pred = sess.run(softmax, feed_dict={x: sess.run(img)})
        predicted_label = pred.argmax(axis=1)
	predictions.append(predicted_label[0])
	output_file.write(str(predicted_label[0]) + "," + str(groundtruth[i]) + ", "+ str(int(predicted_label[0] == groundtruth[i]))  +  '\n')
 #   output_file.write(str(sess.run(confusion)))
    accuracy = sess.run(accuracy,feed_dict={label:groundtruth,prediction_list:predictions})
    auc = sess.run(auc,feed_dict={label:groundtruth,prediction_list:predictions})
    output_file.write("accuracy was " + str(accuracy) + ", " + "AUC was " + str(auc))
output_file.close()        
