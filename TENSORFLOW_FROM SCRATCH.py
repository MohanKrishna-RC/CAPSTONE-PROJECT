
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import imgaug as ia
import os, sys
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import math
from sklearn.metrics import confusion_matrix


# # Path to read images

# In[2]:



train_path = '/home/mohan/Downloads/CAPSTON_PROJECT1/augmented_img/'
test_path = '/home/mohan/Downloads/CAPSTON_PROJECT1/test/'

labels = pd.read_csv("/home/mohan/Downloads/CAPSTON_PROJECT1/labels.csv")
breed_classify1 = pd.read_csv("/home/mohan/Downloads/CAPSTON_PROJECT1/breed_classify1.csv")
breed_classify2 = pd.read_csv("/home/mohan/Downloads/CAPSTON_PROJECT1/breed_classify2.csv")
breed_classify3 = pd.read_csv("/home/mohan/Downloads/CAPSTON_PROJECT1/breed_classify3.csv")
Labels = pd.concat([labels,breed_classify1,breed_classify2,breed_classify3])

#Labels = pd.read_csv("/home/mohan/Downloads/CAPSTON_PROJECT1/labels.csv")

print(Labels.shape)
print ('The train data has {} images.'.format(Labels.shape[0]))
print(Labels.columns)

print("Total number of unique labels: " + str(Labels['breed'].nunique()))
print("Total count of each category") 

#os.system('tensorboard --logdir=/home/mohan/Downloads/CAPSTON_PROJECT1 --port 6006')


# In[3]:


# Counting each breed of dogs
print(Labels["breed"].value_counts())

id = Labels['id']
print(id)


print(Labels["breed"].value_counts())
Labels["breed"] = Labels["breed"].astype('category')

# Integer encoding
Labels["breed"] = Labels["breed"].cat.codes   

# onehot encoding 
y_onehot = pd.get_dummies(Labels['breed'])
print(y_onehot)


# # Read imges into arrays

# In[4]:


def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (200, 200))
    return(img)
train_labels = Labels['breed'].values
print(" Names of all the breeds of dogs")
print(train_labels)


# # Building neural network and assigning parameters

# In[5]:


learning_rate = 0.005
epochs = 10
batch_size = 128
display_step = 10

# Network Parameters
img_height = 200 
img_width = 200
num_classes = 120 
dropout = 0.5 

X = tf.placeholder(tf.float32, [None, img_height,img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# In[6]:


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# # weights & bias for layers

# In[7]:



weights = {
    # 5x5 conv, 1 input, 32 outputs
    'w1': tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'w2': tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.1)),
        # 5x5 conv, 32 inputs, 64 outputs
    'w3': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wf1': tf.Variable(tf.random_normal([25*25*64, 1024],stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes],stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.0, shape=[32])),
    'b2': tf.Variable(tf.constant(0.0, shape=[64])),
    'b3': tf.Variable(tf.constant(0.0, shape=[64])),
    'bf1': tf.Variable(tf.constant(0.0, shape=[1024])),
    'out1': tf.Variable(tf.constant(0.0, shape=[num_classes]))
}


# # Creating model

# In[8]:


def conv_net(x, weights, biases, dropout):

    # Convolution Layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wf1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), biases['bf1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out1'])
    return out


# # Construct the model and evaluate it

# In[9]:


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


# # Define loss and optimizer

# In[10]:


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)


# # Evaluating the model

# In[11]:


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[12]:


tf.summary.scalar('loss', loss_op)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()


# # Initialize the variables (assign their default value)

# In[13]:


init = tf.global_variables_initializer()


# # save and restore all the variables.

# In[14]:


save_file = './saved_models1/tensorflow_model_0.ckpt' 
saver = tf.train.Saver()

writer = tf.summary.FileWriter('/home/mohan/Downloads/CAPSTON_PROJECT1', tf.get_default_graph())


# In[15]:


cd


# # convert image into an array of pixels

# In[16]:


def input(image_path,h,b):
    
    train_data=[]   
    for img in Labels['id'][h:b]:
        train_data.append(read_img(train_path + '{}.jpg'.format(img)))
        #plt.imshow(read_image(train_path + '{}.jpg'.format(img)))
    train_data = np.array(train_data)
    
    #ta.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2],train_data.shape[3])
    print(train_data.shape)
    
    y = y_onehot[h:b]
    print(y.shape)
    return(train_data,y)
 


# In[17]:


# def input(image_path,h,b):
    
#     train_data=[]   
#     for img in Labels['id'][h:b]:
#         train_data.append(read_img(train_path + '{}.jpg'.format(img)))
#         #plt.imshow(read_image(train_path + '{}.jpg'.format(img)))
#     train_data = np.array(train_data)
    
#     #ta.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2],train_data.shape[3])
#     print(train_data.shape)
    
#     y = y_onehot[h:b]
#     print(y.shape)
#     return(train_data,y)
 


# # Training the model

# In[19]:


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    total_loss = []

    for epoch in range(epochs):
        h = 0
        b = batch_size
        losses = 0
        for i in range(int(len(Labels['id'])/batch_size)):
            print(h)
            print(b)

            batch_x, batch_y = input(train_path,h,b)
             
            _, loss, acc = sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            
            h = b
            b += batch_size
            
            if b >= 35000 and b<= 35000:
                break 
                
            losses += loss 
            
            print("Epoch- " + str(epoch) + ", Batch- " + str(i) + ", Mini_batch_Loss = " +      "{:.4f}".format(loss) + ", Training_Accuracy = " +         "{:.3f}".format(acc))
            
        print("Epoch Number" + str(epoch) + ", Minibatch Loss= " +   "{:.4f}".format(loss) + ", Training Accuracy= " +  "{:.3f}".format(acc))    
        total_loss.append(losses)
        
        save_path = saver.save(sess, save_file)
    print("Model saved as {}".format(save_file))
    print("Optimization Finished!")
writer.close() 

best_trained_model_path = './saved_models1/tensorflow_model_0.ckpt' 

# Uncomment this to use your own trained model instead
save_file = best_trained_model_path
    
 print("Optimization Finished!")
    # Calculate accuracy for 256 MNIST test images
    batch_X, batch_Y = input(train_path,30100,37522)
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: batch_X,Y: batch_Y,keep_prob: .75}))
    
# #with tf.Session() as sess:
#     saver.restore(sess, save_file)
#     output = sess.run(accuracy, feed_dict={x:batch_X, y:batch_Y, keep_prob:1.0 })
#     print("Final Accuracy :{}".format(output*100))
#     # Both ways are equivalent
#     print("Accuracy Eval:{}".format(accuracy.eval(feed_dict={x:batch_X, y:batch_Y, keep_prob:1.0 }) * 100 ))

    #print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: batch_X,
  #                                    Y: batch_Y,
#                                      keep_prob: .75}))
    


# # Plotting the required outputs

# In[ ]:


plt.figure()
plt.plot(total_loss, np.arange(epochs), label = 'Loss Curve')
plt.legend()
plt.show()

