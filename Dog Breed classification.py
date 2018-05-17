import numpy as np
import os
import pandas as pd 
import scipy.io
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#define sigmoid function
def nonlin(x,deriv=False):
    return 1/(1+np.exp(-x))
    
cwd = "C:\\Users\\1997.tanuja\\Downloads\\train"

#Load the img files in the directory
trainfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

#Loading the predicting variable into labels
labels = pd.read_csv("labels/labels.csv")
labels.head()

#Loading images of only top 16 recurring labels
top_breeds = sorted(list(labels['breed'].value_counts().head(16).index))
labels = labels[labels['breed'].isin(top_breeds)]

#View id, label and img path
train = labels.copy()
train['filename'] = train.apply(lambda x: (cwd + x['id'] + '.jpg'), axis=1)
train.head()

#converting our image into a matrix and reshaping as 32 by 32
train_datan = np.array([ img_to_array(load_img(img, target_size=(32, 32))) for img in train['filename'].values.tolist()]).astype('float32')

#multiplication of the dimensions of img with rgb code
x_img_szie = 32*32*3
X = train_datan
x = np.reshape(X, (922,3072))
print(x.shape)

#one hot encoding of target categories to binary
labels = train.breed
targets_series = pd.Series(train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
label = one_hot_labels
labels = np.array(label, np.uint8)
y = labels
print(labels.shape)


#Training the model
np.random.seed(1)
syn0 = 2*np.random.random((3072, 922)) - 1
syn1 = 2*np.random.random((922, 8)) - 1

for iter in range(500):
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l2_error = y - l2
    if (iter % 100) ==0:
        dum = np.abs(np.float32(l2_error))
        dum1 = np.mean(dum, dtype=np.float32)
        print("error:")
        print(dum1)
    l2_delta = l2_error * nonlin(l2,True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,True)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print("Output After Training:")
print(l2)

#decode into integers
class_labels = np.argmax(y, axis=1)
print(class_labels)

#Testing an image against the model
cwd = "C:\\Users\\1997.tanuja\\Downloads\\test"
testfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f)) ]
value = testfiles[0]
pre = cwd + "\\" + value
test_datan = np.array([ img_to_array(load_img(pre, target_size=(32, 32)))]).astype('float32')

t1 = np.reshape(test_datan, (1,3072))
lt1 = nonlin(np.dot(t1,we1))
lt2 = nonlin(np.dot(lt1,we2))
print(lt2)

class_labely = np.argmax(lt2, axis=1)
print(class_labely)
