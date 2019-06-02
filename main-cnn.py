from __future__ import print_function
from matplotlib import pyplot as plt
from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1671)

print('Reading train 60000.cdb ...')
x_train, y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                    images_height=28,
                                    images_width=28,
                                    one_hot=False,
                                    reshape=True)

print('Reading test 20000.cdb ...')
x_test, y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                                images_height=28,
                                images_width=28,
                                one_hot=False,
                                reshape=True)



# NB_EPOCH = 3
# BATCH_SIZE = 128
# VERBOSE = 1
# NB_CLASSES = 10
# OPTIMIZER = Adam()
# N_HIDDEN = 128
# VALIDATION_SPLIT = 0.2
# RESHAPE = 784
# DROPOUT = 0.2
# IMG_ROW, IMG_COL = 28, 28

nb_classes = 10

INPUT_SHAPE = (1, 28,28)

K.set_image_dim_ordering("th")
x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(20000, 1, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test  /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3, 3)))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Activation('softmax'))
# model.summary()


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3, 3)))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=100, epochs=3)
score = model.evaluate(x_test, y_test)

print('Test Score:', score[0])
print('Test accuracy:', score[1])



#/////////////////////////////////////////////#

predicted_classes = model.predict_classes(x_test)

tp =[0 for i in range(10)]
tn =[0 for i in range(10)]
fp =[0 for i in range(10)]
fn =[0 for i in range(10)]

for i in range(20000):
	z=0
	p=predicted_classes[i]
	for j in range(10):
		if(y_test[i][j]==1):
			z=j
	if(z==p):
		tp[z]+=1
		for j in range(10):
			if(z!=j):
				tn[j]+=1
	else:
		fp[p]+=1
		for j in range(10):
			if(z==j):
				fn[j]+=1
			elif(p==j):
				fp[j]+=1
			else:
				tn[j]+=1

predicted_classes=model.predict_classes(x_train)

for i in range(60000):
	z=0
	p=predicted_classes[i]
	for j in range(10):
		if(y_train[i][j]==1):
			z=j
	if(z==p):
		tp[z]+=1
		for j in range(10):
			if(z!=j):
				tn[j]+=1
	else:
		fp[p]+=1
		for j in range(10):
			if(z==j):
				fn[j]+=1
			elif(p==j):
				fp[j]+=1
			else:
				tn[j]+=1

'''
f1=2tp/(2tp+fp+fn)
recall=tp/(tp+fn)
precision=tp/(tp+fp)

part_b = [,,,tn]
'''
avg_f1=0
avg_re=0
avg_pr=0

for i in range(10):
	if(2*tp[i]+fp[i]+fn[i]==0):
		f1=0
	else:
		f1=2*tp[i]/(2*tp[i]+fp[i]+fn[i])
	if(tp[i]+fn[i]==0):
		recall=0
	else:
		recall=tp[i]/(tp[i]+fn[i])
	if(tp[i]+fp[i]==0):
		precision=0
	else:
		precision=tp[i]/(tp[i]+fp[i])
	avg_f1+=f1
	avg_pr+=precision
	avg_re+=recall
	print("class ",i,": f1=",f1," recall=",recall," precision=",precision)

avg_f1/=10
avg_re/=10
avg_pr/=10
print("total   ",": f1=",avg_f1," recall=",avg_re," precision=",avg_pr)


