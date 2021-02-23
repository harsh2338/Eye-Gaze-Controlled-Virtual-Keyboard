from keras.models import Sequential
from keras.layers import Dense,Dropout, Conv2D, Flatten,MaxPooling2D

##First
from keras.models import Sequential
from keras.layers import Dense,Dropout, Conv2D, Flatten,MaxPooling2D
model = Sequential()
model.add(Conv2D(512, kernel_size=3, activation='relu', input_shape=(50,60,1)))
model.add(MaxPooling2D(pool_size=(4,1)))
model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(4,1)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#batch size : 32 ->  90%

#create model
model = Sequential()
#add model layers
model.add(Conv2D(512, kernel_size=3,activation='relu', input_shape=(50,60,1)))
model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1),))

#model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(Conv2D(512, kernel_size=(5,1), strides=(2,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(4,1)))

model.add(Flatten())
model.add(Dense(128, activation='softmax'))
model.add(Dense(2, activation='softmax'))

# model.add(Dropout(.2,input_shape=(2,)))
# model.add(Dense(2, activation='softmax'))

model.summary()


####



model.add(Conv2D(512, kernel_size=3,activation='relu', input_shape=(50,60,1)))
model.add(MaxPooling2D(pool_size=(4,1),strides=4,))
model.add(Conv2D(512, kernel_size=(5,1), strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=(4,1),strides=4,))
model.add(Flatten())
model.add(Dense(4096, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(2))

net = slim.conv2d(input, 512, (3, 86796), 1, padding='VALID', scope='conv_1')
net = slim.max_pool2d(net, (4, 1), 4, padding='VALID', scope='pool_2')
net = slim.conv2d(net, 512, (5, 1), 1, scope='conv_3')
net = slim.max_pool2d(net, (4, 1), 4, padding='VALID', scope='pool_4')
net = slim.flatten(net, scope='flatten_5')
net = slim.fully_connected(net, 4096, scope='fc5')
net = slim.dropout(net, 0.5, scope='dropout6')
net = slim.fully_connected(net, 4096, scope='fc7')
net = slim.dropout(net, 0.5, scope='dropout8')
net = slim.fully_connected(net, 2, activation_fn=None, scope='fc9')






###

model.add(Conv2D(512, kernel_size=5,activation='relu', input_shape=(50,60,1)))
model.add(MaxPooling2D(pool_size=(4,1),strides=4,))
model.add(Conv2D(512, kernel_size=(5,1), strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=(4,1),strides=4,))
model.add(Conv2D(1024, kernel_size=(5,1), strides=1,activation='relu'))
model.add(Flatten())
model.add(Dense(4096, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(2))


net = slim.conv2d(input, 512, (5, 86796), 1, padding='SAME', scope='conv_1')
net = slim.max_pool2d(net, (4, 1), 4, padding='VALID', scope='pool_2')
net = slim.conv2d(net, 512, (5, 1), 1, scope='conv_3')
net = slim.max_pool2d(net, (4, 1), 4, padding='VALID', scope='pool_4')
net = slim.conv2d(net, 1024, (5, 1), 1, scope='conv_4')
net = slim.flatten(net, scope='flatten_5')
net = slim.fully_connected(net, 4096, scope='fc5')
net = slim.dropout(net, keep=0.5, scope='dropout6')
net = slim.fully_connected(net, 4096, scope='fc7')
net = slim.dropout(net, 0.5, scope='dropout8')
net = slim.fully_connected(net, 2, activation_fn=None, scope='fc9')