# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Input, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import ReLU, Add, ZeroPadding2D



SEED = 777
BATCH_SIZE = 64

FOLDER_DIR = '/content/drive/MyDrive/ML/ResNet-50/'
TRAINING_DIR = '/content/train'
VALIDATION_DIR = '/content/valid'
TESTING_DIR = '/content/test'

tf.random.set_seed(SEED)

training_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                              rescale=1./255,
                                                              width_shift_range= 0.2, 
                                                              height_shift_range= 0.2, 
                                                              zoom_range=0.2, 
                                                              vertical_flip = True,
                                                              rotation_range = 40)
training_dataset = training_gen.flow_from_directory(TRAINING_DIR, 
                                                    class_mode = 'categorical',
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (224, 224), 
                                                    )
validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_dataset = validation_gen.flow_from_directory(VALIDATION_DIR, 
                                                    class_mode = 'categorical',
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (224, 224), 
                                                    )
testing_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testing_dataset = testing_gen.flow_from_directory(TESTING_DIR, 
                                                    class_mode = 'categorical',
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (224, 224), 
                                                    )

class ResNet50(tf.keras.Model) : 
  def __init__(self, input_shape, num_classes) : 
    super().__init__() 
    self.inputs = Input(shape = input_shape)
    self.pad = ZeroPadding2D(3)
    
    self.s1_conv = Conv2D(64, 7, 2)
    self.s1_batch_norm = BatchNormalization()
    self.s1_relu = ReLU()
    self.s1_maxpooling = MaxPooling2D(pool_size = 3, strides = 2)
    
    self.s2_conv = ConvBlock(1, [64, 64, 256], 3)
    self.s2_identity_1 = IdentityBlock([64, 64, 256], 3)
    self.s2_identity_2 = IdentityBlock([64, 64, 256], 3)

    self.s3_conv = ConvBlock(2, [128, 128, 512], 3)
    self.s3_identity_1 = IdentityBlock([128, 128, 512], 3)
    self.s3_identity_2 = IdentityBlock([128, 128, 512], 3)
    self.s3_identity_3 = IdentityBlock([128, 128, 512], 3)

    self.s4_conv = ConvBlock(2, [256, 256, 1024], 3)
    self.s4_identity_1 = IdentityBlock([256, 256, 1024], 3)
    self.s4_identity_2 = IdentityBlock([256, 256, 1024], 3)
    self.s4_identity_3 = IdentityBlock([256, 256, 1024], 3)
    self.s4_identity_4 = IdentityBlock([256, 256, 1024], 3)
    self.s4_identity_5 = IdentityBlock([256, 256, 1024], 3)

    self.s5_conv = ConvBlock(2, [512, 512, 2048], 3)
    self.s5_identity_1 = IdentityBlock([256, 256, 2048], 3)
    self.s5_identity_2 = IdentityBlock([256, 256, 2048], 3)

    self.avg_pool = AveragePooling2D(2)

    self.flatten = Flatten()
    self.classifier = Dense(num_classes, activation='softmax')

  def __call__(self) : 
    inputs = self.inputs
    x = self.pad(inputs)

    x = self.s1_conv(x)
    x = self.s1_batch_norm(x)
    x = self.s1_relu(x)
    x = self.s1_maxpooling(x)

    x = self.s2_conv(x)
    x = self.s2_identity_1(x)
    x = self.s2_identity_2(x)
    x = self.s3_conv(x)
    x = self.s3_identity_1(x)
    x = self.s3_identity_2(x)
    x = self.s3_identity_3(x)

    x = self.s4_conv(x)
    x = self.s4_identity_1(x)
    x = self.s4_identity_2(x)
    x = self.s4_identity_3(x)
    x = self.s4_identity_4(x)
    x = self.s4_identity_5(x)

    x = self.s5_conv(x)
    x = self.s5_identity_1(x)
    x = self.s5_identity_2(x)

    x = self.avg_pool(x)
    x = self.flatten(x)
    classifier = self.classifier(x)
    return tf.keras.Model(inputs = inputs, outputs = classifier)

class IdentityBlock(tf.keras.Model) : 
    def __init__(self, filters, f) : 
      super().__init__()
      self.c1_conv = Conv2D(filters[0], 1, strides = 1, padding = 'valid')
      self.c1_batch_norm = BatchNormalization()
      self.c1_relu = ReLU()

      self.c2_conv = Conv2D(filters[1], f, strides= 1, padding = 'same')
      self.c2_batch_norm = BatchNormalization()
      self.c2_relu = ReLU()

      self.c3_conv =  Conv2D(filters[2], 1, strides=1, padding = 'valid')
      self.c3_batch_norm = BatchNormalization()

      self.fin_relu = ReLU()

    def __call__(self, x) : 
      x_shortcut = x 
      x = self.c1_conv(x)
      x = self.c1_batch_norm(x)
      x = self.c1_relu(x)

      x = self.c2_conv(x)
      x = self.c2_batch_norm(x)
      x = self.c2_relu(x)

      x = self.c3_conv(x)
      x = self.c3_batch_norm(x)

      x = Add()([x_shortcut, x])
      x = self.fin_relu(x)
      return x

class ConvBlock(tf.keras.Model) : 
    def __init__(self, strides, filters, f) :
      super().__init__()
      self.c1_conv = Conv2D(filters[0], 1, strides = strides, padding = 'valid')
      self.c1_batch_norm = BatchNormalization()
      self.c1_relu = ReLU()

      self.c2_conv = Conv2D(filters[1], f, strides = 1, padding = 'same')
      self.c2_batch_norm = BatchNormalization()
      self.c2_relu = ReLU()

      self.c3_conv = Conv2D(filters[2], 1, strides = 1, padding='valid')
      self.c3_batch_norm = BatchNormalization()

      self.sp_conv =  Conv2D(filters[2], 1, strides = strides, padding = 'valid')
      self.sp_batch_norm =  BatchNormalization()

      self.fin_relu = ReLU()


    def __call__(self, x) : 
      x_shortcut = x

      x = self.c1_conv(x)
      x = self.c1_batch_norm(x)
      x = self.c1_relu(x)

      x = self.c2_conv(x)
      x = self.c2_batch_norm(x)
      x = self.c2_relu(x)

      x = self.c3_conv(x)
      x = self.c3_batch_norm(x)

      # shortcut path 
      x_shortcut = self.sp_conv(x_shortcut)
      x_shortcut = self.sp_batch_norm(x_shortcut)

      x = Add()([x_shortcut, x])
      x = self.fin_relu(x)
      return x

class SaveModelCertainEpochs(tf.keras.callbacks.Callback) : 
  def on_epoch_end(self, epoch, logs={}) :
    if epoch % 5 == 0 and epoch > 5:      
      print('Saved in' + FOLDER_DIR + 'model_{}_resnet50.hd5 sucessfully !'.format(epoch))
      self.model.save(FOLDER_DIR + 'model_{}_resnet50.hd5'.format(epoch))
callback_1 = SaveModelCertainEpochs()      
class LearningRateDecay(tf.keras.callbacks.Callback) : # mark
  def on_epoch_begin(self, epoch, logs = {}) : 
    if epoch < 15:
      return 
    else:
      self.model.optimizer.learning_rate = self.model.optimizer.learning_rate * tf.math.exp(-0.1)
callback_2 = LearningRateDecay()
class EarlyStopping(tf.keras.callbacks.Callback) : 
  def on_epoch_end(self, epoch, logs = {}) : 
    if logs['val_accuracy'] > 0.90 : 
      self.model.stop_training = True 
      print('Early Stopping !, learning rate now is {}'.format(self.model.optimizer.learning_rate)) 
callback_3 = EarlyStopping()
class SaveBestModel(tf.keras.callbacks.Callback) : 
  def __init__(self, save_best_metric='val_loss') : 
    self.save_best_metric = save_best_metric
    self.best = float('inf')
  def on_epoch_end(self, epoch, logs={}) : 
    metric_val = logs[self.save_best_metric]
    if metric_val < self.best : 
      self.best = metric_val
      self.best_weights = self.model.get_weights()
callback_4 = SaveBestModel()

EPOCHS = 20
NUM_CLASSES = 300

lrs = [0.0001]
for lr in lrs : 
  model = ResNet50((224, 224, 3), NUM_CLASSES)()
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())
  print(lr)
  history = model.fit(training_dataset,
                      epochs=30, 
                      validation_data=validation_dataset, 
                      verbose = 1,
                      callbacks=[callback_1, callback_2, callback_3, callback_4])

model.set_weights(save_best_model.best_weights)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save(FOLDER_DIR + 'resnet_tune_1_30_lr0_0001')