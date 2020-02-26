###2 kinds of structure, first one is a sample, the next one is structure published by google

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Dropout, Flatten, add
from tensorflow.keras.regularizers import  l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2


class Inception_Resnetv2():
    
    def __init__(self, input_shape = (299,299,3), output_units = 1001,
                 
                 regularizer = l2(1e-4), initializer = "he_normal", init_maxpooling = True, include_top = True):
        
        super(Inception_Resnetv2, self).__init__()
        self.input_shape = input_shape
        self.output_units = output_units
        self.regularizer = regularizer
        self.initializer = initializer
        self.init_maxpooling = init_maxpooling
        self.include_top = include_top
        
        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1
        
    def conv2d_bn(self, x, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False, activation_use = 'relu', use_bn = True, use_activation = True):
        

        x = Conv2D(nb_filter, (num_row, num_col),
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   kernel_regularizer=regularizers.l2(0.00004),
                   kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
        
        if use_bn:
            x = BatchNormalization(axis=self.channel_axis, momentum=0.9997, scale=False)(x)
        if use_activation:
            x = Activation(activation_use)(x)
        return x
    
    # a model sample
    def build_model(self):
        
        input_x = Input(self.input_shape)
        #299x299x3    
        
        net = self.conv2d_bn(input_x, 32, 3, 3, strides=(2,2), padding='valid')
        #149x149x32
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid')
        #147x147x32
        net = self.conv2d_bn(net, 64, 3, 3)
        #147x147x64
        
        branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid')
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #73x73x160
        
        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid')
        
        
        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='valid')
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #71x71x192
        
        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid')
        branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #35x35x384
        
        # 5 x Inception-reasnet-A blocks
        
        for idx in range(5):
    	    net = self.block_inception_a(net)
        
        # Reduction-A block
        
        net = self.block_reduction_a(net)
        # 17 x 17 x 1152
        
        for idx in range(10):
            net = self.block_inception_b(net)
            
        net = self.block_reduction_b(net)
        #8x8x2144
        
        for idx in range(5):
    	    net = self.block_inception_c(net)
        
        if self.include_top:
            
            # 1 x 1 x 2144
            net = AveragePooling2D((8,8), padding='valid')(net)
            net = Dropout(0.2)(net)
            net = Flatten()(net)
            net = Dense(units=self.output_units, activation='softmax')(net)
            
        inception_resnetv2_model = Model(inputs = [input_x], outputs = [net], name='inception_v4')
        return inception_resnetv2_model
        
        
        
    def block_inception_a(self, input_incepa, scale = 1.0):
        
        
        branch_0 = self.conv2d_bn(input_incepa, 32, 1, 1)
    
        branch_1 = self.conv2d_bn(input_incepa, 32, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 32, 3, 3)
    
        branch_2 = self.conv2d_bn(input_incepa, 32, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 48, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 64, 3, 3)
        
        mixed = concatenate([branch_0, branch_1, branch_2], axis=self.channel_axis)
        
        x1 = self.conv2d_bn(mixed, input_incepa.get_shape()[3], 1, 1, activation_use='linear')
        
        x1 = x1*scale
        out = add([x1, input_incepa])
        
        out = BatchNormalization(axis=self.channel_axis)(out)
        
        out = Activation("relu")(out)
    
        return out
    
    
        
    def block_inception_b(self, input_incepb, scale = 1.0):
            
        
        branch_0 = self.conv2d_bn(input_incepb, 192, 1, 1)
        
        
        branch_1 = self.conv2d_bn(input_incepb, 128, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 160, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 192, 7, 1)
        
        x = concatenate([branch_0, branch_1], axis=self.channel_axis)
        
        x1 = self.conv2d_bn(x, input_incepb.get_shape()[3], 1, 1, activation_use='linear')
        #1152 = 384(input_dimension) + 384 + 384
        
        x1 = x1*scale
        out = add([x1, input_incepa])
        
        out = BatchNormalization(axis=self.channel_axis)(out)
        
        out = Activation("relu")(out)
        
        return out
    
    
    
    def block_inception_c(self, input_incepc, scale = 1.0):
            
        
        branch_0 = self.conv2d_bn(input_incepc, 192, 1, 1)
        
        
        branch_1 = self.conv2d_bn(input_incepc, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 256, 7, 1)
        
        x1 = concatenate([branch_0, branch_1], axis=self.channel_axis)
        
        out = self.conv2d_bn(x1, input_incepc.get_shape()[3], 1, 1, activation_use='linear')
        
        x1 = x1*scale
        out = add([x1, input_incepa])
        
        out = BatchNormalization(axis=self.channel_axis)(out)
        
        out = Activation("relu")(out)
        
        return out
    
    
    
    def block_reduction_a(self, input_reda):
        
            
        branch_0 = self.conv2d_bn(input_reda, 384, 3, 3, strides=(2,2), padding='valid')
    
        branch_1 = self.conv2d_bn(input_reda, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 3)
        branch_1 = self.conv2d_bn(branch_1, 384, 3, 3, strides=(2,2), padding='valid')
    
        branch_2 = MaxPooling2D((3,3), strides=(2,2))(input_reda)
        
        x = concatenate([branch_0, branch_1, branch_2], axis=self.channel_axis)
        
        return x
    
    
    
    def block_reduction_b(self, input_redb):
        
        branch_0 = MaxPooling2D((3,3), strides=(2,2), padding = 'valid')(input_redb)
        
        branch_1 = self.conv2d_bn(input_redb, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 384, 3, 3, strides=(2,2), padding = 'valid')
        
        branch_2 = self.conv2d_bn(input_redb, 256, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 288, 3, 3, strides=(2,2), padding = 'valid')
     
        branch_3 = self.conv2d_bn(input_redb, 256, 1, 1)
        branch_3 = self.conv2d_bn(branch_3, 288, 3, 3)
        branch_3 = self.conv2d_bn(branch_3, 320, 3, 3, strides=(2,2), padding = 'valid')
    
        
        
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation('relu')(x)
        return x
    
    
    #for training
    def build_inception_resnetv2_model(self, input_x):
            
        
        net = self.conv2d_bn(input_x, 32, 3, 3, strides=(2,2), padding='valid')
        #149x149x32
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid')
        #147x147x32
        net = self.conv2d_bn(net, 64, 3, 3)
        #147x147x64
        
        branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid')
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #73x73x160
        
        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid')
        
        
        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='valid')
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #71x71x192
        
        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid')
        branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        
        
        net = concatenate([branch_0, branch_1], axis=self.channel_axis)
        #35x35x384
        
        # 5 x Inception-reasnet-A blocks
        
        for idx in range(5):
    	    net = self.block_inception_a(net)
        
        # Reduction-A block
        
        net = self.block_reduction_a(net)
        # 17 x 17 x 1152
        
        for idx in range(10):
            net = self.block_inception_b(net)
            
        net = self.block_reduction_b(net)
        #8x8x2144
        
        for idx in range(5):
    	    net = self.block_inception_c(net)
        
        if self.include_top:
            
            # 1 x 1 x 2144
            net = AveragePooling2D((8,8), padding='valid')(net)
            net = Dropout(0.2)(net)
            net = Flatten()(net)
            net = Dense(units=self.output_units, activation='softmax')(net)
            
        inception_resnetv2_model = Model(inputs = [input_x], outputs = [net], name='inception_v4')
        return inception_resnetv2_model
    
    
    
    
    
    #structure published by google research
    def build_orignial_model(self):
        
        input_x = Input(self.input_shape)
        #input: 299x299x3
        
        net = self.conv2d_bn(input_x, 32, 3, 3, strides=(2,2), padding='valid')
        #149x149x32
        
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid')
        #147x147x32
        
        net = self.conv2d_bn(net, 64, 3, 3)
        #147x147x64
        
        net = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        #73x73x64
        
        net = self.conv2d_bn(net, 80, 1, 1)
        #73x73x80
        
        net = self.conv2d_bn(net, 192, 3, 3, padding='valid')
        #71x71x192
        
        net = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        #35x35x192
        
        branch_0 = self.conv2d_bn(net, 96, 1, 1)
    
        branch_1 = self.conv2d_bn(net, 48, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 5, 5)
    
        branch_2 = self.conv2d_bn(net, 64, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        
        branch_3 = AveragePooling2D((3, 3), strides=1, padding='same')(net)
        
        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        #35x35x320
        
        for i in range(10):
            net = self.block_inception_a_scale(net, scale = 0.17)
        
        
        branch_0 = self.conv2d_bn(net, 256, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 256, 3, 3)
        branch_0 = self.conv2d_bn(branch_0, 384, 3, 3, strides = 2, padding = 'valid')
        
        branch_1 = self.conv2d_bn(net, 384, 3, 3, strides = 2, padding = 'valid')
        
        branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        
        net = concatenate([branch_0, branch_1, branch_2], axis=self.channel_axis)
        #17x17x1088
        
        for i in range(20):
            net = self.block_inception_b_scale(net, scale = 0.10)
        
        
        
        branch_0 = self.conv2d_bn(net, 256, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 384, 3, 3, strides = 2, padding = 'valid')
        
        
        branch_1 = self.conv2d_bn(net, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 288, 3, 3, strides = 2, padding = 'valid')
        
        branch_2 = self.conv2d_bn(net, 256, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 288, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 320, 3, 3, strides = 2, padding = 'valid')
                
        branch_3 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        
        net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self.channel_axis)
        #8x8x2080
        
        for i in range(9):
            net = self.block_inception_c_scale(net, scale = 0.20)
        
        
        if self.include_top:
            
            # 1 x 1 x 2144
            net = AveragePooling2D((8,8), padding='valid')(net)
            net = Dropout(0.2)(net)
            net = Flatten()(net)
            net = Dense(units=self.output_units, activation='softmax')(net)
            
        inception_resnetv2_model = Model(inputs = [input_x], outputs = [net], name='inception_v4')
        return inception_resnetv2_model
        
        
    def block_inception_a_scale(self, input_incepa, scale = 1.0):
        
        
        branch_0 = self.conv2d_bn(input_incepa, 32, 1, 1)
    
        branch_1 = self.conv2d_bn(input_incepa, 32, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 32, 3, 3)
    
        branch_2 = self.conv2d_bn(input_incepa, 32, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 48, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 64, 3, 3)
        
        mixed = concatenate([branch_0, branch_1, branch_2], axis=self.channel_axis)
        
        x1 = self.conv2d_bn(mixed, input_incepa.get_shape()[3], 1, 1, use_bn = False, activation_use='linear')
        
        scaled_up = x1 * scale
        
        out = add([scaled_up, input_incepa])
        
        out = Activation("relu")(out)
    
        return out
    
    
    def block_inception_b_scale(self, input_incepa, scale = 1.0):
        
        
        branch_0 = self.conv2d_bn(input_incepa, 192, 1, 1)
    
        branch_1 = self.conv2d_bn(input_incepa, 128, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 192, 7, 1)
        
        mixed = concatenate([branch_0, branch_1], axis=self.channel_axis)
        
        x1 = self.conv2d_bn(mixed, input_incepa.get_shape()[3], 1, 1, use_bn = False, activation_use='linear')
        
        scaled_up = x1 * scale
        
        out = add([scaled_up, input_incepa])
        
        out = Activation("relu")(out)
    
        return out
    
    
    def block_inception_c_scale(self, input_incepa, scale = 1.0):
        
        
        branch_0 = self.conv2d_bn(input_incepa, 192, 1, 1)
    
        branch_1 = self.conv2d_bn(input_incepa, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 3)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 1)
        
        mixed = concatenate([branch_0, branch_1], axis=self.channel_axis)
        
        x1 = self.conv2d_bn(mixed, input_incepa.get_shape()[3], 1, 1, use_bn = False, activation_use='linear')
        
        scaled_up = x1 * scale
        
        out = add([scaled_up, input_incepa])
        
        out = Activation("relu")(out)
    
        return out
    
    
    def plot_nnmodel(self, path = 'inception_resnetv2_model.png', mode = 'ori'):
        if mode = 'ori':
            modelx = self.build_orignial_model()
            plot_model(modelx, to_file=path, show_shapes = True)
        elif mode = 'sam':
            modelx = self.build_model()
            plot_model(modelx, to_file=path, show_shapes = True)
        else:
            raise('Value error')
    
    
    def summary(self):
        modelx = self.build_model()
        modelx.summary()
        
        
    def load_pretrained(self):
        base_model = InceptionResNetV2(weights='imagenet', include_top = self.include_top, input_shape = self.input_shape)
        return base_model
