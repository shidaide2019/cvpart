import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Dropout, Flatten
from tensorflow.keras.regularizers import  l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

class Inceptionv4():
    
    def __init__(self, input_shape = (299,299,3), output_units = 1000,
                 
                 regularizer = l2(1e-4), initializer = "he_normal", init_maxpooling = True, include_top = True):
        
        super(Inceptionv4, self).__init__()
        self.input_shape = input_shape
        self.output_units = output_units
        self.regularizer = regularizer
        self.initializer = initializer
        self.init_maxpooling = init_maxpooling
        self.include_top = include_top
        
    
    def conv2d_bn(self, x, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        
        else:
            channel_axis = -1

        x = Conv2D(nb_filter, (num_row, num_col),
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   kernel_regularizer=regularizers.l2(0.00004),
                   kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
        
        x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
        x = Activation('relu')(x)
        return x



    def block_inception_a(self, input_incepa):
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
    
    
        branch_0 = self.conv2d_bn(input_incepa, 96, 1, 1)
    
        branch_1 = self.conv2d_bn(input_incepa, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3)
    
        branch_2 = self.conv2d_bn(input_incepa, 64, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
    
        branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input_incepa)
        branch_3 = self.conv2d_bn(branch_3, 96, 1, 1)
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    
        return x


    def block_reduction_a(self, input_reda):
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
            
        branch_0 = self.conv2d_bn(input_reda, 384, 3, 3, strides=(2,2), padding='valid')
    
        branch_1 = self.conv2d_bn(input_reda, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 3, 3)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 3, strides=(2,2), padding='valid')
    
        branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input_reda)
        
        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        
        return x



    def block_inception_b(self, input_incepb):
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        
        branch_0 = self.conv2d_bn(input_incepb, 384, 1, 1)
        
        
        branch_1 = self.conv2d_bn(input_incepb, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 256, 7, 1)
        
        
        branch_2 = self.conv2d_bn(input_incepb, 192, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self.conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 256, 1, 7)
        
        
        branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input_incepb)
        branch_3 = self.conv2d_bn(branch_3, 128, 1, 1)
        
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        
        
        return x



    def block_reduction_b(self, input_redb):
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
    
        branch_0 = self.conv2d_bn(input_redb, 192, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')
        
        branch_1 = self.conv2d_bn(input_redb, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 320, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 320, 3, 3, strides=(2,2), padding='valid')
        
        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input_redb)
        
        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        
        return x
    
    
    def block_inception_c(self, input_incepc):
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
    
        branch_0 = self.conv2d_bn(input_incepc, 256, 1, 1)
    
    
        branch_1 = self.conv2d_bn(input_incepc, 384, 1, 1)
    
        branch_10 = self.conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self.conv2d_bn(branch_1, 256, 3, 1)
    
        branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)
    
    
        branch_2 = self.conv2d_bn(input_incepc, 384, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self.conv2d_bn(branch_2, 512, 1, 3)
    
        branch_20 = self.conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self.conv2d_bn(branch_2, 256, 3, 1)
    
        branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)
    
    
    
        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_incepc)
        branch_3 = self.conv2d_bn(branch_3, 256, 1, 1)
    
        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    
        return x
    
    
    def build_model(self):
        
        input_x = Input(self.input_shape)
        
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        
        net = self.conv2d_bn(input_x, 32, 3, 3, strides=(2,2), padding='valid')
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid')
        net = self.conv2d_bn(net, 64, 3, 3)
        
        
        branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid')
        
        
        net = concatenate([branch_0, branch_1], axis=channel_axis)
        
        
        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid')
        
        
        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='valid')
        
        net = concatenate([branch_0, branch_1], axis=channel_axis)
        
        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid')
        branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
        
        
        net = concatenate([branch_0, branch_1], axis=channel_axis)
        # 35 x 35 x 384
        
        # 4 x Inception-A blocks
        
        for idx in range(4):
    	    net = self.block_inception_a(net)
        
        # Reduction-A block
            
        net = self.block_reduction_a(net)
        # 17 x 17 x 1024
        
        # 7 x Inception-B blocks
        
        for idx in range(7):
    	    net = self.block_inception_b(net)
        
        # Reduction-B block
        
        net = self.block_reduction_b(net)
        # 8 x 8 x 1536
        
        # 3 x Inception-C blocks
        
        for idx in range(3):
    	    net = self.block_inception_c(net)
            
        if self.include_top:
            
            # 1 x 1 x 1536
            net = AveragePooling2D((8,8), padding='valid')(net)
            net = Dropout(0.2)(net)
            net = Flatten()(net)
            
            # 1536
            net = Dense(units=self.output_units, activation='softmax')(net)
        inceptionv4_model = Model(inputs = [input_x], outputs = [net], name='inception_v4')

        return inceptionv4_model
    
    def plot_nnmodel(self, path = 'inceptionv4_model.png'):
        modelx = self.build_model()
        plot_model(modelx, to_file=path, show_shapes = True)
        
    
    def summary(self):
        modelx = self.build_model()
        modelx.summary()
    
    
model1 = Inceptionv4()

model1.summary()
