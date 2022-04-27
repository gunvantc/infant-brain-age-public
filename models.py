import keras
from keras.layers.core import Dense, Activation
from keras.layers import Input, MaxPooling3D, Conv3D, Flatten, BatchNormalization, Dropout, AveragePooling3D, Lambda
from keras.layers import Concatenate, Reshape
from keras import Model

'''Contains same convolution layers for each modality'''
class brain_tumor_network:
    def __init__(self, scale, channels):
        self.configs= {
            'shape':(int(172*scale), int(220*scale), int(156*scale) ,channels),
            'num_filters' : 32,
            'kernel_size' : (3,3,3),
            'pool_size' : (2,2,2),
            'kernel_init' : 'he_normal',

            'first_block' : 5,
            'second_block' : 4,
            'third_block' : 3,
            'fourth_block': 2,
            'fifth_block': 1,

            'learning_rate': 1e-3
        }
        
    def get_configs(self):
        return self.configs
    
    def build_model(self):
        configs = self.configs
        
        inputs  = Input(shape = configs['shape'])
    #     genders = Input(shape = (1,))

        layer = Conv3D(    
            filters = configs['num_filters'],
            kernel_size = configs['kernel_size'],
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=configs['kernel_init'])(inputs)
        layer = Activation('relu')(layer)

        for ii in range(configs['first_block'] - 1):
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        for ii in range(configs['second_block']):
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)


        for ii in range(configs['third_block']):
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        for ii in range(configs['fourth_block']):
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        for ii in range(configs['fifth_block']):
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        layer = Conv3D(    
            filters = configs['num_filters'],
            kernel_size = configs['kernel_size'],
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=configs['kernel_init'])(layer)

        layer = Flatten()(layer)

        layer = Dense(512)(layer)
        layer = Dense(512)(layer)

        output = Dense(1, activation='linear')(layer)

        model = Model(inputs=[inputs], outputs=[output])

        return model

'''Contains separate convolution layers for each modality with LATE fusion'''
class brain_tumor_multimodal_network:
    def __init__(self, scale, channels):
        self.configs= {
            'shape':(int(172*scale), int(220*scale), int(156*scale) ,channels),
            'num_filters' : 32,
            'kernel_size' : (3,3,3),
            'pool_size' : (2,2,2),
            'kernel_init' : 'he_normal',

            'first_block' : 5,
            'second_block' : 4,
            'third_block' : 3,
            'fourth_block': 2,
            'fifth_block': 1,

            'learning_rate': 1e-3
        }
    
    
    def get_configs(self):
        return self.configs
    
    def build_model(self):
        configs = self.configs
        inputs  = Input(shape = configs['shape'])
    #     genders = Input(shape = (1,))

        conv_outputs = []

        for i in range(configs['shape'][-1]):
            layer = Lambda( lambda x: x[:,:,:,:,i])(inputs)
            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(inputs)
            layer = Activation('relu')(layer)

            for ii in range(configs['first_block'] - 1):
                layer = Conv3D(    
                    filters = configs['num_filters'],
                    kernel_size = configs['kernel_size'],
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = Activation('relu')(layer)

            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            for ii in range(configs['second_block']):
                layer = Conv3D(    
                    filters = configs['num_filters'],
                    kernel_size = configs['kernel_size'],
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = Activation('relu')(layer)

            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)


            for ii in range(configs['third_block']):
                layer = Conv3D(    
                    filters = configs['num_filters'],
                    kernel_size = configs['kernel_size'],
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = Activation('relu')(layer)

            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            for ii in range(configs['fourth_block']):
                layer = Conv3D(    
                    filters = configs['num_filters'],
                    kernel_size = configs['kernel_size'],
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = Activation('relu')(layer)

            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            for ii in range(configs['fifth_block']):
                layer = Conv3D(    
                    filters = configs['num_filters'],
                    kernel_size = configs['kernel_size'],
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = Activation('relu')(layer)

            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            layer = Conv3D(    
                filters = configs['num_filters'],
                kernel_size = configs['kernel_size'],
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            conv_outputs.append(layer)

        layer = Concatenate()(conv_outputs) 
        layer = Flatten()(layer)

    #     gender_layer = Dense(32)(genders)
    #     layer = Concatenate()([layer, gender_layer])

        layer = Dense(512)(layer)
        layer = Dense(256)(layer)

        output = Dense(1, activation='linear')(layer)

        model = Model(inputs=[inputs], outputs=[output])

        return model

'''Contains same convolution layers for each modality'''
class uk_biobank_network:
    def __init__(self, scale, channels):
        self.configs= {
            'shape':(int(172*scale), int(220*scale), int(156*scale) ,channels),
            'kernel_init' : 'he_normal',
            'channels' : [16, 64, 128, 256, 256, 64, 40],
            'pool_size' : (2,2,2),

            'learning_rate': 1e-3
        }
    
    
    def get_configs(self):
        return self.configs
    
    def build_model(self):
        configs = self.configs
    
        inputs  = Input(shape = configs['shape'])
    #     genders = Input(shape = (1,))

        layer = Conv3D(    
            filters = configs['channels'][0],
            kernel_size = (3,3,3),
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=configs['kernel_init'])(inputs)
        layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        for ii in range(4):
            layer = Conv3D(    
                filters = configs['channels'][ii+1],
                kernel_size = (3,3,3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

        layer = Conv3D(    
            filters = configs['channels'][5],
            kernel_size = (1,1,1),
            strides=(1, 1, 1),
            padding="same",
            kernel_initializer=configs['kernel_init'])(layer)
        layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
        layer = Activation('relu')(layer)

        layer = AveragePooling3D(pool_size=configs['pool_size'])(layer)
        layer = Dropout(rate=0.5)(layer)

        layer = Flatten()(layer)

    #     layer = Dense(1024)(layer)
        layer = Dense(512)(layer)

        output = Dense(1, activation='linear')(layer)

        model = Model(inputs=[inputs], outputs=[output])

        return model


'''Contains separate convolution layers for each modality with LATE fusion'''
class multichannel_uk_biobank_network:
    def __init__(self, scale, channels):
        self.configs= {
            'shape':(int(172*scale), int(220*scale), int(156*scale) ,channels),
            'kernel_init' : 'he_normal',
            'channels' : [16, 64, 128, 256, 256, 64, 40],
            'pool_size' : (2,2,2),

            'learning_rate': 1e-3
        }
    
    
    def get_configs(self):
        return self.configs
    
    def build_model(self):
        configs = self.configs
        inputs  = Input(shape = configs['shape'])
    #     genders = Input(shape = (1,))

        conv_outputs = []

        for i in range(configs['shape'][-1]):
            layer = Lambda( lambda x: x[:,:,:,:,i])(inputs)
            layer = Reshape(configs['shape'][:-1] + (1,))(layer)
            layer = Conv3D(    
                filters = configs['channels'][0],
                kernel_size = (3,3,3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            for ii in range(4):
                layer = Conv3D(    
                    filters = configs['channels'][ii+1],
                    kernel_size = (3,3,3),
                    strides=(1, 1, 1),
                    padding="same",
                    kernel_initializer=configs['kernel_init'])(layer)
                layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
                layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=configs['pool_size'])(layer)

            layer = Conv3D(    
                filters = configs['channels'][5],
                kernel_size = (1,1,1),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=configs['kernel_init'])(layer)
            layer = BatchNormalization(momentum=0.99, epsilon=0.001)(layer)
            layer = Activation('relu')(layer)

            layer = AveragePooling3D(pool_size=configs['pool_size'])(layer)
            layer = Dropout(rate=0.5)(layer)

            layer = Flatten()(layer)
            conv_outputs.append(layer)


        layer = Concatenate()(conv_outputs)

        layer = Dense(512)(layer)
        layer = Dense(256)(layer)

        output = Dense(1, activation='linear')(layer)

        model = Model(inputs=[inputs], outputs=[output])

        return model