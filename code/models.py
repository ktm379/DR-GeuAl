# from tensorflow import keras
# import tensorflow as tf

# def ConvBlock(x, n_filters):
#   x = keras.layers.Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
#   x = keras.layers.BatchNormalization()(x)
#   x = keras.layers.Activation('relu')(x)
  
#   return x

# def UpsampleBlock(x, skip, n_filters):
#   x = keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')(x)
#   x = keras.layers.Concatenate()([x, skip]) 
#   x = ConvBlock(x, n_filters)

#   return x

# def Encoder(x, filters):
#   skips = []
  
#   for f in filters:
#     x = ConvBlock(x, f)
#     x = ConvBlock(x, f)
#     # 맨 마지막 층을 제외하고는 skip connection, downsampling을 진행
#     if f != 1024:
#       skips.append(x)
#       x = keras.layers.MaxPooling2D(2)(x)
      
#   return x, skips

# def Decoder(x, filters, skips):
#   for f, skip in zip(filters, skips):
#     x = keras.layers.Conv2DTranspose(f, (2, 2), strides=2, padding='same')(x)
#     x = ConvBlock(x, f)
#     x = keras.layers.Concatenate()([x, skip]) 
#     x = ConvBlock(x, f)
      
#   x = ConvBlock(x, 2)
#   x = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='linear')(x)
  
#   return x

# def Unet(img_size):
#   inputs = keras.Input(shape=img_size + (1,))
  
#   # 축소 경로
#   filters = [64, 128, 256, 512, 1024]

#   x, skips = Encoder(inputs, filters)
  
#   # 확장 경로
#   x = Decoder(x, filters[::-1][1:], skips[::-1])
  
#   # loss = mse  
#   model = keras.Model(inputs, x)
  
#   return model


# --------------------------------------------------------------------------------------------------------#

# class api로 다시 정의


import tensorflow as tf
from tensorflow import keras

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(ConvBlock, self).__init__()
        self.filters = n_filters
        self.conv = keras.layers.Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
    
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
  
class UpsampleBlock(tf.keras.layers.Layer):
  def __init__(self, n_filters):
    super(UpsampleBlock, self).__init__()
    self.filters = n_filters
    self.conv_T = keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')
    self.conv_block_1 = ConvBlock(n_filters)
    self.concat = keras.layers.Concatenate()
    self.conv_block_2 = ConvBlock(n_filters)
  
  def call(self, x, skip):
    x = self.conv_T(x)
    x = self.conv_block_1(x)
    x = self.concat([x, skip])
    x = self.conv_block_2(x)
    
    return x   

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(EncoderBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        self.conv_blocks = [(ConvBlock(f), ConvBlock(f)) for f in self.filters]
        self.max_poolings = [keras.layers.MaxPooling2D(2) for _ in range(len(self.filters)-1)]
        # conv_block, max_pooling 리스트 길이 맞춰주기 위한 트릭
        self.max_poolings += [0]

    def call(self, x):
        skips = []
        for conv_block, max_pooling in zip(self.conv_blocks, self.max_poolings):
            x = conv_block[0](x)
            x = conv_block[1](x)

            # 맨 마지막 층을 제외하고는 skip connection, downsampling을 진행
            if conv_block[0].filters != 1024:
                skips.append(x)
                x = max_pooling(x)
        return x, skips

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(DecoderBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        self.upsample_blocks = [UpsampleBlock(f) for f in self.filters]
        self.last_block = tf.keras.Sequential([ConvBlock(2), 
                                               keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='linear')])

    def call(self, x, skips):
        for skip, upsample_block in zip(skips, self.upsample_blocks):
            x = upsample_block(x, skip)
        
        x = self.last_block(x)
        
        return x
      
class SMD_Unet(tf.keras.Model):
    def __init__(self):
        super(SMD_Unet, self).__init__()

        self.filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.encoder = EncoderBlock(self.filters)

        # Decoder
        # 5종류의 decoder가 있음
        # reconstruction
        # HardExudate, Hemohedge, Microane, SoftExudates
        self.reconstruction = DecoderBlock(self.filters[::-1][1:])
        self.HardExudate = DecoderBlock(self.filters[::-1][1:])
        self.Hemohedge = DecoderBlock(self.filters[::-1][1:])
        self.Microane = DecoderBlock(self.filters[::-1][1:])
        self.SoftExudates = DecoderBlock(self.filters[::-1][1:])


    def call(self, inputs, only_recons=False):
        # Encoder
        x, skips = self.encoder(inputs)
        # Decoder
        input_hat = self.reconstruction(x, skips[::-1])
        # reconstruction만 학습할 때 구분
        if only_recons:     
            return [input_hat]
        else:
            ex = self.HardExudate(x, skips[::-1])
            he = self.Hemohedge(x, skips[::-1])
            ma = self.Microane(x, skips[::-1])
            se = self.SoftExudates(x, skips[::-1])
            
            return [input_hat, ex, he, ma, se]