import tensorflow as tf
from tensorflow import keras

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(ConvBlock, self).__init__()
        self.filters = n_filters
        self.conv = keras.layers.Conv2D(n_filters, 3, padding='same')
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
        # [64, 128, 256, 512, 1024]
        self.conv_blocks_0_1 = ConvBlock(self.filters[0])
        self.conv_blocks_0_2 = ConvBlock(self.filters[0])
        self.conv_blocks_1_1 = ConvBlock(self.filters[1])
        self.conv_blocks_1_2 = ConvBlock(self.filters[1])
        self.conv_blocks_2_1 = ConvBlock(self.filters[2])
        self.conv_blocks_2_2 = ConvBlock(self.filters[2])
        self.conv_blocks_3_1 = ConvBlock(self.filters[3])
        self.conv_blocks_3_2 = ConvBlock(self.filters[3])
        self.conv_blocks_4_1 = ConvBlock(self.filters[4])
        self.conv_blocks_4_2 = ConvBlock(self.filters[4])
        
        self.max_poolings_1 = keras.layers.MaxPooling2D(2)
        self.max_poolings_2 = keras.layers.MaxPooling2D(2)
        self.max_poolings_3 = keras.layers.MaxPooling2D(2)
        self.max_poolings_4 = keras.layers.MaxPooling2D(2)
        
    def call(self, x):
        skips = []
        
        x = self.conv_blocks_0_1(x)
        x = self.conv_blocks_0_2(x)
        skips.append(x)
        x = self.max_poolings_1(x)
        
        x = self.conv_blocks_1_1(x)
        x = self.conv_blocks_1_2(x)
        skips.append(x)
        x = self.max_poolings_2(x)
        
        x = self.conv_blocks_2_1(x)
        x = self.conv_blocks_2_2(x)
        skips.append(x)
        x = self.max_poolings_3(x)
        
        x = self.conv_blocks_3_1(x)
        x = self.conv_blocks_3_2(x)
        skips.append(x)
        x = self.max_poolings_4(x)
        
        x = self.conv_blocks_4_1(x)
        x = self.conv_blocks_4_2(x)

        return x, skips

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, is_recons=False, input_channel=3):
        super(DecoderBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        # enc_filters = [64, 128, 256, 512, 1024]
        # dec_filters = [512, 256, 128, 64]
        self.upsample_blocks_0 = UpsampleBlock(self.filters[0])
        self.upsample_blocks_1 = UpsampleBlock(self.filters[1])
        self.upsample_blocks_2 = UpsampleBlock(self.filters[2])
        self.upsample_blocks_3 = UpsampleBlock(self.filters[3])
        
        self.last_conv = ConvBlock(2)
        if is_recons:
            self.last_block = keras.layers.Conv2D(filters=input_channel, 
                                                  kernel_size=1, 
                                                  padding='same', 
                                                  activation='linear')
        else:
            self.last_block = keras.layers.Conv2D(filters=1, 
                                                  kernel_size=1, 
                                                  padding='same', 
                                                  activation='sigmoid')
        

    def call(self, x, skips):
        x = self.upsample_blocks_0(x, skips[0])
        x = self.upsample_blocks_1(x, skips[1])
        x = self.upsample_blocks_2(x, skips[2])
        x = self.upsample_blocks_3(x, skips[3])
        
        x = self.last_conv(x)
        x = self.last_block(x)
        
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

class ClsBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ClsBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        # [64, 128, 256, 512, 1024]
        self.conv_blocks_0_1 = ConvBlock(self.filters[0])
        self.conv_blocks_0_2 = ConvBlock(self.filters[0])
        self.conv_blocks_1_1 = ConvBlock(self.filters[1])
        self.conv_blocks_1_2 = ConvBlock(self.filters[1])
        self.conv_blocks_2_1 = ConvBlock(self.filters[2])
        self.conv_blocks_2_2 = ConvBlock(self.filters[2])
        self.conv_blocks_3_1 = ConvBlock(self.filters[3])
        self.conv_blocks_3_2 = ConvBlock(self.filters[3])
        self.conv_blocks_4_1 = ConvBlock(self.filters[4])
        self.conv_blocks_4_2 = ConvBlock(self.filters[4])
        
        self.max_poolings_1 = keras.layers.MaxPooling2D(2)
        self.max_poolings_2 = keras.layers.MaxPooling2D(2)
        self.max_poolings_3 = keras.layers.MaxPooling2D(2)
        self.max_poolings_4 = keras.layers.MaxPooling2D(2)
        
        self.concat_0 = keras.layers.Concatenate()
        self.concat_1 = keras.layers.Concatenate()
        self.concat_2 = keras.layers.Concatenate()
        self.concat_3 = keras.layers.Concatenate()
        self.concat_4 = keras.layers.Concatenate()
        
        self.gap = keras.layers.GlobalAveragePooling2D()
        
        self.dense = keras.layers.Dense(5, activation='softmax')
        
    def call(self, inputs, skips, encoded_x):         
        x = self.conv_blocks_0_1(inputs)
        x = self.concat_0([x, skips[0]])
        x = self.conv_blocks_0_2(x) 
        x = self.max_poolings_1(x)
        
        x = self.conv_blocks_1_1(x)
        x = self.concat_1([x, skips[1]])
        x = self.conv_blocks_1_2(x)
        x = self.max_poolings_2(x)
        
        x = self.conv_blocks_2_1(x)
        x = self.concat_2([x, skips[2]])
        x = self.conv_blocks_2_2(x)
        x = self.max_poolings_3(x)
        
        x = self.conv_blocks_3_1(x)
        x = self.concat_3([x, skips[3]])
        x = self.conv_blocks_3_2(x)
        x = self.max_poolings_4(x)
        
        x = self.conv_blocks_4_1(x)
        x = self.conv_blocks_4_2(x)
        
        x = self.concat_4([x, encoded_x])
        
        x = self.gap(x)
        
        x = self.dense(x)

        return x
      
class SnC_Unet(tf.keras.Model):
    def __init__(self, enc_filters, dec_filters, input_channel):
        super(SnC_Unet, self).__init__()

        self.enc_filters = enc_filters
        self.dec_filters = dec_filters

        # Encoder
        self.encoder = EncoderBlock(self.enc_filters)

        # Decoder
        # 5종류의 decoder가 있음
        # reconstruction
        # HardExudate, Hemohedge, Microane, SoftExudates
         # self.HardExudate = DecoderBlock(self.dec_filters)
        # self.Hemohedge = DecoderBlock(self.dec_filters)
        # self.Microane = DecoderBlock(self.dec_filters)
        # self.SoftExudates = DecoderBlock(self.dec_filters)
        
        # 복원하기 위한 branch
        # self.reconstruction = DecoderBlock(self.dec_filters, is_recons=True, input_channel=input_channel)
        
        # 하나의 마스크로 합쳐서 예측
        self.decoder= DecoderBlock(self.dec_filters)
        
        # clsblock
        self.clsblock = ClsBlock(self.enc_filters)
       

    def call(self, inputs, only_cls=False):
        # Encoder
        x, skips = self.encoder(inputs)
        
        # Classification
        cls_pred = self.clsblock(inputs, skips, x)
        
        # Decoder
        if only_cls:
            return [cls_pred]
    
        else:
            mask_hat = self.decoder(x, skips[::-1])
            
            return [cls_pred, mask_hat]
            
        # # Decoder
        # input_hat = self.reconstruction(x, skips[::-1])
        
        # reconstruction만 학습할 때 구분
        # if only_recons:     
        #     return [input_hat]
        # else:
        #     # ex = self.HardExudate(x, skips[::-1])
        #     # he = self.Hemohedge(x, skips[::-1])
        #     # ma = self.Microane(x, skips[::-1])
        #     # se = self.SoftExudates(x, skips[::-1])
            
        #     mask_hat = self.decoder(x, skips[::-1])
            
        #     return [input_hat, mask_hat]