from tensorflow import keras
from keras import backend as K
from scipy.stats import bernoulli

# from keras.engine.topology import Layer
# import keras_cv

import tensorflow as tf
import copy

class DropBlock(keras.layers.Layer):
    """
    Regularization Technique for Convolutional Layers.

    Pseudocode:
    1: Input:output activations of a layer (A), block_size, γ, mode
    2: if mode == Inference then
    3: return A
    4: end if
    5: Randomly sample mask M: Mi,j ∼ Bernoulli(γ)
    6: For each zero position Mi,j , create a spatial square mask with the center being Mi,j , the width,
        height being block_size and set all the values of M in the square to be zero (see Figure 2).
    7: Apply the mask: A = A × M
    8: Normalize the features: A = A × count(M)/count_ones(M)

    # Arguments
        block_size: A Python integer. The size of the block to be dropped.
        gamma: float between 0 and 1. controls how many activation units to drop.
    # References
        - [DropBlock: A regularization method for convolutional networks](
           https://arxiv.org/pdf/1810.12890v1.pdf)
    """
    def __init__(self, block_size, keep_prob):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def call(self, x, training=False):

        # During inference, we do not Drop Blocks. (Similar to DropOut)
        if not training:
            return x

        # Calculate Gamma
        feat_size = int(x.shape[-1])
        gamma = ((1-self.keep_prob)/(self.block_size**2)) * ((feat_size**2) / ((feat_size-self.block_size+1)**2))

        padding = self.block_size//2

        # Randomly sample mask
        sample = bernoulli.rvs(size=(feat_size-(padding*2), feat_size-(padding*2)),p=gamma)

        # The above code creates a matrix of zeros and samples ones from the distribution
        # We would like to flip all of these values
        sample = 1-sample

        # Pad the mask with ones
        sample = np.pad(sample, pad_width=padding, mode='constant', constant_values=1)

        # For each 0, create spatial square mask of shape (block_size x block_size)
        mask = copy.copy(sample)
        for i in range(feat_size):
            for j in range(feat_size):
                if sample[i, j]==0:
                    mask[i-padding : i+padding+1, j-padding : j+padding+1] = 0

        mask = mask.reshape((1, feat_size, feat_size))

        # Apply the mask
        x = x * np.repeat(mask, x.shape[1], 0)

        # Normalize the features
        count = np.prod(mask.shape)
        count_ones = np.count_nonzero(mask == 1)
        x = x * count / count_ones

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, dropout_args=None):
        super(ConvBlock, self).__init__()
        self.filters = n_filters
        self.conv = keras.layers.Conv2D(n_filters, 3, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        if dropout_args != None:
            rate, block_size = dropout_args
            keep_probs = 1 - rate
            
            self.dropout = DropBlock(keep_prob=keep_prob, block_size=block_size)         
            self.use_drop = True
        else:
            self.use_drop = False
        # self.dropout = keras_cv.layers.DropoutBlock2D(rate=0.08, block_size=7)
            
    def call(self, x, training):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_drop:
            x = self.dropout(x, training)
        x = self.relu(x)
        
        return x


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, dropout_args):
        super(UpsampleBlock, self).__init__()
        self.filters = n_filters
        self.conv_T = keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')
        self.conv_block_1 = ConvBlock(n_filters, dropout_args)
        self.concat = keras.layers.Concatenate()
        self.conv_block_2 = ConvBlock(n_filters, dropout_args)

    def call(self, x, skip, training):
        x = self.conv_T(x)
        x = self.conv_block_1(x, training)
        x = self.concat([x, skip])
        x = self.conv_block_2(x, training)

        return x   



class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_args):
        super(EncoderBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        # [64, 128, 256, 512, 1024]
        self.conv_blocks_0_1 = ConvBlock(self.filters[0], dropout_args)
        self.conv_blocks_0_2 = ConvBlock(self.filters[0], dropout_args)
        self.conv_blocks_1_1 = ConvBlock(self.filters[1], dropout_args)
        self.conv_blocks_1_2 = ConvBlock(self.filters[1], dropout_args)
        self.conv_blocks_2_1 = ConvBlock(self.filters[2], dropout_args)
        self.conv_blocks_2_2 = ConvBlock(self.filters[2], dropout_args)
        self.conv_blocks_3_1 = ConvBlock(self.filters[3], dropout_args)
        self.conv_blocks_3_2 = ConvBlock(self.filters[3], dropout_args)
        self.conv_blocks_4_1 = ConvBlock(self.filters[4], dropout_args)
        self.conv_blocks_4_2 = ConvBlock(self.filters[4], dropout_args)
        
        self.max_poolings_1 = keras.layers.MaxPooling2D(2)
        self.max_poolings_2 = keras.layers.MaxPooling2D(2)
        self.max_poolings_3 = keras.layers.MaxPooling2D(2)
        self.max_poolings_4 = keras.layers.MaxPooling2D(2)
        
    def call(self, x, training):
        skips = []
        
        x = self.conv_blocks_0_1(x, training)
        x = self.conv_blocks_0_2(x, training)
        skips.append(x)
        x = self.max_poolings_1(x)
        
        x = self.conv_blocks_1_1(x, training)
        x = self.conv_blocks_1_2(x, training)
        skips.append(x)
        x = self.max_poolings_2(x)
        
        x = self.conv_blocks_2_1(x, training)
        x = self.conv_blocks_2_2(x, training)
        skips.append(x)
        x = self.max_poolings_3(x)
        
        x = self.conv_blocks_3_1(x, training)
        x = self.conv_blocks_3_2(x, training)
        skips.append(x)
        x = self.max_poolings_4(x)
        
        x = self.conv_blocks_4_1(x, training)
        x = self.conv_blocks_4_2(x, training)

        return x, skips

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_args, is_recons=False, input_channel=3):
        super(DecoderBlock, self).__init__()
        self.filters = filters # filters = [n_filters]
        # enc_filters = [64, 128, 256, 512, 1024]
        # dec_filters = [512, 256, 128, 64]
        self.upsample_blocks_0 = UpsampleBlock(self.filters[0], dropout_args)
        self.upsample_blocks_1 = UpsampleBlock(self.filters[1], dropout_args)
        self.upsample_blocks_2 = UpsampleBlock(self.filters[2], dropout_args)
        self.upsample_blocks_3 = UpsampleBlock(self.filters[3], dropout_args)
        
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
        

    def call(self, x, skips, training):
        x = self.upsample_blocks_0(x, skips[0], training)
        x = self.upsample_blocks_1(x, skips[1], training)
        x = self.upsample_blocks_2(x, skips[2], training)
        x = self.upsample_blocks_3(x, skips[3], training)
        
        x = self.last_conv(x, training)
        x = self.last_block(x)
        
        return x
      
class SMD_Unet(tf.keras.Model):
    def __init__(self, enc_filters, dec_filters, input_channel, dropout_args):
        super(SMD_Unet, self).__init__()

        self.enc_filters = enc_filters
        self.dec_filters = dec_filters

        # Encoder
        self.encoder = EncoderBlock(self.enc_filters, dropout_args)

        # Decoder        
        # 복원하기 위한 branch
        self.reconstruction = DecoderBlock(self.dec_filters, dropout_args, is_recons=True, input_channel=input_channel)
        
        # 하나의 마스크로 합쳐서 예측
        self.decoder= DecoderBlock(self.dec_filters, dropout_args)


    def call(self, inputs, only_recons=False, training=False):
        # Encoder
        x, skips = self.encoder(inputs, training)
        # Decoder
        input_hat = self.reconstruction(x, skips[::-1], training)
        # reconstruction만 학습할 때 구분
        if only_recons:     
            return [input_hat]
        else:
            mask_hat = self.decoder(x, skips[::-1], training)
            
            return [input_hat, mask_hat]