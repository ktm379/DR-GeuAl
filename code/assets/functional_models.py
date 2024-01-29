from tensorflow import keras
import tensorflow as tf

def ConvBlock(x, n_filters):
    x = keras.layers.Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x

def UpsampleBlock(x, skip, n_filters):
    x = keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding='same')(x)
    x = keras.layers.Concatenate()([x, skip]) 
    x = ConvBlock(x, n_filters)

    return x

def Encoder(x, filters):
    skips = []

    for f in filters:
        x = ConvBlock(x, f)
        x = ConvBlock(x, f)
        # 맨 마지막 층을 제외하고는 skip connection, downsampling을 진행
        if f != filters[-1]:
            skips.append(x)
            x = keras.layers.MaxPooling2D(2)(x)

    return x, skips

def Decoder(x, filters, skips):
    for f, skip in zip(filters, skips):
        x = keras.layers.Conv2DTranspose(f, (2, 2), strides=2, padding='same')(x)
        x = ConvBlock(x, f)
        x = keras.layers.Concatenate()([x, skip]) 
        x = ConvBlock(x, f)
      
    x = ConvBlock(x, 2)
    x = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)
  
    return x

# filters = [64, 128, 256, 512, 1024]

def Unet(img_size, filters):
    inputs = keras.Input(shape=img_size + (1,))

    # 축소 경로
    x, skips = Encoder(inputs, filters)

    # 확장 경로
    x = Decoder(x, filters[::-1][1:], skips[::-1])

    model = keras.Model(inputs, x)

    return model
