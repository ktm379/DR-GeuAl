import tensorflow as tf
import os
import math
import numpy as np
import pandas as pd

from PIL import Image

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow import keras

from assets.one_mask.Preprocessing import preprocess_image
from assets.one_mask.utils import add_gaussian_noise

import matplotlib.pyplot as plt

class DR_Generator(tf.keras.utils.Sequence):
    '''
    tf.keras.utils.Sequence를 상속받아 Sequence를 커스텀해 사용
    '''
    def __init__(self, 
               dir_path,
               mask_path=None,
               use_mask=False,
               batch_size=4,
               img_size=(512, 512),
               dataset=None,
               start_end_index=None,
               is_train=True,
               use_3channel=False,
               CLAHE_args=None,
               add_noise_std=None,
               label_path=None):
        '''
        dir_path (str): image path
        mask_path ([str]): mask path , 순서는 HardExudate, Hemohedge, Microane, SoftExudates
        use_mask (bool) : mask 사용 여부
        batch_size (int): batch_size
        img_size (224, 224, 3): 다루는 이미지 사이즈
        dataset (str) : 가져오려는 dataset 이름 , APTOS or FGADR or EyePacks      
        start_end_index : dataset 중 가져오려는 index의 시작과 끝
        '''
        self.dir_path = dir_path
        self.mask_path = mask_path
        self.use_mask = use_mask
        self.batch_size = batch_size
        self.img_size = img_size
        self.dataset = dataset
        self.start_end_index = start_end_index
        self.is_train=is_train
        self.use_3channel=use_3channel
        
        self.CLAHE_args = CLAHE_args
        if self.CLAHE_args == None:
            self.use_hist = False
        self.add_noise_std = add_noise_std
        
        if label_path != None:
            self.withCls = True 
            label_df = pd.read_csv(label_path, header=None)
            self.label_df = label_df.rename(columns={0:"file_name"})
        else:
            self.withCls = False 
        
        # load_dataset()을 통해서 directory path에서 라벨과 이미지를 확인
        self.data_paths = self.load_dataset()


    def get_label(self, file_name):
        return int(self.label_df[self.label_df['file_name'] == file_name][1].values[0])


    def load_dataset(self):
        '''
        path 불러오는 함수  
        tr : val : ts = 70 : 10 : 20
        1290: 184: 368
        '''
        # 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드하는 함수
        # is_train, is_valid, is_test로 분리함
        
        # 이미지 path 지정
        file_list = os.listdir(self.dir_path)
        input_images_paths = [os.path.join(self.dir_path, file) for file in file_list]
        input_images_paths.sort()

        # mask path 지정
        # mask : HardExudate, Hemohedge, Microane, SoftExudates
        if self.use_mask:
            ex_file_list = os.listdir(self.mask_path[0])
            ex_paths = [os.path.join(self.mask_path[0], file) for file in ex_file_list]
            ex_paths.sort()

            he_file_list = os.listdir(self.mask_path[1])
            he_paths = [os.path.join(self.mask_path[1], file) for file in he_file_list]
            he_paths.sort()

            me_file_list = os.listdir(self.mask_path[2])
            me_paths = [os.path.join(self.mask_path[2], file) for file in me_file_list]
            me_paths.sort()

            se_file_list = os.listdir(self.mask_path[3])
            se_paths = [os.path.join(self.mask_path[3], file) for file in se_file_list]
            se_paths.sort()

            mask_images_paths = [ _ for _ in zip(ex_paths, he_paths, me_paths, se_paths)]

            # image와 label 길이가 같은지 확인
            assert len(input_images_paths) == len(mask_images_paths)

            # path 합치기
            data_paths = [ _ for _ in zip(input_images_paths, mask_images_paths)]
        
        else:
          # unsupervsion 떄는 필요없음
          data_paths = [_ for _ in zip(input_images_paths)]
        
        start, end = self.start_end_index
        return data_paths[start:end] 
    
    def __len__(self):
        # Generator의 length로서 전체 dataset을 batch_size로 나누고 소숫점 첫째자리에서 올림한 값을 반환
        return math.ceil(len(self.data_paths) / self.batch_size)

    def __getitem__(self, index):
        '''
        입력과 출력을 만드는 함수
        '''
        batch_data_paths = self.data_paths[
                               index*self.batch_size:
                               (index + 1)*self.batch_size
                               ]
        
        # 채널 수 바꿔줘야 됨
        if self.use_3channel:
            inputs = np.zeros([self.batch_size, *self.img_size, 3])
        else:
            inputs = np.zeros([self.batch_size, *self.img_size])
        
        if self.dataset == "FGADR":
            # mask가 4개임 , HardExudate, Hemohedge, Microane, SoftExudates
            # ex = np.zeros([self.batch_size, *self.img_size])
            # he = np.zeros([self.batch_size, *self.img_size])
            # ma = np.zeros([self.batch_size, *self.img_size])
            # se = np.zeros([self.batch_size, *self.img_size])

            mask = np.zeros([self.batch_size, *self.img_size])
            
        if self.withCls:
            label = np.zeros([self.batch_size])        
        
        for i, data in enumerate(batch_data_paths):
            # supervsion 일때는 4개의 label
            # unsupervision 일때는 label이 image가 됨 
            if self.use_mask:
                input_img_path, output_paths = data

                # mask : HardExudate, Hemohedge, Microane, SoftExudates
                _ex = preprocess_image(output_paths[0], img_size=self.img_size, use_hist=False)
                _ex[_ex != 0] = 1
                _he = preprocess_image(output_paths[1], img_size=self.img_size, use_hist=False)
                _he[_he != 0] = 1
                _ma = preprocess_image(output_paths[2], img_size=self.img_size, use_hist=False)
                _ma[_ma != 0] = 1
                _se = preprocess_image(output_paths[3], img_size=self.img_size, use_hist=False)
                _se[_se != 0] = 1
                
                # ex[i] = _ex; he[i] = _he; ma[i] = _ma; se[i] = _se
                _mask = np.maximum(_ex, _he)
                _mask = np.maximum(_mask, _ma)
                _mask = np.maximum(_mask, _se)
                mask[i] = _mask
                
            else:
                # mask 없음
                input_img_path,  = data
                
                
            if self.withCls:
                label[i] = self.get_label(input_img_path.split('/')[-1])

            # image
            _input = preprocess_image(input_img_path, 
                                      img_size=self.img_size, 
                                      use_3channel=self.use_3channel, 
                                      CLAHE_args=self.CLAHE_args, 
                                      use_hist=self.use_hist)
   
            inputs[i] = _input

            
        # shape 추가해주기
        # (batch, 512, 512) -> (batch, 512, 512, 1)
        # 3채널이 아닌 경우에는 shape 바꿔줘야됨. 안그러면 오류남.
        if not self.use_3channel:
            inputs = inputs.reshape(self.batch_size, *self.img_size, 1)
            
        if self.add_noise_std != None:
            noisy_inputs = add_gaussian_noise(inputs, self.add_noise_std)
        
        # mask 없을 때는 input이 label이 됨
        if self.dataset == "EyePacks":
            if self.add_noise_std != None:
                return [tf.cast(inputs, dtype=tf.float32), tf.cast(noisy_inputs, dtype=tf.float32)], None

            return [tf.cast(inputs, dtype=tf.float32)], None
        
        # input, [mask 4개]
        if self.dataset == "FGADR":
            
            # shape 추가해주기
            # (batch, 512, 512) -> (batch, 512, 512, 1)
            # ex = ex.reshape(self.batch_size, *self.img_size, 1)
            # he = he.reshape(self.batch_size, *self.img_size, 1)
            # ma = ma.reshape(self.batch_size, *self.img_size, 1)
            # se = se.reshape(self.batch_size, *self.img_size, 1)
            
            mask = mask.reshape(self.batch_size, *self.img_size, 1)
            
            if self.add_noise_std != None:
                return [tf.cast(inputs, dtype=tf.float32), tf.cast(noisy_inputs, dtype=tf.float32)], [tf.cast(mask, dtype=tf.float32)]
            
            if self.withCls:
                label = label.reshape(self.batch_size, 1)
                
                return [tf.cast(inputs, dtype=tf.float32)], [tf.cast(mask, dtype=tf.float32), tf.cast(label, dtype=tf.int32)]
            
            return [tf.cast(inputs, dtype=tf.float32)], [tf.cast(mask, dtype=tf.float32)]
        
    def on_epoch_end(self):
        # 한 epoch가 끝나면 실행되는 함수
        # 학습중인 경우에 순서를 random shuffle하도록 적용
        self.indexes = np.arange(len(self.data_paths))
        if self.is_train == True :
            np.random.shuffle(self.indexes)
            return self.indexes