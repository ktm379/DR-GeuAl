from tensorflow.keras.utils import Progbar
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import math

from assets.one_mask.models import SMD_Unet
from assets.one_mask.data_generator import DR_Generator

tf.config.run_functions_eagerly(True)

class Trainer:
    def __init__(self, 
                 model, 
                 epochs, 
                 optimizer, 
                 for_recons,
                 alpha, 
                 beta=None, 
                 first_epoch=1, 
                 file_name=None, 
                 save_model_path=None, 
                 add_noise=False,
                 with_mask=False):
        '''
        for_recons : bool, 학습 단계 구분하기 위함
        alpha : recons loss에 곱해줄 가중치
        beta : [] , mask loss에 곱해줄 가중치 리스트
        first_epoch : 기록하기 위한 값, 처음 시작하는 epoch값이 뭐인지 
        '''
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.for_recons = for_recons
        self.alpha = tf.cast(alpha, dtype=tf.float32)
        self.beta = beta 
        self.first_epoch = first_epoch
        self.file_name = file_name
        self.save_model_path = save_model_path
        self.add_noise = add_noise
        self.with_mask = with_mask
         
        self.CE = tf.keras.losses.SparseCategoricalCrossentropy()

    # loss 함수 계산하는 부분 
    # return 값이 텐서여야 하는건가? -> 아마도 그런 것 같다.
    def dice_coef(self, y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def mean_square_error(self, input_hats, inputs):        
        mses = []

        for input_hat, input in zip(input_hats, inputs):
            mses.append(tf.reduce_mean(tf.square(input_hat - input)))

        result = tf.reduce_mean(mses) # 배치 나눠서 계산하고 평균해주기
        return result

    @tf.function
    def train_on_batch(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            preds = self.model(x_batch_train[0], training=True)    # 모델이 예측한 결과
            
            # loss 계산하기            
            cls_loss = self.CE(y_batch_train[0], preds[0])
            
            return_loss = (cls_loss.numpy())
                
            
        grads = tape.gradient(cls_loss, self.model.trainable_weights)  # gradient 계산
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청
        
        del preds
        
        return return_loss

    def train(self, train_dataset, val_dataset):
        
        for epoch in range(self.epochs):
            print("\nEpoch {}/{}".format(epoch+self.first_epoch, self.epochs))
            # train_dataset = train_dataset.take(steps_per_epoch)
            # val_dataset = val_dataset.take(val_step)

            tr_progBar = Progbar(target=len(train_dataset) * train_dataset.batch_size, stateful_metrics=['cls_loss'])
            
            # 데이터 집합의 배치에 대해 반복
            
            # epoch 단위로 계산하기 위함
            cls_batch_loss = []
            
            for step_train, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                cls_loss = self.train_on_batch(x_batch_train, y_batch_train)
                values = [('cls_loss', cls_loss)]
                                    
                cls_batch_loss.append(cls_loss)
                
                if (step_train + 1) == len(train_dataset):
                    values = [('cls_loss', np.mean(cls_batch_loss))]
                        
                tr_progBar.update((step_train + 1) * train_dataset.batch_size, values=values)
                                
                del cls_loss
                del x_batch_train
                del y_batch_train
            
            # txt 파일에 기록하기
            if self.file_name != None:
                with open(self.file_name, 'a') as f:
                    f.write(f"epoch:{epoch + self.first_epoch}/cls_loss:{np.mean(cls_batch_loss)}\n")  
            
            
            # epoch 단위로 계산하기 위함
            cls_batch_loss = []

            val_progBar = Progbar(target=len(val_dataset) * val_dataset.batch_size, stateful_metrics=['cls_loss'])
            
            for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                # 모델이 예측한 결과
                preds = self.model(x_batch_val[0], training=False)    # 모델이 예측한 결과
                
                cls_loss = self.CE(y_batch_val[0], preds[0])
                values = [('cls_loss', cls_loss)]
                                        
                cls_batch_loss.append(cls_loss.numpy())
                
                if (step_val + 1) == len(val_dataset):
                    values = [('cls_loss', np.mean(cls_batch_loss))]
                
                val_progBar.update((step_val + 1) * val_dataset.batch_size, values=values)
                
                
                del cls_loss
                del x_batch_val
                del y_batch_val
                del preds
            
             # txt 파일에 기록하기
            if self.file_name != None:
                with open(self.file_name, 'a') as f:
                    f.write(f"epoch:{epoch + self.first_epoch}/cls_loss:{np.mean(cls_batch_loss)}\n")
        
            # 학습한 모델 저장하기
            if self.save_model_path != None:  
                self.model.save_weights(f"{self.save_model_path}/{epoch+self.first_epoch}")
        
        return None