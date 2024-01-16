from tensorflow.keras.utils import Progbar
import tensorflow as tf

import math

from models import SMD_Unet

import numpy as np
from data_generator import DR_Generator

tf.config.run_functions_eagerly(True)

class Trainer:
    def __init__(self, model, epochs, optimizer, for_recons, alpha, beta=None):
        '''
        for_recons : bool, 학습 단계 구분하기 위함
        alpha : recons loss에 곱해줄 가중치
        beta : [] , mask loss에 곱해줄 가중치 리스트
        '''
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.for_recons = for_recons
        self.alpha = tf.cast(alpha, dtype=tf.float64)
        self.beta = beta 

        if beta!=None:
            self.b1, self.b2, self.b3, self.b4 = beta
            self.b1 = tf.cast(self.b1, dtype=tf.float64)
            self.b2 = tf.cast(self.b2, dtype=tf.float64)
            self.b3 = tf.cast(self.b3, dtype=tf.float64)
            self.b4 = tf.cast(self.b4, dtype=tf.float64)
        
        # reconstruction만 학습하는거면 안쓰는 decoder trainable=False로 해주기
        if self.for_recons:
            self.model.HardExudate.trainable=False
            self.model.Hemohedge.trainable=False
            self.model.Microane.trainable=False
            self.model.SoftExudates.trainable=False
        else:
            self.model.HardExudate.trainable=True
            self.model.Hemohedge.trainable=True
            self.model.Microane.trainable=True
            self.model.SoftExudates.trainable=True

    # loss 함수 계산하는 부분 
    # return 값이 텐서여야 하는건가? -> 아마도 그런 것 같다.
    def dice_loss(self, inputs, targets, smooth = 1.):
        dice_losses = []
        
        for input, target in zip(inputs, targets): 
            input_flat = tf.reshape(input, [-1])
            target_flat = tf.reshape(target, [-1])
            
            input_flat = tf.cast(input_flat, dtype=tf.float64)
            target_flat = tf.cast(target_flat, dtype=tf.float64) 
            
            intersection = tf.reduce_sum(input_flat * target_flat)
            dice_coef = (2. * intersection + smooth) / (tf.reduce_sum(input_flat) + tf.reduce_sum(target_flat) + smooth)

            dice_losses.append(1. - dice_coef)
            
        result = tf.reduce_mean(dice_losses) 
        return result
    
    def mean_square_error(self, input_hats, inputs):        
        mses = []
        
        for input_hat, input in zip(input_hats, inputs):
            mses.append(tf.reduce_mean(tf.square(input_hat - input)))
            
        result = tf.reduce_mean(mses) # 배치 나눠서 계산하고 평균해주기
        return result

    @tf.function
    def train_on_batch(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            preds = self.model(x_batch_train, only_recons=self.for_recons)    # 모델이 예측한 결과
#             input_hat, ex_hat, he_hat, ma_hat, se_hat = preds
            
#             ex, he, ma, se = y_batch_train
            
            # loss 계산하기
            # reconstruction
            loss_recons = tf.cast(self.mean_square_error(preds[0], x_batch_train), dtype=tf.float64)

            if not self.for_recons:
            # ex, he, ma, se
                ex_loss = self.dice_loss(y_batch_train[0], preds[1])
                he_loss = self.dice_loss(y_batch_train[1], preds[2])
                ma_loss = self.dice_loss(y_batch_train[2], preds[3])
                se_loss = self.dice_loss(y_batch_train[3], preds[4])            
                # loss 가중합 해주기
                train_loss = self.b1 * ex_loss + self.b2 * he_loss + self.b3 * ma_loss + self.b4 * se_loss + self.alpha * loss_recons
            else:     
                train_loss = loss_recons 
            
        grads = tape.gradient(train_loss, self.model.trainable_weights)  # gradient 계산
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청
        
        del preds
        
        return train_loss

    def train(self, train_dataset, val_dataset):
        epochs = []
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            print("\nEpoch {}/{}".format(epoch+1, self.epochs))
            epochs.append(epoch)
            # train_dataset = train_dataset.take(steps_per_epoch)
            # val_dataset = val_dataset.take(val_step)

            tr_progBar = Progbar(target=len(train_dataset) * train_dataset.batch_size, stateful_metrics=['train_loss'])
            
            # 데이터 집합의 배치에 대해 반복합니다
            for step_train, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                train_loss = self.train_on_batch(x_batch_train, y_batch_train)

                # train metric(mean, auc, accuracy 등) 업데이트
                # acc_metric.update_state(y_batch_train, logits)

                values = [('train_loss', train_loss.numpy())]
                tr_progBar.update((step_train + 1) * train_dataset.batch_size, values=values)
                
                train_losses.append(train_loss)
                
                del train_loss
                del x_batch_train
                del y_batch_train
            
            val_progBar = Progbar(target=len(val_dataset) * val_dataset.batch_size, stateful_metrics=['val_loss'])
            
            for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                preds = self.model(x_batch_val, only_recons=self.for_recons)    # 모델이 예측한 결과
#                 input_hat, ex_hat, he_hat, ma_hat, se_hat = preds
                
#                 ex, he, ma, se = y_batch_val
                
                # loss 계산하기
                # reconstruction
                loss_recons = tf.cast(self.mean_square_error(preds[0], x_batch_val), dtype=tf.float64)
                
                if not self.for_recons:
                # ex, he, ma, se
                    ex_loss = self.dice_loss(y_batch_val[0], preds[1])
                    he_loss = self.dice_loss(y_batch_val[1], preds[2])
                    ma_loss = self.dice_loss(y_batch_val[2], preds[3])
                    se_loss = self.dice_loss(y_batch_val[3], preds[4])            
                    # loss 가중합 해주기
                    val_loss = self.b1 * ex_loss + self.b2 * he_loss + self.b3 * ma_loss + self.b4 * se_loss + self.alpha * loss_recons
                else:     
                    val_loss = loss_recons
                    
                values = [('val_loss', val_loss.numpy())]
                val_progBar.update((step_val + 1) * val_dataset.batch_size, values=values)
                
                val_losses.append(val_loss)
                
                del val_loss
                del x_batch_val
                del y_batch_val
                del preds


        history = {}
        history['train_loss'] = train_losses
        history['val_loss'] = val_losses
        history['epochs'] = epochs
        
        return history