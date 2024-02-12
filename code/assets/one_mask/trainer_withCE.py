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
                 bce_weight=0):
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
        self.bce_weight = tf.cast(bce_weight, dtype=tf.float32)
        
        if beta!=None:
            self.b1, self.b2, self.b3, self.b4 = beta
            self.b1 = tf.cast(self.b1, dtype=tf.float32)
            self.b2 = tf.cast(self.b2, dtype=tf.float32)
            self.b3 = tf.cast(self.b3, dtype=tf.float32)
            self.b4 = tf.cast(self.b4, dtype=tf.float32)
        
        # reconstruction만 학습하는거면 안쓰는 decoder trainable=False로 해주기
        if self.for_recons:
            # self.model.HardExudate.trainable=False
            # self.model.Hemohedge.trainable=False
            # self.model.Microane.trainable=False
            # self.model.SoftExudates.trainable=False
            self.model.decoder.trainable=False
        else:
            # self.model.HardExudate.trainable=True
            # self.model.Hemohedge.trainable=True
            # self.model.Microane.trainable=True
            # self.model.SoftExudates.trainable=True
            self.model.decoder.trainable=True
            
        if self.alpha == 0.0:
            self.model.reconstruction.trainable=False            
            
        self.BCE_fn = tf.keras.losses.BinaryCrossentropy()

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
            if self.add_noise:
                preds = self.model(x_batch_train[1], only_recons=self.for_recons, training=True)
            else:
                preds = self.model(x_batch_train[0], only_recons=self.for_recons, training=True)    # 모델이 예측한 결과
#             input_hat, ex_hat, he_hat, ma_hat, se_hat = preds
            
#             ex, he, ma, se = y_batch_train
            
            # loss 계산하기
            # reconstruction
            loss_recons = self.mean_square_error(preds[0], x_batch_train[0])

            if not self.for_recons:
                # ex, he, ma, se
                # ex_loss = self.dice_loss(y_batch_train[0], preds[1])
                # he_loss = self.dice_loss(y_batch_train[1], preds[2])
                # ma_loss = self.dice_loss(y_batch_train[2], preds[3])
                # se_loss = self.dice_loss(y_batch_train[3], preds[4])
                
                dice_loss = self.dice_loss(y_batch_train, preds[1])
                bce_loss = self.BCE_fn(y_batch_train, preds[1])
                
                mask_loss = dice_loss + self.bce_weight * bce_loss 
                
                # loss 가중합 해주기
                # train_loss = self.b1 * ex_loss + self.b2 * he_loss + self.b3 * ma_loss + self.b4 * se_loss + self.alpha * loss_recons
                # return_loss = (ex_loss, he_loss, ma_loss, se_loss, loss_recons, train_loss)
                if self.alpha == 0.0:
                    train_loss = mask_loss
                else:
                    train_loss = self.alpha * loss_recons + mask_loss
                return_loss = (loss_recons.numpy(), train_loss.numpy(), mask_loss.numpy(), dice_loss.numpy(), bce_loss.numpy())
                
            else:     
                train_loss = loss_recons 
                return_loss = (loss_recons.numpy(), train_loss.numpy())
            
        grads = tape.gradient(train_loss, self.model.trainable_weights)  # gradient 계산
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청
        
        del preds
        
        return return_loss

    def train(self, train_dataset, val_dataset):
        
        for epoch in range(self.epochs):
            print("\nEpoch {}/{}".format(epoch+self.first_epoch, self.epochs))
            # train_dataset = train_dataset.take(steps_per_epoch)
            # val_dataset = val_dataset.take(val_step)

            tr_progBar = Progbar(target=len(train_dataset) * train_dataset.batch_size, stateful_metrics=['train_loss', 'loss_recons', 'mask_loss'])
            
            # 데이터 집합의 배치에 대해 반복
            
            # epoch 단위로 계산하기 위함
            mask_batch_loss = []
            recons_batch_loss = []
            total_batch_loss = []
            dice_batch_loss = []
            bce_batch_loss = []
            
            for step_train, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if not self.for_recons:
                    loss_recons, train_loss, mask_loss, dice_loss, bce_loss = self.train_on_batch(x_batch_train, y_batch_train)
                    
                    values = [('train_loss', train_loss), 
                              ('mask_loss', mask_loss), 
                              ('loss_recons', loss_recons),
                              ('dice_loss', dice_loss),
                              ('bce_loss', bce_loss)]
                                        
                    mask_batch_loss.append(mask_loss)
                    recons_batch_loss.append(loss_recons)
                    total_batch_loss.append(train_loss)
                    dice_batch_loss.append(dice_loss)
                    bce_batch_loss.append(bce_loss)
                    
                    if (step_train + 1) == len(train_dataset):
                        values = [('train_loss', np.mean(total_batch_loss)), 
                                  ('mask_loss', np.mean(mask_batch_loss)), 
                                  ('loss_recons', np.mean(recons_batch_loss)),
                                  ('dice_loss', np.mean(dice_batch_loss)),
                                  ('bce_loss', np.mean(bce_batch_loss))]
                else:
                    loss_recons, train_loss = self.train_on_batch(x_batch_train, y_batch_train)
                    values = [('train_loss', train_loss), ('loss_recons', loss_recons)]
                                        
                    recons_batch_loss.append(loss_recons)
                    total_batch_loss.append(train_loss)
                    
                    if (step_train + 1) == len(train_dataset):
                        values = [('train_loss', np.mean(total_batch_loss)),  
                                  ('loss_recons', np.mean(recons_batch_loss))]
                
                        
                tr_progBar.update((step_train + 1) * train_dataset.batch_size, values=values)
                                
                del train_loss
                del x_batch_train
                del y_batch_train
            
            # txt 파일에 기록하기
            if self.file_name != None:
                with open(self.file_name, 'a') as f:
                    f.write(f"epoch:{epoch + self.first_epoch}/val_loss:{np.mean(total_batch_loss)}/mask_loss:{np.mean(mask_batch_loss)}/recons_loss:{np.mean(recons_batch_loss)}/dice_loss:{np.mean(dice_batch_loss)}/bce_loss:{np.mean(bce_batch_loss)}\n") 
            
            
            # epoch 단위로 계산하기 위함
            mask_batch_loss = []
            recons_batch_loss = []
            total_batch_loss = []
            dice_batch_loss = []
            bce_batch_loss = []
            
            val_progBar = Progbar(target=len(val_dataset) * val_dataset.batch_size, stateful_metrics=['val_loss','mask_loss', 'loss_recons', 'dice_loss', 'bce_loss'])
            
            for step_val, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                # 모델이 예측한 결과
                if self.add_noise:
                    preds = self.model(x_batch_val[1], only_recons=self.for_recons, training=False)
                else:
                    preds = self.model(x_batch_val[0], only_recons=self.for_recons, training=False)    
                
                # loss 계산하기
                # reconstruction
                loss_recons = self.mean_square_error(preds[0], x_batch_val[0])
                
                if not self.for_recons:
                # ex, he, ma, se
                    # ex_loss = self.dice_loss(y_batch_val[0], preds[1])
                    # he_loss = self.dice_loss(y_batch_val[1], preds[2])
                    # ma_loss = self.dice_loss(y_batch_val[2], preds[3])
                    # se_loss = self.dice_loss(y_batch_val[3], preds[4])   
                    
                    dice_loss = self.dice_loss(y_batch_val, preds[1])
                    bce_loss = self.BCE_fn(y_batch_val, preds[1])

                    mask_loss = dice_loss + self.bce_weight * bce_loss
                    
                    # loss 가중합 해주기
                    val_loss = self.alpha * loss_recons + mask_loss
                    values = [('val_loss', val_loss.numpy()),
                              ('mask_loss', mask_loss.numpy()), 
                              ('loss_recons', loss_recons.numpy()),
                              ('dice_loss', dice_loss.numpy()),
                              ('bce_loss', bce_loss.numpy())]
                                        
                    mask_batch_loss.append(mask_loss.numpy())
                    recons_batch_loss.append(loss_recons.numpy())
                    total_batch_loss.append(val_loss.numpy())
                    dice_batch_loss.append(dice_loss.numpy())
                    bce_batch_loss.append(bce_loss.numpy())
                    
                    if (step_val + 1) == len(val_dataset):
                        values = [('val_loss', np.mean(total_batch_loss)), 
                                  ('mask_loss', np.mean(mask_batch_loss)), 
                                  ('loss_recons', np.mean(recons_batch_loss)),
                                  ('dice_loss', np.mean(dice_batch_loss)),
                                  ('bce_loss', np.mean(bce_batch_loss))]
                    
                else:     
                    val_loss = loss_recons
                    values = [('val_loss', val_loss.numpy()), ('loss_recons', loss_recons.numpy())]
                    
                    recons_batch_loss.append(loss_recons.numpy())
                    total_batch_loss.append(val_loss.numpy())
                    
                    if (step_val + 1) == len(val_dataset):
                        values = [('val_loss', np.mean(total_batch_loss)), 
                                  ('loss_recons', np.mean(recons_batch_loss))]
                    
                val_progBar.update((step_val + 1) * val_dataset.batch_size, values=values)
                
                
                del val_loss
                del x_batch_val
                del y_batch_val
                del preds
            
             # txt 파일에 기록하기
            if self.file_name != None:
                with open(self.file_name, 'a') as f:
                    f.write(f"epoch:{epoch + self.first_epoch}/val_loss:{np.mean(total_batch_loss)}/mask_loss:{np.mean(mask_batch_loss)}/recons_loss:{np.mean(recons_batch_loss)}/dice_loss:{np.mean(dice_batch_loss)}/bce_loss:{np.mean(bce_batch_loss)}\n")
        
            # 학습한 모델 저장하기
            if self.save_model_path != None:  
                self.model.save_weights(f"{self.save_model_path}/{epoch+self.first_epoch}")
        
        return None