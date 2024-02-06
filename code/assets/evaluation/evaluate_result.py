import tensorflow as tf
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from assets.test.data_generator import DR_Generator_forInference
from assets.one_mask.models import SMD_Unet 
from assets.one_mask.trainer import Trainer
from tqdm import tqdm
import warnings

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    intersection = tf.reduce_sum(y_true * y_pred)  
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)  
    return (2. * intersection + smooth) / (union + smooth)

def mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))  

def calculate_iou(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  
    intersection = tf.reduce_sum(y_true * y_pred)  
    union = tf.reduce_sum(tf.cast(y_true + y_pred > threshold, tf.float32))  
    return intersection / (union + 1e-7)  

def evaluate_segmentation(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    dice = dice_coefficient(y_true_flat, y_pred_flat)
    iou = calculate_iou(y_true_flat, y_pred_flat)
    pr_auc = average_precision_score(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    return dice, iou, pr_auc, mae

def evaluate_model(generator, model_path, input_channel):
    model = SMD_Unet(enc_filters=[64, 128, 256, 512, 1024], dec_filters=[512, 256, 64, 32], input_channel=input_channel)
    model.load_weights(model_path)
    
    warnings.filterwarnings('ignore')

    file_name_list = []
    dice_list = []
    iou_list = []
    pr_auc_list = []
    mae_list = []

    for inputs, target in tqdm(generator):
        image = inputs[0]
        file_name = inputs[1]

        preds = model(image, training=False)
        mask_hat = preds[1]

        for i in range(mask_hat.shape[0]):
            file_name_list.append(file_name[i])
            dice, iou, pr_auc, mae = evaluate_segmentation(target[i], mask_hat[i])

            dice_list.append(dice.numpy())
            iou_list.append(iou.numpy())
            pr_auc_list.append(pr_auc)
            mae_list.append(mae.numpy())

    result = pd.DataFrame()
    result['file_name'] = file_name_list
    result['dice'] = dice_list
    result['iou'] = iou_list
    result['pr_auc'] = pr_auc_list
    result['mae'] = mae_list

    return result

def evaluate_main(generator_type, model_path, input_channel, use_3channel, mask_dir, dir_path):
    masks = ['HardExudate_Masks', 'Hemohedge_Masks', 'Microaneurysms_Masks', 'SoftExudate_Masks']
    mask_paths = [os.path.join(mask_dir, mask) for mask in masks]
    model_name = os.path.basename(os.path.dirname(model_path))
    csv_directory = "assets/evaluation/evaluation_results_csv"  # 저장할 디렉토리 경로
    csv_filename = os.path.join(csv_directory, f"{model_name}_{generator_type}_evaluation_result.csv")
    
    generator_args = {
        'dir_path':dir_path,
        'mask_path':mask_paths,
        'use_mask':True,
        'img_size':(512, 512),  
        'batch_size':4,
        'dataset':'FGADR',
        'use_3channel':use_3channel,
        'CLAHE_args':None,
    }

    if generator_type == 'train':
        generator = DR_Generator_forInference(start_end_index=(0, 1108), is_train=False, **generator_args)
    elif generator_type == 'validation':
        generator = DR_Generator_forInference(start_end_index=(1108, 1660), is_train=False, **generator_args)
    elif generator_type == 'test':
        generator = DR_Generator_forInference(start_end_index=(1660, 1840), is_train=False, **generator_args)
    else:
        raise ValueError("유형이 잘못됨. 다시 선택 : 'train', 'validation', or 'test'.")

    evaluation_result = evaluate_model(generator, model_path, input_channel)
    evaluation_result = evaluation_result.sort_values(by='dice', ascending=False)
    evaluation_result.to_csv(csv_filename, index=False)
    print(f"CSV 파일 저장 완료: {csv_filename}")
    
    return evaluation_result

if __name__ == "__main__":
    evaluate_main('train', "../models/one_mask/withoutCLAHE_withRecons_alpha01_lr00001_3channel/26")