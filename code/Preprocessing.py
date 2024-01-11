import cv2
import numpy as np

def crop_and_resize_image(channels, img_size):
    '''
    각 채널을 받아 크기 조정(512x512 픽셀로 다운샘플링), 
    이미지 크롭 수행(FOV의 직경과 동일한 정사각형),
    각 채널 합치기
    
    img_size=(512, 512)
    '''
    resized_channels = [cv2.resize(channel, img_size) for channel in channels]

    diameter = min(resized_channels[0].shape[0], resized_channels[0].shape[1])
    center_x, center_y = resized_channels[0].shape[1] // 2, resized_channels[0].shape[0] // 2
    crop_size = min(center_x, center_y, diameter // 2)
    cropped_channels = [channel[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size] for channel in resized_channels]

    cropped_image = cv2.merge(cropped_channels)

    return cropped_image
  
def preprocess_image(image_path, img_size=(512, 512)):
    '''
    image_path를 받아 이미지를 읽고, 
    각 채널에 대한 대비 향상, 
    크롭 및 크기 조정
    '''
    original_image = cv2.imread(image_path)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced_channels = [clahe.apply(original_image[:, :, i]) for i in range(3)]

    cropped_image = crop_and_resize_image(contrast_enhanced_channels, img_size)

    return cropped_image