import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_and_resize_image(image, img_size, use_3channel=True):
    '''
    각 채널을 받아 크기 조정(512x512 픽셀로 다운샘플링), 
    이미지 크롭 수행(FOV의 직경과 동일한 정사각형),
    각 채널 합치기
    
    img_size=(512, 512)
    '''
    # 3개의 채널 전부 전처리할 경우
    if use_3channel:

        # 크기 조정
        cropped_image = cv2.resize(image, img_size, interpolation=cv2.INTER_CUBIC)

    # 녹색 채널만 전처리할 경우
    else:
        resized_image = cv2.resize(image, img_size)
        diameter = min(resized_image.shape[0], resized_image.shape[1])
        center_x, center_y = resized_image.shape[1] // 2, resized_image.shape[0] // 2
        crop_size = min(center_x, center_y, diameter // 2)
        cropped_image = resized_image[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]
        
    return cropped_image

def apply_clahe(image, clahe):
    # LAB 색 공간으로 변환
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 각 채널에 CLAHE 적용
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    
    # BGR 색 공간으로 다시 변환
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_image

def preprocess_image(image_path, img_size=(512, 512), use_hist=True, use_3channel=True, CLAHE_args=None):
    '''
    image_path를 받아 이미지를 읽고, 
    각 채널에 대한 대비 향상, 
    크롭 및 크기 조정
    '''
    if CLAHE_args != None:
        clipLimit, tileGridSize = CLAHE_args
    
    original_image = cv2.imread(image_path)

    if use_3channel:
        
        if use_hist:
            # 대비 향상 적용
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            contrast_enhanced_image = apply_clahe(original_image, clahe)

            # 크롭 및 크기 조정
            cropped_image = crop_and_resize_image(contrast_enhanced_image, img_size, use_3channel=True)
        
        else:
            cropped_image = crop_and_resize_image(original_image, img_size, use_3channel=True)
    else:
        # 녹색 채널만 사용
        green_channel = original_image[:, :, 1]

        # 대비 향상 적용
        if use_hist:
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            contrast_enhanced_image = clahe.apply(green_channel)
            cropped_image = crop_and_resize_image(contrast_enhanced_image, img_size, use_3channel=False)
        else:
            cropped_image = crop_and_resize_image(green_channel, img_size, use_3channel=False)

    # 0~1로 scale 맞추기
    cropped_image = cropped_image.astype(np.float32) / 255.0
      
    return cropped_image
