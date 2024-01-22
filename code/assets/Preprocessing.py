import cv2

def crop_and_resize_image(image, img_size, use_3channel=False):
    '''
    각 채널을 받아 크기 조정(512x512 픽셀로 다운샘플링), 
    이미지 크롭 수행(FOV의 직경과 동일한 정사각형),
    각 채널 합치기
    
    img_size=(512, 512)
    '''
    # 3개의 채널 전부 전처리할 경우
    if use_3channel:
        channels = [cv2.resize(image[:, :, i], img_size) for i in range(3)]
        diameter = min(channels[0].shape[0], channels[0].shape[1])
        center_x, center_y = channels[0].shape[1] // 2, channels[0].shape[0] // 2
        crop_size = min(center_x, center_y, diameter // 2)
        cropped_channels = [channel[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size] for channel in channels]
        cropped_image = cv2.merge(cropped_channels)
    
    # 녹색 채널만 전처리할 경우
    else:
        resized_image = cv2.resize(image, img_size)
        diameter = min(resized_image.shape[0], resized_image.shape[1])
        center_x, center_y = resized_image.shape[1] // 2, resized_image.shape[0] // 2
        crop_size = min(center_x, center_y, diameter // 2)
        cropped_image = resized_image[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]
        
    return cropped_image

def preprocess_image(image_path, img_size=(512, 512), use_hist=True, use_3channel=False):
    '''
    image_path를 받아 이미지를 읽고, 
    각 채널에 대한 대비 향상, 
    크롭 및 크기 조정
    '''
    original_image = cv2.imread(image_path)

    if use_3channel:
        # 모든 채널 가져오기
        channels = [original_image[:, :, i] for i in range(3)]

        # 대비 향상 적용
        if use_hist:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast_enhanced_channels = [clahe.apply(channel) for channel in channels]

            # 각 채널 합치기
            contrast_enhanced_image = cv2.merge(contrast_enhanced_channels)
            cropped_image = crop_and_resize_image(contrast_enhanced_image, img_size, use_3channel=True)
        else:
            # 각 채널 합치기
            cropped_image = crop_and_resize_image(original_image, img_size, use_3channel=True)
    else:
        # 녹색 채널만 사용
        green_channel = original_image[:, :, 1]

        # 대비 향상 적용
        if use_hist:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast_enhanced_image = clahe.apply(green_channel)
            cropped_image = crop_and_resize_image(contrast_enhanced_image, img_size, use_3channel=False)
        else:
            cropped_image = crop_and_resize_image(green_channel, img_size, use_3channel=False)

    # 0~1로 scale 맞추기
    cropped_image = cropped_image / 255.0
      
    return cropped_image
