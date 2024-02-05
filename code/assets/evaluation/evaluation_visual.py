import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from assets.one_mask.Preprocessing import preprocess_image
from assets.one_mask.models import SMD_Unet

# GPU 사용하지 않도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_and_resize_images(image_path, use_3channel=True, use_hist=False):
    image = preprocess_image(image_path, 
                            img_size=(512, 512), 
                            use_3channel=use_3channel,
                            use_hist=use_hist)
    return image

def apply_color_to_mask(mask, color):
    # 마스크의 흰색 영역을 해당 색으로 변경
    colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    indices = np.where(mask)
    for i in range(len(indices[0])):
        row, col = indices[0][i], indices[1][i]
        colored_mask[row, col] = color
    return colored_mask

def combine_masks(mask_ex, mask_he, mask_ma, mask_se):
    # 각 마스크를 색상으로 변환
    mask_ex_color = apply_color_to_mask(mask_ex, [76, 0, 153])  # 빨간색
    mask_he_color = apply_color_to_mask(mask_he, [0, 0, 255])  # 파란색
    mask_ma_color = apply_color_to_mask(mask_ma, [255, 255, 0])  # 노란색
    mask_se_color = apply_color_to_mask(mask_se, [0, 255, 0])  # 초록색

    # 색상별 마스크를 합치기
    combined_mask = mask_ex_color + mask_he_color + mask_ma_color + mask_se_color
    combined_mask[combined_mask > 255] = 255  # 최대값을 255로 제한
    return combined_mask

def extract_mask_boundary(mask):
    # 이진화된 마스크를 uint8 데이터 타입으로 변환
    mask = mask.astype(np.uint8)

    # 침식 연산을 위한 커널 생성
    kernel = np.ones((3, 3), np.uint8)

    # 침식 연산을 통해 내부를 제외한 테두리만 얻음
    eroded = cv2.erode(mask, kernel, iterations=8)

    # 테두리와 원본 마스크의 차이를 계산하여 테두리 부분 추출
    boundary = mask - eroded

    return boundary

def visualize_segmentation(image, mask_ex, mask_he, mask_ma, mask_se, mask_true, mask_pred, image_filename):
    plt.figure(figsize=(18, 12))

    # 원본 이미지
    plt.subplot(2, 5, 1)
    scaled_image = (image * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title('Original Image\n{}'.format(os.path.basename(image_filename)))
    plt.axis('off')

    # Ex 마스크 출력 (빨간색)
    plt.subplot(2, 5, 2)
    mask_ex_color = apply_color_to_mask(mask_ex, [76, 0, 153])
    plt.imshow(mask_ex_color)
    plt.title('Ex Mask\n{}'.format(os.path.basename(image_filename)))
    plt.axis('off')

    # He 마스크 출력 (파란색)
    plt.subplot(2, 5, 3)
    mask_he_color = apply_color_to_mask(mask_he, [0, 0, 255])
    plt.imshow(mask_he_color)
    plt.title('He Mask\n{}'.format(os.path.basename(image_filename)))
    plt.axis('off')

    # Ma 마스크 출력 (노란색)
    plt.subplot(2, 5, 4)
    mask_ma_color = apply_color_to_mask(mask_ma, [255, 255, 0])
    plt.imshow(mask_ma_color)
    plt.title('Ma Mask\n{}'.format(os.path.basename(image_filename)))
    plt.axis('off')

    # Se 마스크 출력 (초록색)
    plt.subplot(2, 5, 5)
    mask_se_color = apply_color_to_mask(mask_se, [0, 255, 0])
    plt.imshow(mask_se_color)
    plt.title('Se Mask\n{}'.format(os.path.basename(image_filename)))
    plt.axis('off')

    plt.tight_layout()

    # Target 및 Predicted 마스크
    plt.figure(figsize=(15, 10))

    # 실제 세그멘테이션 마스크 출력 (Target 마스크)
    mask_target_combined = combine_masks(mask_ex, mask_he, mask_ma, mask_se)
    plt.subplot(1, 2, 1)
    plt.imshow(mask_target_combined)
    plt.title('Target Mask')
    plt.axis('off')
    
    # 예측된 세그멘테이션 마스크 출력
    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(mask_pred), cmap='gray')

    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.tight_layout()
    
    
    # Original에 Predicted 마스크와 Target 마스크
    plt.figure(figsize=(15, 10))
    
    # Target 마스크 시각화
    plt.subplot(1, 2, 1)
    # 마스크를 원본 이미지 크기에 맞게 resize
    resized_mask = cv2.resize(mask_target_combined, (rgb_image.shape[1], rgb_image.shape[0]))
    # 마스크를 원본 이미지 위에 겹쳐서 시각화
    masked_image = cv2.addWeighted(rgb_image, 1, resized_mask.astype(np.uint8), 10, 0)
    plt.imshow(masked_image)
    plt.title('Original Image with Target Mask')
    plt.axis('off')

    # 원본 이미지에 예측된 세그멘테이션 마스크 겹쳐서 시각화
    plt.subplot(1, 2, 2)
    # 예측된 세그멘테이션 마스크를 numpy 배열로 변환
    predicted_mask = tf.squeeze(mask_pred).numpy()
    # 예측된 세그멘테이션 마스크를 이진화하여 0 또는 1의 값으로 변환
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    # 마스크를 원본 이미지 위에 겹쳐서 시각화
    masked_image = cv2.addWeighted(rgb_image, 1, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
    plt.imshow(masked_image)
    plt.title('Original Image with Predicted Mask')
    plt.axis('off')

    
    plt.tight_layout()
    plt.show()


def visualize_segmentation_results(image_filenames, model_path):
    mask_dir = '../data/FGADR/Seg-set'
    image_dir = '../data/FGADR/Seg-set/Original_Images/'
    masks = ['HardExudate_Masks', 'Hemohedge_Masks', 'Microaneurysms_Masks', 'SoftExudate_Masks']
    mask_paths = [os.path.join(mask_dir, mask) for mask in masks]

    # 이미지 파일들을 정렬하여 가져옴
    image_files = sorted(os.listdir(image_dir))

    model = SMD_Unet(enc_filters=[64, 128, 256, 512, 1024], dec_filters=[512, 256, 64, 32], input_channel=3)
    model.load_weights(model_path)

    for image_filename in image_filenames:
        # 이미지 파일의 인덱스 가져오기
        image_index = image_files.index(image_filename)

        # 시각화할 원본 마스크 선택 및 리사이징
        selected_ex_mask = cv2.imread(os.path.join(mask_paths[0], image_filename), cv2.IMREAD_UNCHANGED)
        selected_ex_mask = cv2.resize(selected_ex_mask, (512, 512))

        selected_he_mask = cv2.imread(os.path.join(mask_paths[1], image_filename), cv2.IMREAD_UNCHANGED)
        selected_he_mask = cv2.resize(selected_he_mask, (512, 512))

        selected_ma_mask = cv2.imread(os.path.join(mask_paths[2], image_filename), cv2.IMREAD_UNCHANGED)
        selected_ma_mask = cv2.resize(selected_ma_mask, (512, 512))

        selected_se_mask = cv2.imread(os.path.join(mask_paths[3], image_filename), cv2.IMREAD_UNCHANGED)
        selected_se_mask = cv2.resize(selected_se_mask, (512, 512))

        # 이미지 파일의 경로
        image_path = os.path.join(image_dir, image_filename)

        # 이미지 로드 및 전처리
        image = load_and_resize_images(image_path, use_3channel=True, use_hist=False)

        # 모델에 이미지 전달하여 예측
        preds = model(image[np.newaxis, ...])

        # 시각화 함수 호출 (이미지 파일명도 함께 전달)
        visualize_segmentation(image, selected_ex_mask, selected_he_mask, selected_ma_mask, selected_se_mask, None, preds[1], image_filename)
        print("============================================================================================================================")


if __name__ == "__main__":
    image_filenames = ["0381_1.png", "0311_1.png", "1134_1.png", "1181_3.png"]  # 원하는 이미지 파일명으로 수정
    model_path = "../models/one_mask/withoutCLAHE_withRecons_alpha01_lr00001_3channel/26"
    visualize_segmentation_results(image_filenames, model_path)
