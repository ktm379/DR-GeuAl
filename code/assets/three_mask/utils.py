import numpy as np

# history/.txt로부터 history를 불러오는 함수
# history/.txt로부터 history를 불러오는 함수
def parse_history_text(path):    
    epochs = []
    
    tr_loss = []
    tr_mask_loss = []
    tr_recons_loss = []
    tr_ex_loss = []
    tr_he_loss = []
    tr_ma_loss = []
    tr_se_loss = []
    
    
    val_loss = []
    val_mask_loss = []
    val_recons_loss = []
    val_ex_loss = []
    val_he_loss = []
    val_ma_loss = []
    val_se_loss = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            splits = line.split('/')
            
            epoch = int(splits[0][6:])
            mask_loss = float(splits[2][10:])
            recons_loss = float(splits[3][12:])
            
            ex_loss = float(splits[4][8:])
            he_loss = float(splits[5][8:])
            ma_loss = float(splits[6][8:])
            se_loss = float(splits[7][8:])
            
            if i % 2 == 0:        
                train_loss = float(splits[1][11:])
                
                epochs.append(epoch)
                tr_loss.append(train_loss)
                tr_mask_loss.append(mask_loss)
                tr_recons_loss.append(recons_loss)
                
                tr_ex_loss.append(ex_loss)
                tr_he_loss.append(he_loss)
                tr_ma_loss.append(ma_loss)
                tr_se_loss.append(se_loss)
                
            else:
                validation_loss = float(splits[1][9:])
                
                val_loss.append(validation_loss)
                val_mask_loss.append(mask_loss)
                val_recons_loss.append(recons_loss)
                
                val_ex_loss.append(ex_loss)
                val_he_loss.append(he_loss)
                val_ma_loss.append(ma_loss)
                val_se_loss.append(se_loss)
    
    history = {}
    
    history['epoch'] = epochs
    
    history['train_loss'] = tr_loss
    history['train_mask_loss'] = tr_mask_loss
    history['tr_recons_loss'] = tr_recons_loss
    history['tr_ex_loss'] = tr_ex_loss
    history['tr_he_loss'] = tr_he_loss
    history['tr_ma_loss'] = tr_ma_loss
    history['tr_se_loss'] = tr_se_loss
    
    
    history['val_loss'] = val_loss
    history['val_mask_loss'] = val_mask_loss
    history['val_recons_loss'] = val_recons_loss
    history['val_ex_loss'] = val_ex_loss
    history['val_he_loss'] = val_he_loss
    history['val_ma_loss'] = val_ma_loss
    history['val_se_loss'] = val_se_loss
    
    return history


# gaussian noise를 추가하는 함수
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image[noisy_image > 1] = 1
    noisy_image[noisy_image < 0] = 0
    
    return noisy_image