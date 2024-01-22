

# history/.txt로부터 history를 불러오는 함수
def parse_history_text(path):    
    epochs = []
    
    tr_loss = []
    tr_mask_loss = []
    tr_recons_loss = []
    
    val_loss = []
    val_mask_loss = []
    val_recons_loss = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            splits = line.split('/')
            
            epoch = int(splits[0][6:])
            mask_loss = float(splits[2][10:])
            recons_loss = float(splits[3][12:])
            
            if i % 2 == 0:        
                train_loss = float(splits[1][11:])
                
                epochs.append(epoch)
                tr_loss.append(train_loss)
                tr_mask_loss.append(mask_loss)
                tr_recons_loss.append(recons_loss)
            else:
                validation_loss = float(splits[1][9:])
                
                val_loss.append(validation_loss)
                val_mask_loss.append(mask_loss)
                val_recons_loss.append(recons_loss)
    
    history = {}
    
    history['epoch'] = epochs
    
    history['train_loss'] = tr_loss
    history['train_mask_loss'] = tr_mask_loss
    history['tr_recons_loss'] = tr_recons_loss
    
    history['val_loss'] = val_loss
    history['val_mask_loss'] = val_mask_loss
    history['val_recons_loss'] = val_recons_loss
    
    return history