import math
import numpy as np

"""
This code is generic as it works with any model
Just be careful for output format in case of instance segmentation
"""

def apply_patch(model, patch):
    patch = np.asarray(patch)[:, :, :3] / 255.
    pred = model.predict(np.array([patch]))[0]
    return pred

def apply_model_batch(model, batch):
    batch = [np.asarray(patch)[:, :, :3] / 255. for patch in batch]
    pred = model.predict(np.array(batch))
    return list(pred)

def binary_mask(patch_mask):
    patch_mask = patch_mask * np.array([1, 1, 1.5]) # increasing border
    patch_mask = np.argmax(patch_mask, axis=2)
    return patch_mask == 1

def segment_image(model, image, patch_size=(224, 224), margin=12):
    arr = np.asarray(image)
    size = arr.shape[:2]
    if arr.ndim == 2:
        arr = arr.reshape((*arr.shape, 1))
        arr = np.repeat(arr, 3, axis=-1)
        
    # this is the size of the part we keep in the prediction (the center cropped)
    kernel_patch_size = (patch_size[0] - margin * 2, patch_size[1] - margin * 2)
    
    nb_patch_x = math.ceil(size[0] / kernel_patch_size[0])
    nb_patch_y = math.ceil(size[1] / kernel_patch_size[1])
    
    kernel_size = (nb_patch_x * kernel_patch_size[0], nb_patch_y * kernel_patch_size[1])
    
    image_margin = np.zeros((
        kernel_size[0] + margin * 2,
        kernel_size[1] + margin * 2,
        3
    ))
    
    image_margin[margin:margin+arr.shape[0], margin:margin+arr.shape[1], :] = arr[:, :, :]
    mask = np.zeros(kernel_size, dtype=np.bool) # this might be biger than input image because of `ceil`
    
    for i in range(nb_patch_x):
        for j in range(nb_patch_y):
            patch = image_margin[i * kernel_patch_size[0]: i * kernel_patch_size[0] + patch_size[0],
                                 j * kernel_patch_size[1]: j * kernel_patch_size[1] + patch_size[1]]
            
            patch_mask = apply_patch(model, patch)

            mask[i * kernel_patch_size[0]: (i + 1) * kernel_patch_size[0],
                 j * kernel_patch_size[1]: (j + 1) * kernel_patch_size[1]] \
            = binary_mask(patch_mask[margin:-margin, margin:-margin])
            
    # resize mask
    mask = mask[:arr.shape[0], :arr.shape[1]]
    
    return mask

def quick_segment(model, image, patch_size=(224, 224), margin=12):
    arr = np.asarray(image)
    size = arr.shape[:2]
    if arr.ndim == 2:
        arr = arr.reshape((*arr.shape, 1))
        arr = np.repeat(arr, 3, axis=-1)
        
    # this is the size of the part we keep in the prediction (the center cropped)
    kernel_patch_size = (patch_size[0] - margin * 2, patch_size[1] - margin * 2)
    
    nb_patch_x = math.ceil(size[0] / kernel_patch_size[0])
    nb_patch_y = math.ceil(size[1] / kernel_patch_size[1])
    
    kernel_size = (nb_patch_x * kernel_patch_size[0], nb_patch_y * kernel_patch_size[1])
    
    image_margin = np.zeros((
        kernel_size[0] + margin * 2,
        kernel_size[1] + margin * 2,
        3
    ))
    
    image_margin[margin:margin+arr.shape[0], margin:margin+arr.shape[1], :] = arr[:, :, :]
    mask = np.zeros(kernel_size, dtype=np.bool) # this might be bigger than input image because of `ceil`
    
    patch_list = []
    
    for i in range(nb_patch_x):
        for j in range(nb_patch_y):
            patch = image_margin[i * kernel_patch_size[0]: i * kernel_patch_size[0] + patch_size[0],
                                 j * kernel_patch_size[1]: j * kernel_patch_size[1] + patch_size[1]]
            
            patch_list.append(patch)
            
    mask_list = apply_model_batch(model, patch_list)
    count = 0
            
    for i in range(nb_patch_x):
        for j in range(nb_patch_y):
            mask[i * kernel_patch_size[0]: (i + 1) * kernel_patch_size[0],
                 j * kernel_patch_size[1]: (j + 1) * kernel_patch_size[1]] \
            = binary_mask(mask_list[count][margin:-margin, margin:-margin])
            
            count += 1
            
    # resize mask
    mask = mask[:arr.shape[0], :arr.shape[1]]
    
    return mask