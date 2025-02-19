import torch
import numpy as np
import os
import cv2
import argparse
import torchvision

def load_model():
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False, num_classes=91,
        pretrained_backbone=False
    )

    state_dict = torch.load('ml/models/retinanet_resnet50_fpn_coco-eeacb38b.pth')
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    return model

def detect_image(image_path):
    model = load_model()

    image = cv2.imread(image_path)

    if image is None:
        print('ERROR: Image not found')

    image_orig = image.copy()

    rows, cols, cns = image.shape
    
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side
    
    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
        
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    
    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape
    
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))
    
    with torch.no_grad():
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        
        print(image.shape)

detect_image('input_images/carvision.png')
detect_image('input_images/objects.jpg')
