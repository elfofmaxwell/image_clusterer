from typing import List, Literal
import os

import shutil
from PIL import Image

from classes import ImageFile

def image_sort_by_size(images: List[ImageFile]) -> List[ImageFile]: 
    images.sort(key=lambda img: img.fname)
    images.sort(key=lambda img: img.height)
    images.sort(key=lambda img: img.width)
    return images

def image_group_by_size(images: List[ImageFile]) -> List[List[ImageFile]]: 
    grouped_images: List[List[ImageFile]] = []
    sorted_images = image_sort_by_size(images)
    
    break_point: int = 0
    for idx in range(len(sorted_images)): 
        if idx == len(sorted_images)-1: 
            grouped_images.append(sorted_images[break_point:idx+1])
        elif (sorted_images[idx].width != sorted_images[idx+1].width or 
            sorted_images[idx].height != sorted_images[idx+1].height): 
            grouped_images.append(sorted_images[break_point:idx+1])
            break_point = idx+1
    return grouped_images

def cp_clustered_images(
        images: List[ImageFile], 
        dest_dir: str, 
        representative_img: Literal[-1, 0], 
        copy_or_move: Literal['copy', 'move']
    ) -> None: 
    if copy_or_move == "copy": 
        file_op = shutil.copyfile
    elif copy_or_move == "move": 
        file_op = os.rename
    else: 
        raise ValueError("Invalid copy_or_move")
    max_label = max([im.label for im in images])
    if max_label >= 0: 
        for label in range(max_label+1): 
            images_w_label = list(filter(lambda im: im.label == label, images))
            label_dir = os.path.join(
                dest_dir, 
                os.path.splitext(images_w_label[representative_img].fname)[0]
            )
            if not os.path.isdir(label_dir): 
                os.makedirs(label_dir)
            
            shutil.copyfile(
                images_w_label[representative_img].path, 
                os.path.join(dest_dir, images_w_label[representative_img].fname)
            )

            for im in images_w_label: 
                file_op(im.path,os.path.join(label_dir, im.fname))
    
    noise_images = list(filter(lambda im: im.label == -1, images))
    for im in noise_images:
        file_op(im.path, os.path.join(dest_dir, im.fname))

def validate_img(img_path: str) -> bool: 
    try: 
        img_format = Image.open(img_path).format.lower()
        return (img_format in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'])
    except: 
        return False