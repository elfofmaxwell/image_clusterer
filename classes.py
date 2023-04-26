import os
from typing import Tuple

from PIL import Image
import cv2 as cv
import numpy as np

class ImageFile(): 
    def __init__(self, path: str) -> None:
        self.path = path
        self.dir = os.path.split(path)[0]
        self.fname = os.path.split(path)[1]
        img_size: Tuple[int] = Image.open(path).size
        self.width: int = img_size[0]
        self.height: int = img_size[1]
        self.contents: np.ndarray | None = None # flattened array
        self.label: int | None = None
        self.probability: float | None = None
    
    def read_contents(self, grey_scale: bool, resize_factor: float | None) -> None: 
        if grey_scale: 
            color_flag = cv.IMREAD_GRAYSCALE
        else: 
            color_flag = cv.IMREAD_COLOR
        contents = cv.imread(self.path, color_flag)
        if resize_factor: 
            contents = cv.resize(contents, (0, 0), fx=resize_factor, fy=resize_factor)
        self.contents = contents.flatten()