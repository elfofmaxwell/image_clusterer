import os
from typing import List, Literal
import argparse

import yaml

import utils
import clustering
from classes import ImageFile

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        prog="python main.py", 
        description="Clustering variations of images together, based on PCA " 
            "and hdbscan"
    )

    parser.add_argument(
        '-c', 
        '--config', 
        default="config.yml", 
        help="manually specify configuration file path"
    )

    args = parser.parse_args()

    with open(args.config) as f: 
        config_dict: dict = yaml.safe_load(f)
    
    source_dir: str = config_dict['source_dir']
    dest_dir: str = config_dict['dest_dir']
    representive_img: Literal[0, -1] = config_dict['representive_img']
    n_jobs: int = config_dict['n_jobs']
    probability_threshold: float = config_dict['clustering']['probability_threshold']
    n_pca: int = config_dict['clustering']['n_pca']
    min_cluster_size: int = config_dict['clustering']['min_cluster_size']
    normalize: bool = config_dict['clustering']['normalize']
    resize: bool = config_dict['clustering']['resize']
    resize_factor: float | None = config_dict['clustering']['resize_factor'] if resize else None
    copy_or_move: Literal['copy', 'move'] = config_dict['copy_or_move']
    grey_scale: bool = config_dict['grey_scale']


    if not os.path.isdir(dest_dir): 
        os.makedirs(dest_dir)

    images: List[ImageFile] = []
    for img_path in filter(
        utils.validate_img, 
        [os.path.join(source_dir, path) for path in os.listdir(source_dir)]
    ): 
        images.append(ImageFile(img_path))
    
    grouped_images = utils.image_group_by_size(images)
    while grouped_images: 
        image_group = grouped_images.pop()
        if len(image_group) < n_pca: 
            for img in image_group: 
                img.label = -1
            clustered_images = image_group
        else: 
            for img in image_group: 
                img.read_contents(grey_scale, resize_factor)
            clustered_images = clustering.cluster_images(
                image_group, 
                n_pca, 
                min_cluster_size=2, 
                prob_thre=probability_threshold, 
                normalize=normalize, 
                n_jobs=n_jobs
            )
        utils.cp_clustered_images(
            clustered_images, dest_dir, representive_img, copy_or_move
        )
        del image_group