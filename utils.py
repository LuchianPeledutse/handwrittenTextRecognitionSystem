import pickle
from typing import Tuple

from tqdm import tqdm

import datasets

import torch
import torchvision





class ViTSplit:
    """
    Class representing splitting object
    """
    def __init__(self, pic_size:tuple[int], split_size:int):
        """
        Parameters
        ----------
        pic_size: tuple[int]
            Shape of picture in form (C, H, W)
        split_size: int
            Width that each strip should have
        """
        assert pic_size[-1] % split_size == 0, "Width of the image has to be devidable by split size"
        self.split_size = split_size
        self.pic_size = pic_size

    def __call__(self, img: torch.tensor):
        """
        Returns a list of splitted pictures

        parameters
        ----------
        img: torch.tensor
            Image to be splitted
        """
        assert len(img.shape) == 3, "Picture has to have 3 dimensions"
        splitted_picture = []
        for strip_ind in range(self.pic_size[-1]//self.split_size):
            splitted_picture.append(img[:, :, strip_ind*self.split_size:(strip_ind+1)*self.split_size])
        return splitted_picture



class StatsCalc:
    """
    Class for calculating and saving statistics
    
    Parameters
    ----------
    dataset: datasets.arrow_dataset.Dataset
        Arrow dataset from HF datasets containing column 'image' with PILImages

    img_transforms: torchvision.transforms.transforms.Compose
        Transforms to apply to pictures
    """
    def __init__(self, dataset: datasets.arrow_dataset.Dataset,
                 img_transforms: torchvision.transforms.transforms.Compose,
                 img_size: Tuple[int, int]):
        self.dataset = dataset
        self.transform = img_transforms
        self.img_size = img_size
        self.statistics = {}
    
    def compute_statistics(self):
        pix_values = []
        obj_count = 0

        for img in tqdm(self.dataset["image"], desc="Iterating over objects..."):
            obj_count += 1
            tensor_picture = self.transform(img)
            assert len(tensor_picture.shape) == 3 and tensor_picture.shape[0] == 1 and tuple(tensor_picture.shape[1:]) == self.img_size, "Issue with image shape or incompatible sizes"
            pix_values.extend(tensor_picture.view(-1).tolist())
        
        global_pix_num = obj_count*self.img_size[0]*self.img_size[1]
        print("Computing statistics...")
        self.statistics["mean"] = (torch.sum(torch.tensor(pix_values))/global_pix_num).item()
        self.statistics["std"] = (torch.sqrt(torch.sum((torch.tensor(pix_values)-self.statistics["mean"])**2)/(global_pix_num - 1))).item()

    def save(self, path) -> None:
        """
        Save python dict with statistics at path
        """
        with open(path, 'wb') as pickle_file:
            pickle.dump(self.statistics, pickle_file)
'--------------------------------------------------------------------------------'

