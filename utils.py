import torch




class ViTSplit:
    def __init__(self, pic_size:tuple[int], split_size:int):
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
        assert len(img.shape) == 3, 'Picture has to have 3 dimensions'
        splitted_picture = []
        for strip_ind in range(self.pic_size[-1]//self.split_size):
            splitted_picture.append(img[:, :, strip_ind*self.split_size:(strip_ind+1)*self.split_size])
        return splitted_picture

