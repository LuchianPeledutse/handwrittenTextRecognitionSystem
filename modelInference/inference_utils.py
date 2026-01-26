import sys
sys.path.append(r"c:\main\GitHub\handwrittenTextRecognitionSystem")

from typing import Callable

import PIL

from utils import ViTSplit

import numpy as np

from transformers import PreTrainedTokenizerFast

import torch
import torch.nn as nn
from torchvision import transforms




class ModelInference:
    def __init__(self, model: nn.Module, transformations: transforms.transforms.Compose,
                 tokenizer: PreTrainedTokenizerFast, split: ViTSplit,
                 soft_func: Callable[[torch.tensor], torch.tensor] = nn.Softmax(dim=-1),
                 max_seq_length:int = 86, device: str = 'cpu'):
        self.model = model
        self.split = split
        self.device = device
        self.soft_func = soft_func
        self.tokenizer = tokenizer
        self.transform = transformations
        self.max_seq_length = max_seq_length
        self.initial_mask = model.decoder.mask
    
    @property
    def sos(self):
        return self.tokenizer.get_vocab()["<sos>"]
    
    @property
    def eos(self):
        return self.tokenizer.get_vocab()["<eos>"]
    
    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def decoder(self):
        return self.model.decoder
    
    @property
    def linear(self):
        return self.model.linear
    
    def update_decoder_mask(self, mask: torch.tensor):
        for decoder_layer in self.decoder.layers:
            decoder_layer.mask = mask

    def generate(self, picture: PIL.JpegImagePlugin.JpegImageFile):
        """
        Given a single picture transforms it for ViT input and
        generates text based on it
        """
        assert picture.mode == "L", "Picture should be in gray mode"
        # Move model to evalutaion mode for inference
        self.model.eval()
        # Picture preparation for model inputing
        tensor_img = self.transform(picture)
        splitted_image = torch.cat(self.split(tensor_img), dim=0).unsqueeze(dim=0)
        # Flatten patches to get encoder representations
        flattened_patches = splitted_image.flatten(start_dim=-2, end_dim=-1).to(device=self.device)
        encoder_features = self.encoder(flattened_patches)
        # Generate index by index
        gen_indecies = [self.sos]
        while gen_indecies[-1] != self.eos and len(gen_indecies) < self.max_seq_length:
            N_seq = len(gen_indecies)
            tensor_indecies = torch.tensor(gen_indecies, dtype=torch.long).unsqueeze(dim=0).to(device=self.device)
            # Each time depending on sequence length the decoder mask dimension has to be changed accordingly
            new_mask = torch.triu(torch.ones(N_seq, N_seq).to(device=self.device)).T
            self.update_decoder_mask(new_mask)
            # Get decoder features to predict next token
            decoder_features = self.decoder(tensor_indecies, x_dec=encoder_features).squeeze(dim=0)
            vocab_pre_distribution = self.linear(decoder_features)
            distribution = self.soft_func(vocab_pre_distribution)
            # Add generated from distribution token and continue
            generated_token = distribution[-1].argmax().item()
            gen_indecies.append(generated_token)
        # Return decoder to previous mask
        self.update_decoder_mask(self.initial_mask)
        return self.tokenizer.decode(gen_indecies, skip_special_tokens=True)

'--------------------------------------------------------------------------------'
    


