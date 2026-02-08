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
                 max_seq_length:int = 86, device: str = 'cuda'):
        self.model = model
        self.split = split
        self.device = device
        self.soft_func = soft_func
        self.tokenizer = tokenizer
        self.transform = transformations
        self.max_seq_length = max_seq_length
    
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

    def generate(self, picture: PIL.JpegImagePlugin.JpegImageFile):
        """
        Given a single picture transforms it for ViT input and
        generates text based on it
        """
        picture = picture.convert("RGB")
        assert picture.mode == "RGB", "Picture should be in gray mode"
        # Move model to evalutaion mode for inference
        self.model.eval()
        # Picture preparation for model inputing
        tensor_img = self.transform(picture, return_tensors="pt").pixel_values.to(device=self.device)
        # Flatten patches to get encoder representations
        encoder_features = self.encoder(tensor_img)["last_hidden_state"]
        # Generate index by index
        gen_indecies = [self.sos]
        while gen_indecies[-1] != self.eos and len(gen_indecies) < self.max_seq_length:
            N_seq = len(gen_indecies)
            tensor_indecies = torch.tensor(gen_indecies, dtype=torch.long).unsqueeze(dim=0).to(device=self.device)
            # Get decoder features to predict next token
            decoder_features = self.decoder(tensor_indecies, encoder_hidden_states=encoder_features).logits.squeeze(dim=0)
            distribution = self.soft_func(decoder_features)
            # Add generated from distribution token and continue
            generated_token = distribution[-1].argmax().item()
            gen_indecies.append(generated_token)
        return self.tokenizer.decode(gen_indecies, skip_special_tokens=True)


    


