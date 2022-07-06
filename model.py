import torch
import re
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 

from helpers import url_to_img


device = "cpu"
version = "nlpconnect/vit-gpt2-image-captioning"


class ImageCaptioning:

    def __init__(self):
        encoder_checkpoint = version
        decoder_checkpoint = version
        model_checkpoint = version
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

    def run(self, image_url):
        image = url_to_img(image_url)
        image = self.feature_extractor(image, return_tensors="pt").pixel_values.to(device)
        clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
        caption_ids = self.model.generate(image, max_length = 64)[0]
        caption_text = clean_text(self.tokenizer.decode(caption_ids))
        return caption_text 