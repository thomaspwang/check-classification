""" 
Wraps the LLaVA library into a callable class, which allows one to feed in a prompt and an image
and recieve a text response.

Example:
    model = LLaVA()
    model.eval("../data/image.jpg", "print the text on this image")

Classes:
    LLaVA:
        Wraps the llava library. The only functions meant to be used are __init__ and eval.

        Args:
            model_path: This path will be checked first locally and then on huggingface. A full list
            of huggingface models usable is listed here: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md

            temperature: how random / non-deterministic generated output should be.
            max_new_tokens: how many words the model is allowed to generate.
            quantize: when True, the 4-bit quantized model is used, reducing memory consumption by 75%.

"""

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

class LLaVA:
    def __init__(self, model_path, model_base=None, conv_mode=None, temperature=0, top_p=None, num_beams=1, max_new_tokens=512, quantize=True):
        disable_torch_init()
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_4bit=quantize
        )

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def process_query(self, query, args_conv_mode, model_name):
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if True or self.model.config.mm_use_im_start_end: # i think this is for first msg
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if True or self.model.config.mm_use_im_start_end: # i think this is for first msg
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args_conv_mode is not None and conv_mode != args_conv_mode:
            raise(Exception("This exception should not be thrown"))
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args_conv_mode, args_conv_mode
                )
            )
        else:
            args_conv_mode = conv_mode

        conv = conv_templates[args_conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()


    def eval(self, image_file, query):
        prompt = self.process_query(query, self.conv_mode, self.model_name)
        images = [self.load_image(image_file)]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs