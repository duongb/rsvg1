#!/usr/bin/env python
# -*-coding: utf -8-*-
"""
@ Author: ZhanYang
@ File Name: data_loader.py
@ Email: zhanyang@mail.nwpu.edu.cn
@ Github: https://github.com/ZhanYang-nwpu/RSVG-pytorch
@ Paper: https://ieeexplore.ieee.org/document/10056343
@ Dataset: https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing
"""
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
import utils
from utils.transforms import letterbox

import matplotlib.pyplot as plt
import torch.utils.data as data
#from transformers import BertTokenizer
from transformers import AutoTokenizer
import random

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=640, transform=None, augment=False,
                 split='train', testmode=False, max_query_len=40, bert_model='vinai/phobert-base-v2'):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        self.query_len = max_query_len
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        # Cache for images and tokenized text
        self.image_cache = {}
        self.token_cache = {}
        
        # Parse and cache XML annotations
        split_file = os.path.join(os.path.dirname(anno_path), split + '.txt')
        file = open(split_file, "r").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        
        # Cache XML parsing results
        xml_cache = {}
        annotations = filelist(anno_path, '.xml')
        
        # Pre-load all images in background
        def preload_images():
            for anno_path in annotations:
                if anno_path not in xml_cache:
                    root = ET.parse(anno_path).getroot()
                    xml_cache[anno_path] = root
                else:
                    root = xml_cache[anno_path]
                    
                for member in root.findall('object'):
                    if count in Index:
                        imageFile = str(images_path) + '/' + root.find("./filename").text
                        if imageFile not in self.image_cache:
                            try:
                                img = cv2.imread(imageFile)
                                if img is not None:
                                    # Pre-resize image to target size
                                    h, w = img.shape[:2]
                                    mask = np.zeros_like(img)
                                    img, mask, _, _, _ = letterbox(img, mask, self.imsize)
                                    self.image_cache[imageFile] = img
                            except Exception as e:
                                print(f"Error loading image {imageFile}: {e}")
                                
                        box = np.array([int(member[2][0].text), int(member[2][1].text), 
                                      int(member[2][2].text), int(member[2][3].text)], dtype=np.float32)
                        text = member[3].text
                        self.images.append((imageFile, box, text))
                    count += 1
        
        # Start preloading in background
        import threading
        preload_thread = threading.Thread(target=preload_images)
        preload_thread.daemon = True
        preload_thread.start()
        
        # Pre-tokenize all texts
        for idx, (_, _, text) in enumerate(self.images):
            if text not in self.token_cache:
                examples = read_examples(text.lower(), idx)
                features = convert_examples_to_features(examples=examples, 
                                                      seq_length=self.query_len,
                                                      tokenizer=self.tokenizer)
                self.token_cache[text] = features[0]

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)
        
        # Use cached image if available
        if img_path in self.image_cache:
            img = self.image_cache[img_path]
        else:
            # If not in cache, load and resize
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                mask = np.zeros_like(img)
                img, mask, _, _, _ = letterbox(img, mask, self.imsize)
                self.image_cache[img_path] = img
            
        return img, phrase, bbox

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        phrase_out = phrase

        # Image is already resized in cache
        h, w = img.shape[:2]
        mask = np.zeros_like(img)

        # No need to resize again since it's done in cache
        ratio = self.imsize / max(h, w)
        dw = (self.imsize - w * ratio) / 2
        dh = (self.imsize - h * ratio) / 2
        
        bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
        bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

        if self.transform is not None:
            img = self.transform(img)

        # Use cached tokenization
        features = self.token_cache[phrase]
        word_id = features.input_ids
        word_mask = features.input_mask
        word_split = features.tokens[1:-1]

        if self.testmode:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                   np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                   np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], phrase_out
        else:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)

    def __len__(self):
        return len(self.images)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


