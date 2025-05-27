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
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import torch.utils.data as data
from transformers import AutoTokenizer
import random
import pickle
from typing import Dict, List, Tuple, Optional
import torch.jit

# Cache for XML parsing
XML_CACHE: Dict[str, List[Tuple[str, np.ndarray, str]]] = {}

@torch.jit.script
def process_bbox(bbox: torch.Tensor, ratio: float, dw: float, dh: float) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
    bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
    return bbox

@torch.jit.script
def letterbox_tensor(img: torch.Tensor, mask: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    h, w = img.shape[1:3]
    ratio = min(size / h, size / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Resize using F.interpolate
    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)
    
    # Calculate padding
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    
    # Pad using F.pad
    img = F.pad(img, (int(dw), int(size - new_w - dw), int(dh), int(size - new_h - dh)))
    mask = F.pad(mask, (int(dw), int(size - new_w - dw), int(dh), int(size - new_h - dh)))
    
    return img, mask, ratio, dw, dh

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

def parse_xml_cache(anno_path: str) -> List[Tuple[str, np.ndarray, str]]:
    if anno_path in XML_CACHE:
        return XML_CACHE[anno_path]
        
    results = []
    annotations = filelist(anno_path, '.xml')
    for xml_path in annotations:
        root = ET.parse(xml_path).getroot()
        for member in root.findall('object'):
            imageFile = root.find("./filename").text
            box = np.array([int(member[2][0].text), int(member[2][1].text), 
                          int(member[2][2].text), int(member[2][3].text)], dtype=np.float32)
            text = member[3].text
            results.append((imageFile, box, text))
            
    XML_CACHE[anno_path] = results
    return results

class RSVGDataset(data.Dataset):
    def __init__(self, images_path: str, anno_path: str, imsize: int = 640, 
                 transform: Optional[callable] = None, augment: bool = False,
                 split: str = 'train', testmode: bool = False, 
                 max_query_len: int = 40, bert_model: str = 'vinai/phobert-base-v2'):
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
        self.cache = {}
        self.text_cache = {}  # Cache for tokenized text
        
        # Load split indices
        with open(f'/kaggle/input/diorvn/dior/{split}.txt', "r") as f:
            indices = set(int(index.strip()) for index in f.readlines())
            
        # Parse XML with caching
        all_annotations = parse_xml_cache(anno_path)
        count = 0
        for imageFile, box, text in all_annotations:
            if count in indices:
                self.images.append((os.path.join(images_path, imageFile), box, text))
            count += 1
            
        # Pre-tokenize all texts
        self._pre_tokenize_texts()
        
    def _pre_tokenize_texts(self):
        """Pre-tokenize all texts to avoid doing it in __getitem__"""
        for idx, (_, _, text) in enumerate(self.images):
            if idx not in self.text_cache:
                examples = read_examples(text.lower(), idx)
                features = convert_examples_to_features(
                    examples=examples, 
                    seq_length=self.query_len,
                    tokenizer=self.tokenizer
                )
                self.text_cache[idx] = (
                    np.array(features[0].input_ids, dtype=int),
                    np.array(features[0].input_mask, dtype=int)
                )

    def pull_item(self, idx: int) -> Tuple[np.ndarray, str, torch.Tensor]:
        img_path, bbox, phrase = self.images[idx]
        bbox = torch.from_numpy(bbox).float()
        
        # Read image as numpy array, don't convert to tensor yet
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, phrase, bbox

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
            
        img, phrase, bbox = self.pull_item(idx)
        
        # Create mask as numpy array
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        
        # Process image and mask using numpy operations first
        img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
        
        # Convert to tensor after letterbox
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)
        
        # Apply transforms if any
        if self.transform is not None:
            img = self.transform(img)
        
        # Process bbox
        bbox = process_bbox(bbox, ratio, dw, dh)
        
        # Get pre-tokenized text
        word_id, word_mask = self.text_cache[idx]
        word_id = torch.from_numpy(word_id)
        word_mask = torch.from_numpy(word_mask)
        
        if self.testmode:
            result = (img, mask, word_id, word_mask, bbox, 
                     torch.tensor(ratio), torch.tensor(dw), torch.tensor(dh),
                     self.images[idx][0], phrase)
        else:
            result = (img, mask, word_id, word_mask, bbox)
            
        self.cache[idx] = result
        return result

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    if len(batch[0]) == 5:  # Training mode
        imgs, masks, word_ids, word_masks, bboxes = zip(*batch)
        return (
            torch.stack(imgs),
            torch.stack(masks),
            torch.stack(word_ids),
            torch.stack(word_masks),
            torch.stack(bboxes)
        )
    else:  # Test mode
        imgs, masks, word_ids, word_masks, bboxes, ratios, dws, dhs, paths, phrases = zip(*batch)
        return (
            torch.stack(imgs),
            torch.stack(masks),
            torch.stack(word_ids),
            torch.stack(word_masks),
            torch.stack(bboxes),
            torch.stack(ratios),
            torch.stack(dws),
            torch.stack(dhs),
            paths,
            phrases
        )

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

def train_epoch(train_loader, model, optimizer, epoch):
    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        with torch.cuda.stream(stream1):
            # Transfer data to GPU
            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            
        with torch.cuda.stream(stream2):
            # Transfer other data to GPU
            word_id = word_id.cuda(non_blocking=True)
            word_mask = word_mask.cuda(non_blocking=True)
            gt_bbox = gt_bbox.cuda(non_blocking=True)
            
        # Synchronize streams
        torch.cuda.synchronize()
        
        # ... rest of training code ...


