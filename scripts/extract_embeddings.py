import os
import pickle
import json
import logging
import torch
from tqdm import tqdm
from typing import Dict, List, Any

from models.embedding.clip_extractor import CLIPExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

##### WORK IN PROGRESS 