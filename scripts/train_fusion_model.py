import os
import pickle
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from models.embedding.fusion import EmbeddingFusion
from models.metric_learning.trainer import MetricLearningTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

##### WROK IN PROGRESS 