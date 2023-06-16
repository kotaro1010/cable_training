import os
import random

import numpy as np
import torch


def fix_seed(seed):
    # random
    random.seed(seed)
    # OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Tensorflow
    # tf.random.set_seed(seed)
