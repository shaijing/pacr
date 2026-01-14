import os
import random
import warnings

import numpy as np
import torch


def setup_seed(seed, deterministic: bool = False):
    # random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        # 如果使用了复杂的线性代数运算，可能还需要：
        torch.use_deterministic_algorithms(False)
        # warnings.warn(
        #     "You have chosen to seed training. "
        #     "This will turn on the CUDNN deterministic setting, "
        #     "which can slow down your training considerably! "
        #     "You may see unexpected behavior when restarting "
        #     "from checkpoints."
        # )
