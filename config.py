from enum import Enum
import orjson
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Union
from datetime import datetime

import torch

__all__ = [
    "BaseConfig",
    "OptimizerType",
    "OptimizerConfig",
    "LRSchedulerType",
    "LRSchedulerConfig",
    "ModelConfig",
    "TrainConfig",
]


@dataclass
class BaseConfig:
    exp_name: str = "default_exp"
    seed: int = 42
    deterministic: bool = False
    # 自动生成带毫秒的时间戳，orjson 能直接序列化 datetime 对象
    timestamp: datetime = field(default_factory=datetime.now)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_path = os.path.join(save_directory, "config.json")

        # orjson.dumps 返回 bytes
        # OPT_INDENT_2: 缩进
        # OPT_SERIALIZE_DATACLASS: 直接序列化 dataclass (虽然我们用了 asdict)
        # OPT_APPEND_NEWLINE: 文件末尾加换行符
        binary_data = orjson.dumps(
            asdict(self),
            option=orjson.OPT_INDENT_2
            | orjson.OPT_APPEND_NEWLINE
            | orjson.OPT_SERIALIZE_NUMPY,
        )

        with open(file_path, "wb") as f:  # 注意必须以二进制模式 'wb' 写入
            f.write(binary_data)

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, os.PathLike]):
        file_path = os.path.join(load_directory, "config.json")
        with open(file_path, "rb") as f:  # 以二进制模式 'rb' 读取
            config_dict = orjson.loads(f.read())
        return cls(**config_dict)


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


@dataclass
class OptimizerConfig:
    # 默认为 AdamW
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    lr: float = 1e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9  # 仅针对 SGD 等有效

    def get_optimizer(self, params) -> torch.optim.Optimizer:
        """
        根据当前配置实例化 PyTorch 优化器
        params: 模型参数 model.parameters()
        """
        # 映射表
        maps = {
            OptimizerType.ADAM: torch.optim.Adam,
            OptimizerType.ADAMW: torch.optim.AdamW,
            OptimizerType.SGD: torch.optim.SGD,
            OptimizerType.RMSPROP: torch.optim.RMSprop,
        }

        opt_cls = maps.get(self.optimizer_type)
        # 3. 核心修复：处理 None 情况，让 Pylance 知道 opt_cls 此时绝不是 None
        if opt_cls is None:
            raise ValueError(
                f"Unsupported optimizer type: {self.optimizer_type}. "
                f"Supported types are: {list(maps.keys())}"
            )
        # 过滤掉不属于该优化器的参数（例如 SGD 需要 momentum 而 AdamW 不需要）
        # 这里可以通过 inspect 检查构造函数，或者简单地按需传递
        kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        if self.optimizer_type == OptimizerType.SGD:
            kwargs["momentum"] = self.momentum

        return opt_cls(params, **kwargs)


class LRSchedulerType(str, Enum):
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    LINEAR_WARMUP = "linear_warmup"  # 模拟 Transformer 常用的线性预热


@dataclass
class LRSchedulerConfig:
    scheduler_type: LRSchedulerType = LRSchedulerType.COSINE

    # 通用参数
    warmup_steps: int = 0

    # StepLR 参数
    step_size: int = 30
    gamma: float = 0.1

    # CosineAnnealing 参数
    t_max: int = 100  # 通常对应 total_epochs 或 total_steps
    eta_min: float = 1e-6

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int = 0
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        工厂方法：根据配置实例化调度器
        """
        if self.scheduler_type == LRSchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            )

        elif self.scheduler_type == LRSchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        elif self.scheduler_type == LRSchedulerType.COSINE:
            # 如果配置里没写 t_max，就用传入的 total_steps
            t_max = total_steps if total_steps is not None else self.t_max
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=self.eta_min
            )

        elif self.scheduler_type == LRSchedulerType.LINEAR_WARMUP:
            # 这里简单演示一个自定义 LambdaLR 实现线性预热
            def lr_lambda(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return 1.0

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


@dataclass
class ModelConfig(BaseConfig):
    hidden_size: int = 512
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[LRSchedulerConfig] = None

    def __post_init__(self):
        # 1. 转换 Optimizer
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        # 2. 转换 Scheduler
        if isinstance(self.scheduler, dict):
            self.scheduler = LRSchedulerConfig(**self.scheduler)


@dataclass
class TrainConfig(BaseConfig):
    # 实验基本信息（继承自 BaseConfig）
    exp_name: str = "ResNet_Baseline"
    # 训练全局参数
    epochs: int = 100
    batch_size: int = 128
    save_interval: int = 10

    # 组合子配置类
    # 使用 default_factory 确保每个实例拥有独立的子配置对象
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[LRSchedulerConfig] = None

    def __post_init__(self):
        # 1. 转换 Optimizer
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        # 2. 转换 Scheduler
        if isinstance(self.scheduler, dict):
            self.scheduler = LRSchedulerConfig(**self.scheduler)


if __name__ == "__main__":
    # 1. 初始化大配置
    # config = TrainConfig(
    #     exp_name="MINE_Sparsity_Study",
    #     optimizer=OptimizerConfig(optimizer_type=OptimizerType.ADAMW, lr=3e-4),
    #     scheduler=None,
    # )
    config = TrainConfig.from_pretrained("config_test")
    print(config)
    model = torch.nn.Linear(10, 2)
    optimizer = config.optimizer.get_optimizer(model.parameters())
    config.save_pretrained("config_test")
    print(config.optimizer.lr)
    # scheduler = config.scheduler.get_scheduler(optimizer, total_steps=config.total_epochs)
    # b = ModelConfig.from_pretrained("config_test")
    # print(b)
    # model = torch.nn.Linear(10, 2)
    # optimizer = b.optimizer.get_optimizer(model.parameters())
    # print(optimizer)
    # b.save_pretrained("config_test")
