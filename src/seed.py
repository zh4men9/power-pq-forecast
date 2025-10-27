"""
随机种子设置工具
用于保证实验结果的可复现性
"""

import random
import numpy as np
import logging
import os

def set_random_seed(seed=None):
    """
    设置所有相关库的随机种子
    
    Args:
        seed: 随机种子值，如果为None则不设置
    
    Returns:
        实际使用的种子值
    """
    if seed is None:
        logging.info("🎲 未设置随机种子，使用随机值")
        return None
    
    # 转换为整数
    seed = int(seed)
    
    logging.info(f"🔢 设置随机种子: {seed}")
    
    # Python内置随机数
    random.seed(seed)
    
    # NumPy随机数
    np.random.seed(seed)
    
    # PyTorch随机数（如果已安装）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保PyTorch的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logging.info("   ✓ PyTorch种子已设置")
    except ImportError:
        pass
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"   ✓ 所有随机种子已设置为: {seed}")
    
    return seed


def get_seed_from_config(config):
    """
    从配置中读取随机种子设置
    
    Args:
        config: Config对象
    
    Returns:
        随机种子值，如果未启用则返回None
    """
    # 检查是否启用随机种子
    seed_config = config.config.get('random_seed', {})
    
    if not seed_config:
        return None
    
    enabled = seed_config.get('enabled', False)
    seed = seed_config.get('seed', None)
    
    if not enabled:
        logging.info("🎲 随机种子功能未启用")
        return None
    
    if seed is None:
        logging.info("🎲 随机种子配置为None，使用随机值")
        return None
    
    return seed
