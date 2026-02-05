# Copyright (c) CAIRI AI Lab. All rights reserved

import warnings
warnings.filterwarnings('ignore')
import torch 

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'val_batch_size'])
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    print('>'*35 + ' testing  ' + '<'*35)
    
    # 1. 初始化实验
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

    # ================= 核心修复代码 (Final Fix) =================
    if args.ckpt_path:
        print(f"⚡ [Manual Fix] 正在尝试手动加载权重: {args.ckpt_path}")
        try:
            # 1. 加载文件 (weights_only=False 解决安全报错)
            checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
            
            # 2. 提取 state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 3. 清洗前缀 (这是最关键的一步！)
            # 我们的目标是加载到 exp.method.model (纯 PyTorch 模型)
            # 所以需要把 'module.' (DDP前缀) 和 'model.' (Lightning前缀) 都去掉
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:] # 去掉 module.
                if name.startswith('model.'):
                    name = name[6:] # 去掉 model.
                new_state_dict[name] = v
            
            # 4. 强制加载到内层模型
            # 【注意】这里改成了 exp.method.model
            exp.method.model.load_state_dict(new_state_dict, strict=True) 
            print("✅ [Success] 权重加载成功！(已清洗 module/model 前缀并严格匹配)")
            
        except Exception as e:
            print(f"❌ [Error] 手动加载失败: {e}")
            print("⚠️ 将尝试使用 BaseExperiment 默认加载方式...")
    # ================= 核心修复代码 END =================

    exp.test()