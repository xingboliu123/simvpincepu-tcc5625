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
    
  
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()

   
    if args.ckpt_path:
        print(f"\n⚡ [Manual Fix] 正在尝试手动加载权重: {args.ckpt_path}")
        try:
           
            checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
            
           
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
           
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
               
                if name.startswith('module.'): 
                    name = name[7:] 
                
                if name.startswith('model.'): 
                    name = name[6:] 
                new_state_dict[name] = v
            

            exp.method.model.load_state_dict(new_state_dict, strict=True) 

            
        except RuntimeError as e:
            print(f"   {e}\n")
            exit(1) 
            
        except Exception as e:
            print(f"❌ [Error] 加载过程中发生未知错误: {e}")
    
    exp.test()