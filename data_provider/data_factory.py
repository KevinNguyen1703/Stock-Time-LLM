from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from data_provider.stock_data_loader import Dataset_Stock, Dataset_Stock_WithPrompt, stock_collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'Stock': Dataset_Stock,
    'StockPrompt': Dataset_Stock_WithPrompt,
}


def data_provider(args, flag, with_prompt=False):
    """
    Create dataset and dataloader
    
    Args:
        args: Arguments containing data configuration
        flag: 'train', 'val', or 'test'
        with_prompt: If True and using StockPrompt, use custom collate function
    
    Returns:
        data_set: Dataset object
        data_loader: DataLoader object
    """
    # Use StockPrompt if dynamic prompts requested
    if with_prompt and args.data == 'Stock':
        Data = Dataset_Stock_WithPrompt
    else:
        Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        collate_fn = None
    elif args.data in ['Stock', 'StockPrompt'] or with_prompt:
        # Stock data with optional prompt support
        prompt_data_path = getattr(args, 'prompt_data_path', None)
        if prompt_data_path is None:
            # Auto-detect prompt file based on pred_len and data version
            data_path = getattr(args, 'data_path', '').lower()
            is_v2 = '_v2' in data_path or getattr(args, 'use_v2_data', False)
            is_v0 = '_v0' in data_path
            
            if is_v2:
                if args.pred_len == 1:
                    prompt_data_path = 'prompts_v2_short_term.json'
                else:
                    prompt_data_path = 'prompts_v2_mid_term.json'
            elif is_v0:
                if args.pred_len == 1:
                    prompt_data_path = 'prompts_v0_short_term.json'
                else:
                    prompt_data_path = 'prompts_v0_mid_term.json'
            else:
                if args.pred_len == 1:
                    prompt_data_path = 'prompts_short_term.json'
                else:
                    prompt_data_path = 'prompts_mid_term.json'
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            prompt_data_path=prompt_data_path if with_prompt else None
        )
        # Use custom collate function if prompts are included
        collate_fn = stock_collate_fn if (with_prompt and isinstance(data_set, Dataset_Stock_WithPrompt)) else None
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
        collate_fn = None
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_set, data_loader
