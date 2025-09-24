# ------------------------------------------
# main_leapgen.py - modified for MOMENT backbone
# ------------------------------------------

import sys
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os
import copy   # ✅ necesario para clonar el modelo

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine_leapgen import *
import modelsL   # <-- now includes moment_base
import utils

import warnings
warnings.filterwarnings(
    'ignore',
    'Argument interpolation should be of type InterpolationMode instead of int'
)


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # --- Build dataloaders ---
    data_loader, class_mask = build_continual_dataloader(args)
    print("NB Classes:", args.nb_classes)


    # --- Build model ---
    print(f"Creating model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.dropout,
        drop_path_rate=0.0,
        args=args
    )



    model = create_model(
        args.model,                     # e.g. "moment_base"
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.dropout,
        drop_path_rate=0.0,
        # Prompt + continual args are passed through
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        num_tasks=args.num_tasks,
        top_k=args.top_k,
        top_k_l=args.top_k_l,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type="token",
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        prompts_per_task=args.num_prompts_per_task,
        args=args,
    )
    original_model.to(device)
    model.to(device)

    # --- Freeze layers if requested ---
    if args.freeze:
        for p in original_model.parameters():
            p.requires_grad = False

        for n, p in model.named_parameters():
            if any(f in n for f in args.freeze):
                p.requires_grad = False

    print(args)

    # --- Evaluation only ---
    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(
                args.output_dir,
                f'checkpoint/task{task_id+1}_checkpoint.pth'
            )
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(
                model, data_loader, device,
                task_id, class_mask, acc_matrix, args,
            )
        return

    # --- Training setup ---
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    print("Start training...")
    start_time = time.time()
    acc_matrix = train_and_evaluate(
    model, original_model, criterion, data_loader, lr_scheduler, optimizer, device,
    class_mask=class_mask, args=args
)
    total_time = time.time() - start_time
    print("Total training time:", str(datetime.timedelta(seconds=int(total_time))))
    print("ACC MATRIX: ", acc_matrix)

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    print("Started main")
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    print("Parser created: ", parser)

    # First arg after script is the config name
    print("Getting config")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'dailysports_leapgen':
        from configs.dailysports_leapgen import get_args_parser
        config_parser = subparser.add_parser('dailysports_leapgen', help='DailySports configs for MOMENT backbone')

    else:
        raise NotImplementedError(f"Config {config} not supported")

    # Inject args from the chosen config
    get_args_parser(config_parser)

    print("Reached here")
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Reached here, starting main()")
    main(args)

    sys.exit(0)
