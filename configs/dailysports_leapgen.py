import argparse

def get_args_parser(parser):

    # -------------------
    # Training parameters
    # -------------------
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--pin-mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # -------------------
    # Model parameters
    # -------------------
    parser.add_argument(
        '--model',
        default='moment_base',
        type=str,
        help='Backbone: vit_base_patch16_224 or moment_base'
    )
    parser.add_argument('--seq-length', default=125, type=int, help='Input sequence length for classification')
    parser.add_argument('--num-features', default=45, type=int, help='Number of input features')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num-heads', default=1, type=int, help='Number of attention heads (must divide hidden-size)')
    parser.add_argument('--pretrained', default=True, help='Load pretrained MOMENT model or not')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Clip gradient norm (default: 1.0, 0 to disable)')
    parser.add_argument('--use-clip-grad', type=float, default=True,
                        help='Clip gradient norm (default: 1.0, 0 to disable)')

    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-6)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-epochs', type=float, default=30,
                        help='epoch interval to decay LR')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='LR decay rate')

    # -------------------
    # Optimizer parameters
    # -------------------
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer type')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate (smaller for MOMENT fine-tuning)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--sched', default='cosine', choices=['none', 'cosine', 'step'], type=str, help='LR scheduler type')
    parser.add_argument('--step-size', type=int, default=10, help='Step size for step LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR decay factor for step LR scheduler')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='Scaling lr by batch size (default: True)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='Reinit optimizer (default: True)')

    # -------------------
    # Data parameters
    # -------------------
    parser.add_argument('--data-path', default='./local_datasets/DailySports', type=str, help='Path to dataset folder')
    parser.add_argument('--dataset', default='Split-DailySports', type=str, help='Dataset name')
    parser.add_argument('--target', default='slice_type', type=str, help='Target variable for classification')
    parser.add_argument('--shuffle', default=False, type=bool, help='Shuffle the dataset')
    parser.add_argument('--train-ratio', default=0.8, type=float, help='Training data ratio')

    # -------------------
    # Evaluation / logging
    # -------------------
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training progress')
    parser.add_argument('--output-dir', default='./outputs', type=str, help='Directory to save models and logs')
    parser.add_argument('--device', default='cuda', type=str, help='Device to train on')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of data loader workers')
    parser.add_argument('--cudnn-benchmark', default=False, type=bool, help='Enable cudnn benchmark for faster training')

    # -------------------
    # Continual learning parameters
    # -------------------
    parser.add_argument('--num_tasks', default=6, type=int, help='Number of sequential tasks')
    parser.add_argument('--train_mask', default=True, type=bool, help='If using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='If doing task incremental')

    # -------------------
    # G-Prompt parameters
    # -------------------
    parser.add_argument('--use_g_prompt', default=False, type=bool, help='If using G-Prompt')
    parser.add_argument('--g_prompt_length', default=4, type=int, help='Length of G-Prompt')
    parser.add_argument('--g_prompt_layer_idx', default=[0,2,4], type=int, nargs="+", help='The layer index of the G-Prompt')
    parser.add_argument('--use_prefix_tune_for_g_prompt', default=True, type=bool,
                        help='If using prefix tuning for G-Prompt')

    # -------------------
    # E-Prompt parameters
    # -------------------
    parser.add_argument('--use_e_prompt', default=True, type=bool, help='If using the E-Prompt')
    parser.add_argument('--e_prompt_layer_idx', default=[0, 2, 4, 6], type=int, nargs="+",
                        help='The layer index of the E-Prompt')
    parser.add_argument('--use_prefix_tune_for_e_prompt', default=True, type=bool,
                        help='If using prefix tuning for E-Prompt')

    # -------------------
    # Conv Prompt parameters (unused, but kept for consistency)
    # -------------------
    # parser.add_argument('--kernel_size', default=17, type=int, help='kernel size of the conv prompt')

    # -------------------
    # E-Prompt pool parameters
    # -------------------
    parser.add_argument('--prompt_pool', default=True, type=bool)
    parser.add_argument('--size', default=4500, type=int)
    parser.add_argument('--length', default=20, type=int)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--top_k_l', default=3, type=int)
    parser.add_argument('--initializer', default='uniform', type=str)
    parser.add_argument('--prompt_key', default=True, type=bool)
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=True, type=bool)
    parser.add_argument('--mask_first_epoch', default=False, type=bool)
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=False, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.01, type=float)
    parser.add_argument('--pull_constraint_coeff2', default=0.01, type=float)
    parser.add_argument('--intertask_coeff', default=0.01, type=float)
    parser.add_argument('--k_mul', default=25.0, type=float)
    parser.add_argument('--same_key_value', default=False, type=bool)
    parser.add_argument('--dualopt', default=True, type=bool)
    parser.add_argument('--eval_frequency', default=5, type=int)
    parser.add_argument('--save_best_model', default=True, type=bool)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--replay_loss_weight', default=0.03, type=float)
    parser.add_argument('--prompt_l2_weight', default=0.1, type=float)
    parser.add_argument('--label_smoothing', default=0.05, type=float)
    # -------------------
    # Attribute generator / prompt parameters
    # -------------------
    parser.add_argument('--num_prompts_per_task', default=3, type=int,
                        help='Max number of prompts to be generated for each task')
    parser.add_argument('--variable_num_prompts', default=True, type=bool,
                        help='If variable number of prompts per task are to be used')

    # -------------------
    # MOMENT parameters
    # -------------------
    parser.add_argument(
        '--freeze',
        default=['backbone.encoder', 'backbone.embeddings'],
        nargs='*',
        type=list,
        help='Freeze part of MOMENT backbone'
    )
