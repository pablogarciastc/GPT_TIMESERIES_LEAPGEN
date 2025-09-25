""" Fixed MOMENT Transformer with LEAPGen Prompt Learning
Addresses the classification head issue and proper MOMENT integration
"""

import os

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from momentfm import MOMENTPipeline

from timm.models.layers import Mlp, DropPath, trunc_normal_
from promptL import LPrompt
from attention import PreT_Attention, Block
from linears import SimpleContinualLinear

_logger = logging.getLogger(__name__)


class MOMENTTransformerL(nn.Module):
    """MOMENT-based Transformer with LEAPGen Prompt Learning"""

    def __init__(
            self, input_size=512, n_channels=1, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
            qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            moment_model_name='AutonLab/MOMENT-1-small',
            freeze_moment_encoder=True,  # Usually freeze the encoder
            freeze_moment_embedder=True,
            freeze_moment_head=True,  # We use our own head
            # Prompt parameters
            prompt_length=None, embedding_key='cls', prompt_init='uniform',
            prompt_pool=False, prompt_key=False, pool_size=None,
            num_tasks=5, top_k=None, top_k_l=None,
            batchwise_prompt=False, prompt_key_init='uniform',
            head_type='token', use_prompt_mask=False,
            use_g_prompt=False, g_prompt_length=None, g_prompt_layer_idx=None,
            use_prefix_tune_for_g_prompt=False,
            use_e_prompt=False, e_prompt_layer_idx=None,
            use_prefix_tune_for_e_prompt=False, same_key_value=False,
            prompts_per_task=5, args=None):

        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.input_size = input_size
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_features = self.embed_dim = embed_dim
        self.use_multihead = True
        self.head_type = head_type

        # Initialize MOMENT for embedding extraction only
        self.moment_model = MOMENTPipeline.from_pretrained(
            moment_model_name,
            model_kwargs={
                'task_name': 'embedding',  # Use embedding mode
                'n_channels': n_channels,
                'freeze_encoder': freeze_moment_encoder,
                'freeze_embedder': freeze_moment_embedder,
            }
        )
        self.moment_model.init()

        # Get MOMENT's output dimension
        if 'small' in moment_model_name:
            self.moment_dim = 512
        elif 'base' in moment_model_name:
            self.moment_dim = 768
        else:
            self.moment_dim = 1024

        # Projection layer if dimensions don't match
        if self.moment_dim != embed_dim:
            self.proj_moment = nn.Linear(self.moment_dim, embed_dim)
        else:
            self.proj_moment = nn.Identity()

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, input_size + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Prompt configuration
        self.prompt_pool = prompt_pool
        self.use_prompt_mask = use_prompt_mask

        self.use_g_prompt = use_g_prompt
        self.g_prompt_layer_idx = g_prompt_layer_idx if g_prompt_layer_idx else []
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt

        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx if e_prompt_layer_idx else []
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt

        # Initialize prompts
        if use_e_prompt and e_prompt_layer_idx:
            self.e_prompt = LPrompt(
                length=prompt_length, embed_dim=embed_dim,
                num_tasks=self.num_tasks, num_classes=num_classes,
                embedding_key=embedding_key, prompt_init=prompt_init,
                prompt_pool=prompt_pool, prompt_key=prompt_key,
                pool_size=pool_size, top_k=top_k, top_k_l=top_k_l,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                num_layers=len(e_prompt_layer_idx),
                use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                num_heads=num_heads, same_key_value=same_key_value,
                prompts_per_task=prompts_per_task
            )

        # Transformer blocks after MOMENT
        if use_prefix_tune_for_e_prompt or use_prefix_tune_for_g_prompt:
            attn_layer = PreT_Attention
        else:
            from attention import Attention
            attn_layer = Attention

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, init_values=init_values,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer, attn_layer=attn_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

        # Classification head
        if self.use_multihead:
            self.head = SimpleContinualLinear(embed_dim, int(num_classes / num_tasks))
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        # Calculate total prompt length
        self.total_prompt_len = 0
        if self.prompt_pool:
            if use_e_prompt and not use_prefix_tune_for_e_prompt:
                self.total_prompt_len = prompt_length * top_k * len(e_prompt_layer_idx)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def get_moment_embeddings(self, x):
        """Extract embeddings from MOMENT model
        Args:
            x: [batch_size, n_channels, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, moment_dim]
        """
        # MOMENT expects [batch_size, seq_len, n_channels] for some modes
        # But for embedding mode it should work with [batch_size, n_channels, seq_len]

        with torch.no_grad():
            # Use MOMENT's embed method directly
            embeddings = self.moment_model.model.embed(x)

        return embeddings

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        """Forward features through MOMENT and transformer blocks"""
        batch_size = x.shape[0]

        # Get MOMENT embeddings
        x = self.get_moment_embeddings(x)  # [B, seq_len, moment_dim]

        # Project to our embedding dimension
        x = self.proj_moment(x)  # [B, seq_len, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate position embeddings if needed
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        x = self.pos_drop(x + pos_embed)

        # Apply transformer blocks with prompts
        if self.use_e_prompt:
            e_prompt_counter = -1
            last_e_prompt = None

            for i, block in enumerate(self.blocks):
                if i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1

                    # Get prompts
                    if self.use_prompt_mask and train:
                        start = task_id * self.e_prompt.top_k
                        end = (task_id + 1) * self.e_prompt.top_k
                        single_prompt_mask = torch.arange(start, end).to(x.device)
                        prompt_mask = single_prompt_mask.unsqueeze(0).expand(batch_size, -1)
                    else:
                        prompt_mask = None

                    res = self.e_prompt(
                        x, task_id=task_id, prompt_mask=prompt_mask,
                        layer_num=e_prompt_counter, cls_features=x[:, 0]
                    )
                    e_prompt = res['batched_prompt']

                    if self.use_prefix_tune_for_e_prompt:
                        if e_prompt_counter > 0 and last_e_prompt is not None:
                            ne_prompt = e_prompt + last_e_prompt
                        else:
                            ne_prompt = e_prompt
                        x = block(x, prompt=ne_prompt)
                    else:
                        if e_prompt_counter < len(e_prompt):
                            prompt = e_prompt[e_prompt_counter]
                            x = torch.cat([prompt, x], dim=1)
                        x = block(x)

                    last_e_prompt = e_prompt
                else:
                    x = block(x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return {'x': x}

    def forward_head(self, res, device):
        """Forward through classification head"""
        x = res['x']

        # Extract features based on head type
        if self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        res['pre_logits'] = x
        x = self.fc_norm(x)

        if self.use_multihead:
            out = self.head(x)
            res['logits'] = out['logits']
        else:
            res['logits'] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        """Standard forward pass"""
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res, device=x.device)
        return res

    def forwardA1(self, x, y, task_id=-1, cls_features=None, train=False):
        """Forward with auxiliary information (simplified for now)"""
        # For now, just use regular forward
        # You can extend this to use descriptors as in the original
        return self.forward(x, task_id=task_id, cls_features=cls_features, train=train)