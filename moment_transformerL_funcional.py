# ------------------------------------------
# MOMENT Foundation Model backbone
# With continual learning & prompt support
# Refactored to be more similar to VisionTransformerL structure
# ------------------------------------------

import logging
import torch
import torch.nn as nn

from linears import SimpleContinualLinear
from promptL import LPrompt
from momentfm import MOMENTPipeline
from timm.models.helpers import checkpoint_seq

_logger = logging.getLogger(__name__)


class MomentTransformerL(nn.Module):
    def __init__(
            self,
        pretrained_cfg = None,
        num_classes = 1000,
        drop_rate = 0.,
        drop_path_rate = 0.,
        prompt_length = None,
        embedding_key = 'cls',
        prompt_init = 'uniform',
        prompt_pool = False,
        prompt_key=False, pool_size=None,
        num_tasks=6,
        top_k=None, top_k_l=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token',
        use_prompt_mask=False,
        use_g_prompt=False,
        num_heads = 12,
        g_prompt_length=None,
        g_prompt_layer_idx=None,
        use_prefix_tune_for_g_prompt=False,
        use_e_prompt=False, e_prompt_layer_idx=None, use_prefix_tune_for_e_prompt=False, same_key_value=False,
        prompts_per_task=5,num_features=45,**kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.embed_dim = 768
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.prompt_pool = prompt_pool
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.grad_checkpointing = False
        self.use_multihead=True
        self.num_features = num_features
        self.num_heads = num_heads
        self.embed_proj = nn.Linear(768, 512)
        self.prompt_proj = nn.Linear(64, 512)

        # === Backbone ===
        self.backbone = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small")
        self._disable_gc()
        self.input_proj = nn.Linear(num_features, self.embed_dim)
        self.back_proj = nn.Linear(512, 768)
        # === E-Prompt ===
        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.final_norm = nn.LayerNorm(768)
        if self.use_e_prompt:
            self.e_prompt = LPrompt(
                length=prompt_length,
                embed_dim=self.embed_dim,
                num_tasks=num_tasks,
                num_classes=num_classes,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                top_k_l=top_k_l,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                num_layers=num_e_prompt,
                use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                num_heads=num_heads,
                same_key_value=same_key_value,
                prompts_per_task=prompts_per_task,
            )

        # === G-Prompt (optional) ===
        self.use_g_prompt = use_g_prompt
        if self.use_g_prompt:
            self.g_prompt = LPrompt(
                length=prompt_length,
                embed_dim=self.embed_dim,
                num_tasks=num_tasks,
                num_classes=num_classes,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                prompt_key=prompt_key,
                pool_size=pool_size,
                top_k=top_k,
                top_k_l=top_k_l,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                num_layers=1,
                use_prefix_tune_for_e_prompt=False,
                num_heads=num_heads,
                same_key_value=same_key_value,
                prompts_per_task=prompts_per_task,
                text_embed_dim=128,
            )
        self.g_prompt_layer_idx = g_prompt_layer_idx
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt


        # Calculate total prompt length for head processing
        self.total_prompt_len = 0
        if self.prompt_pool:
            if self.use_e_prompt and not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)

        # === Classifier ===
        self.head = SimpleContinualLinear(self.embed_dim, num_classes)

        print(f"[MomentTransformerL] Initialized with {num_classes} classes, "
              f"{num_tasks} tasks, embed_dim={self.embed_dim}")

    def _disable_gc(self):
        #dont do two forwards
        objs = []
        if hasattr(self, "backbone"):
            objs += [self.backbone]
            if hasattr(self.backbone, "model"):
                objs += [self.backbone.model]
                if hasattr(self.backbone.model, "encoder"):
                    objs += [self.backbone.model.encoder]
                if hasattr(self.backbone.model, "decoder"):
                    objs += [self.backbone.model.decoder]
            if hasattr(self.backbone, "encoder"):
                objs += [self.backbone.encoder]
        for o in objs:
            if hasattr(o, "gradient_checkpointing_disable"):
                try:
                    o.gradient_checkpointing_disable()
                except:
                    pass
            if hasattr(o, "gradient_checkpointing"):
                try:
                    o.gradient_checkpointing = False
                except:
                    pass

    def forward_features(self, x, task_id=-1, train=False):
        if x.dim() == 2:
            x = x.view(x.size(0), self.backbone.seq_len, self.num_features)
        elif x.shape[1] == self.num_features and x.shape[2] == self.backbone.seq_len:
            x = x.permute(0, 2, 1)

        x = x.float()
        x = self.input_proj(x)

        res = dict()

        is_projected = False

        self.use_e_prompt = False

        print("grad_checkpointing: ", self.grad_checkpointing)
        print("use_g_prompt: ", self.use_g_prompt)
        print("use_e_prompt: ", self.use_e_prompt)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.backbone.encoder.block, x)
        else:
            if self.use_e_prompt or self.use_g_prompt:
                if self.use_prompt_mask and train:
                    start = task_id * self.e_prompt.top_k
                    end = (task_id + 1) * self.e_prompt.top_k
                    single_prompt_mask = torch.arange(start, end).to(x.device)
                    prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                    if end > self.e_prompt.pool_size:
                        prompt_mask = None
                else:
                    prompt_mask = None

                g_prompt_counter = -1
                e_prompt_counter = -1
                last_e_prompt = None

                for i, block in enumerate(self.backbone.encoder.block):
                    if i in self.g_prompt_layer_idx:
                        if self.use_prefix_tune_for_g_prompt:
                            e_prompt_counter += 1
                            idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                            g_prompt = self.g_prompt[idx]
                        else:
                            g_prompt= None
                        x = block(x, prompt=g_prompt)[0]

                        e_prompt = res['batched_prompt']
                        desc_embed = res['desc_embed'].unsqueeze(1)  # [B, 1, 768]

                    elif i in self.e_prompt_layer_idx:
                        e_prompt_counter += 1
                        res = self.e_prompt(x, task_id=task_id, prompt_mask=prompt_mask, layer_num=e_prompt_counter,
                                            cls_features=x[:, 0])
                        e_prompt = res['batched_prompt']

                        if self.use_prefix_tune_for_e_prompt:
                            if e_prompt_counter > 0:
                                ne_prompt = e_prompt + last_e_prompt
                            else:
                                ne_prompt = e_prompt
                            x = block(x, prompt=ne_prompt)[0]

                        else:
                            # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                            prompt = e_prompt[e_prompt_counter]
                            x = torch.cat([prompt, x], dim=1)
                            x = block(x)[0]
                        last_e_prompt = e_prompt
                    else:
                        x = block(x)[0]

                else:
                    x = self.backbone.forward(task_name="classification", x_enc=x)
                    res = dict()

            x = self.final_norm(x)
            res['x'] = x

        return res

    def forward_featuresA1(self, x, task_id=-1, train=True, y=None, prompt_mask=None, cls_features=None):
        if x.dim() == 2:
            x = x.view(x.size(0), self.backbone.seq_len, self.num_features)
        elif x.shape[1] == self.num_features and x.shape[2] == self.backbone.seq_len:
            x = x.permute(0, 2, 1)

        x = x.float()
        x = self.input_proj(x)

        res = dict()

        is_projected = False

        if self.use_e_prompt or self.use_g_prompt:
            prompt_mask = None
            g_prompt_counter = -1
            e_prompt_counter = -1
            last_e_prompt = None

            for i, block in enumerate(self.backbone.encoder.block):
                if i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1

                    res = self.e_prompt(x, y, task_id=task_id, prompt_mask=prompt_mask, layer_num=e_prompt_counter,
                                        cls_features=cls_features)

                    e_prompt = res['batched_prompt']
                    desc_embed = res['desc_embed'].unsqueeze(1)  # [B, 1, 768]

                    if self.use_prefix_tune_for_e_prompt:
                        if not is_projected:
                            x = self.embed_proj(x)  # [B, seq_len, 512]
                            is_projected = True

                        desc_embed_proj = self.embed_proj(desc_embed)  # [B, 1, 512]

                        x = torch.cat((x, desc_embed_proj), dim=1)  # [B, seq_len+1, 512]

                        if e_prompt_counter > 0 and last_e_prompt is not None:
                            ne_prompt = e_prompt + last_e_prompt
                        else:
                            ne_prompt = e_prompt

                        ne_prompt = ne_prompt.reshape(ne_prompt.shape[0], -1, ne_prompt.shape[-1])
                        ne_prompt = self.prompt_proj(ne_prompt)  # [B, prompt_len, 512]

                        x = block(x, prompt=ne_prompt)[0]

                    else:
                        x = torch.cat((x, desc_embed), dim=1)
                        prompt = e_prompt[e_prompt_counter]
                        x = torch.cat([prompt, x], dim=1)
                        x = block(x)[0]

                    last_e_prompt = e_prompt
                else:
                    x = block(x)[0]
        else:
            x = self.backbone.forward(task_name="classification", x_enc=x)
            res = dict()
        x = self.back_proj(x)
        x = self.final_norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, device, pre_logits=False):
        x = res["x"]

        # Handle different head types (similar to VisionTransformerL logic)
        if self.head_type == 'token':
            if self.prompt_pool:
                # Skip prompt tokens if using prompt pool
                x = x[:, self.total_prompt_len:] if self.total_prompt_len > 0 else x
            # For sequence data, we might want to use mean pooling or just first token
            if x.dim() == 3:
                x = x.mean(dim=1)  # Global average pooling
        elif self.head_type == 'gap':
            if x.dim() == 3:
                x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            if self.total_prompt_len > 0:
                x = x[:, :self.total_prompt_len]
                x = x.mean(dim=1)
        else:
            if x.dim() == 3:
                x = x.mean(dim=1)

        res["pre_logits"] = x

        # Apply classification head - direct call returns logits tensor
        out = self.head(x)

        if self.use_multihead:
            res['logits'] = out['logits']
        else:
            res['logits'] = out[:, :self.num_classes]
            res['task_logits'] = out[:, self.num_classes:]
        return res

    def forward_headA1(self, res, task_id, device, pre_logits=False):
        x = res["x"]

        # Handle different head types (similar to VisionTransformerL logic)
        if self.head_type == 'token':
            if self.prompt_pool:
                # Skip prompt tokens if using prompt pool
                x = x[:, self.total_prompt_len:] if self.total_prompt_len > 0 else x
            # For sequence data, we might want to use mean pooling or just first token
            if x.dim() == 3:
                x = x.mean(dim=1)  # Global average pooling
        elif self.head_type == 'gap':
            if x.dim() == 3:
                x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            if self.total_prompt_len > 0:
                x = x[:, :self.total_prompt_len]
                x = x.mean(dim=1)
        else:
            if x.dim() == 3:
                x = x.mean(dim=1)

        res["pre_logits"] = x
        max_t = res.get('max_t', None)

        # Use forward2 which returns dictionary with 'logits' key
        out = self.head.forward2(x, max_t)

        if self.use_multihead:
            res['logits'] = out['logits']
        else:
            res['logits'] = out[:, :self.num_classes]
            res['task_logits'] = out[:, self.num_classes:]
        return res

    def forward(self, x, task_id=-1, train=False, cls_features=None):
        res = self.forward_features(x, task_id=task_id, train=train)
        res = self.forward_head(res, device=x.device)
        return res

    def forwardA1(self, x, target=None, task_id=-1, cls_features=None, train=True):
        res = self.forward_featuresA1(
            x, task_id=task_id, train=train, y=target, cls_features=cls_features
        )
        res = self.forward_headA1(res, task_id=task_id, device=x.device)
        return res

    def forward2(self, x, max_t=None):
        res = self.forward_features(x, task_id=-1, train=False)
        res = self.forward_head(res, device=x.device)
        return res