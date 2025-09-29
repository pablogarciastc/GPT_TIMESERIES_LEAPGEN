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

_logger = logging.getLogger(__name__)


class MomentTransformerL(nn.Module):
    def __init__(
            self,
            num_classes=18,
            prompt_length=5,
            embedding_key='mean',
            prompt_init='uniform',
            prompt_pool=True,
            prompt_key=True,
            pool_size=None,
            num_tasks=6,
            top_k=1,
            top_k_l=3,
            batchwise_prompt=False,
            prompt_key_init='uniform',
            e_prompt_layer_idx=[0],
            use_prefix_tune_for_e_prompt=True,
            same_key_value=False,
            prompts_per_task=5,
            head_type='token',
            use_prompt_mask=False,
            args=None,
            **kwargs
    ):
        super().__init__()

        self.args = args
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.embed_dim = 128
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.prompt_pool = prompt_pool

        # === Backbone ===
        self.backbone = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small")
        self.input_proj = nn.Linear(self.args.num_features, self.embed_dim)

        # === E-Prompt ===
        self.use_e_prompt = getattr(args, "use_e_prompt", True)
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt

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
                num_heads=args.num_heads,
                same_key_value=same_key_value,
                prompts_per_task=prompts_per_task,
                text_embed_dim=768,  # SentenceTransformer/roberta embeddings
            )

        # === G-Prompt (optional) ===
        self.use_g_prompt = getattr(args, "use_g_prompt", False)
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
                num_heads=args.num_heads,
                same_key_value=same_key_value,
                prompts_per_task=prompts_per_task,
                text_embed_dim=768,
            )

        # Calculate total prompt length for head processing
        self.total_prompt_len = 0
        if self.prompt_pool:
            if self.use_e_prompt and not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)

        # === Classifier ===
        self.head = SimpleContinualLinear(self.embed_dim, num_classes)

        print(f"[MomentTransformerL] Initialized with {num_classes} classes, "
              f"{num_tasks} tasks, embed_dim={self.embed_dim}")

    def forward_features(self, x, task_id=-1, train=True, y=None, prompt_mask=None, cls_features=None):
        # Normalize shape [B, seq, features]
        if x.dim() == 2:
            x = x.view(x.size(0), self.args.seq_length, self.args.num_features)
        elif x.shape[1] == self.args.num_features and x.shape[2] == self.args.seq_length:
            x = x.permute(0, 2, 1)

        x = x.float()
        x = self.input_proj(x)  # [B, seq_len, embed_dim]

        # Initialize results dictionary
        res = dict()
        reduce_sim, reduce_sim2 = None, None

        # Handle prompt mask similar to VisionTransformerL
        if self.use_prompt_mask and train and self.use_e_prompt:
            start = task_id * self.e_prompt.top_k
            end = (task_id + 1) * self.e_prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(x.device)
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
            if end > self.e_prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None

        # === G-Prompt Processing ===
        if self.use_g_prompt:
            g_out = self.g_prompt(
                x_embed=x,
                task_id=task_id,
                prompt_mask=prompt_mask,
                layer_num=0,
                cls_features=x.mean(dim=1),  # Extract features from x instead of using parameter
            )
            if "batched_prompt" in g_out:
                g_seq = g_out["batched_prompt"]
                if g_seq.dim() == 5:
                    B, dual, length, H, D = g_seq.shape
                    g_seq = g_seq.reshape(B, dual * length, H * D)
                x = torch.cat([g_seq, x], dim=1)

        # === E-Prompt Processing (layer-by-layer like VisionTransformerL) ===
        if self.use_e_prompt:
            # Process each layer that has e-prompts
            for layer_idx, layer_num in enumerate(self.e_prompt_layer_idx):
                e_out = self.e_prompt(
                    x_embed=x,
                    task_id=task_id,
                    prompt_mask=prompt_mask,
                    layer_num=layer_idx,
                    cls_features=x.mean(dim=1),  # Extract features from x instead of using parameter
                )

                if "batched_prompt" in e_out:
                    e_seq = e_out["batched_prompt"]
                    if e_seq.dim() == 5:
                        B, dual, length, H, D = e_seq.shape
                        e_seq = e_seq.reshape(B, dual * length, H * D)

                    # Handle prefix tuning vs prompt tuning
                    if self.use_prefix_tune_for_e_prompt:
                        x = torch.cat([e_seq, x], dim=1)
                        # FIXME UPDATE THIS PART OF THE CODE
                    else:
                        # For prompt tuning, concatenate prompts to sequence
                        x = torch.cat([e_seq, x], dim=1)

                # Accumulate similarity losses
                if reduce_sim is None:
                    reduce_sim = e_out.get("reduce_sim", None)
                    reduce_sim2 = e_out.get("reduce_sim2", None)

        # === Backbone Forward Pass ===
        outputs = self.backbone.forward(task_name="classification", x_enc=x)
        if hasattr(outputs, "reconstruction") and outputs.reconstruction is not None:
            features = outputs.reconstruction
            if features.dim() == 3:
                features = features.mean(dim=1)
        elif hasattr(outputs, "pre_logits"):
            features = outputs.pre_logits
        else:
            feats_fallback = outputs.logits
            if feats_fallback.dim() == 3:
                feats_fallback = feats_fallback.mean(dim=1)
            features = feats_fallback

        res.update({
            "x": features,
            "reduce_sim": reduce_sim,
            "reduce_sim2": reduce_sim2
        })

        return res

    def forward_featuresA1(self, x, task_id=-1, train=True, y=None, prompt_mask=None, cls_features=None):
        # Normalize shape [B, seq, features]
        if x.dim() == 2:
            x = x.view(x.size(0), self.args.seq_length, self.args.num_features)
        elif x.shape[1] == self.args.num_features and x.shape[2] == self.args.seq_length:
            x = x.permute(0, 2, 1)

        x = x.float()
        x = self.input_proj(x)  # [B, seq_len, embed_dim]

        # Initialize results dictionary
        res = dict()
        reduce_sim, reduce_sim2 = None, None

        # Simplified prompt mask (similar to Vision Transformer A1 version)
        prompt_mask = None

        # === E-Prompt Processing (layer-by-layer similar to VisionTransformerA1) ===
        if self.use_e_prompt:
            last_e_prompt = None

            # Process each layer that has e-prompts
            for layer_idx, layer_num in enumerate(self.e_prompt_layer_idx):
                e_out = self.e_prompt(
                    x_embed=x,
                    y=y,
                    task_id=task_id,
                    prompt_mask=prompt_mask,
                    layer_num=layer_idx,
                    cls_features=cls_features,
                )

                if "batched_prompt" in e_out:
                    e_seq = e_out["batched_prompt"]
                    if e_seq.dim() == 5:
                        B, dual, length, H, D = e_seq.shape
                        e_seq = e_seq.reshape(B, dual * length, H * D)

                    # Add description embedding if available (like Vision Transformer A1)
                    if "desc_embed" in e_out:
                        desc_embed = e_out["desc_embed"].unsqueeze(1)
                        x = torch.cat((x, desc_embed), dim=1)

                    # Handle prefix tuning with prompt accumulation (like Vision Transformer A1)
                    if self.use_prefix_tune_for_e_prompt:
                        if layer_idx > 0 and last_e_prompt is not None:
                            ne_prompt = e_seq + last_e_prompt
                        else:
                            ne_prompt = e_seq

                        # Store for next layer
                        last_e_prompt = e_seq

                        # Concatenate accumulated prompt
                        x = torch.cat([ne_prompt, x], dim=1)
                    else:
                        # For prompt tuning, concatenate prompts to sequence
                        x = torch.cat([e_seq, x], dim=1)

                # Accumulate similarity losses
                if reduce_sim is None:
                    reduce_sim = e_out.get("reduce_sim", None)
                    reduce_sim2 = e_out.get("reduce_sim2", None)

        # === Backbone Forward Pass ===
        outputs = self.backbone.forward(task_name="classification", x_enc=x)
        if hasattr(outputs, "reconstruction") and outputs.reconstruction is not None:
            features = outputs.reconstruction
            if features.dim() == 3:
                features = features.mean(dim=1)
        elif hasattr(outputs, "pre_logits"):
            features = outputs.pre_logits
        else:
            feats_fallback = outputs.logits
            if feats_fallback.dim() == 3:
                feats_fallback = feats_fallback.mean(dim=1)
            features = feats_fallback

        res.update({
            "x": features,
            "reduce_sim": reduce_sim,
            "reduce_sim2": reduce_sim2
        })

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
        logits = self.head(x)

        res.update({
            "logits": logits,
            "reduce_sim": res.get("reduce_sim"),
            "reduce_sim2": res.get("reduce_sim2")
        })

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

        res.update({
            "logits": out['logits'],
            "reduce_sim": res.get("reduce_sim"),
            "reduce_sim2": res.get("reduce_sim2")
        })

        return res

    def forward(self, x, task_id=-1, train=False, cls_features=None):
        res = self.forward_features(x, task_id=task_id, train=train, cls_features=cls_features)
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