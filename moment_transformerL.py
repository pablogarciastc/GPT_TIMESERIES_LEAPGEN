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

        # === Backbone ===
        self.backbone = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small")
        self.input_proj = nn.Linear(num_features, self.embed_dim)

        # === E-Prompt ===
        self.use_e_prompt = use_e_prompt
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
                num_heads=num_heads,
                same_key_value=same_key_value,
                prompts_per_task=prompts_per_task,
                text_embed_dim=128,  # SentenceTransformer/roberta embeddings
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
            x = x.view(x.size(0), self.backbone.seq_len, self.num_features)
        elif x.shape[1] == self.num_features and x.shape[2] == self.backbone.seq_len:
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
                    y=y,
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
            x = x.view(x.size(0), self.backbone.seq_len, self.num_features)
        elif x.shape[1] == self.num_features and x.shape[2] == self.backbone.seq_len:
            x = x.permute(0, 2, 1)

        x = x.float()

        # Get device from model parameters
        device = next(self.parameters()).device
        x = x.to(device)

        x = self.input_proj(x)  # [B, seq_len, embed_dim]

        # EN LUGAR DE LLAMAR normalizer y patch_embedding directamente,
        # usa el método encode del backbone que maneja los dispositivos correctamente
        # O simplemente salta esas capas y empieza desde el encoder

        # Opción 1: Usar el método público que maneja dispositivos
        # x_encoded = self.backbone.model.embedder(x)  # Si existe

        # Opción 2: Saltar normalizer y patch_embedding (ya proyectaste con input_proj)
        # y empezar directamente en el encoder

        # Initialize results dictionary
        res = dict()
        reduce_sim, reduce_sim2 = None, None
        x_embed_norm = None

        # Simplified prompt mask
        prompt_mask = None

        # Access MOMENT's encoder blocks directly
        encoder_blocks = self.backbone.encoder.block

        if self.use_e_prompt:
            last_e_prompt = None
            batched_prompt_list = []
            e_prompt_counter = -1

            # Iterate over each transformer block
            for i, block in enumerate(encoder_blocks):

                if i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1

                    # Generate prompt for this specific layer
                    e_out = self.e_prompt(
                        x_embed=x,
                        y=y,
                        task_id=task_id,
                        prompt_mask=prompt_mask,
                        layer_num=e_prompt_counter,
                        cls_features=cls_features if cls_features is not None else x.mean(dim=1)
                    )

                    # Store intermediate results from first layer
                    if e_prompt_counter == 0:
                        if "prompt_key_norm" in e_out:
                            res["prompt_key_norm"] = e_out["prompt_key_norm"]
                        if "similarity" in e_out:
                            res["similarity"] = e_out["similarity"]
                        if "x_embed_norm" in e_out:
                            x_embed_norm = e_out["x_embed_norm"]
                        if "max_t" in e_out:
                            res["max_t"] = e_out["max_t"]

                    if "batched_prompt" in e_out:
                        e_seq = e_out["batched_prompt"]

                        # Store the batched prompt
                        batched_prompt_list.append(e_seq)

                        if e_seq.dim() == 5:
                            B, dual, length, H, D = e_seq.shape
                            e_seq = e_seq.reshape(B, dual * length, H * D)

                        # Add description embedding if available
                        if "desc_embed" in e_out:
                            desc_embed = e_out["desc_embed"].unsqueeze(1)
                            x = torch.cat((x, desc_embed), dim=1)
                            res["desc_embed"] = e_out["desc_embed"]

                        # Handle prefix tuning with prompt accumulation
                        if self.use_prefix_tune_for_e_prompt:
                            if e_prompt_counter > 0 and last_e_prompt is not None:
                                ne_prompt = e_seq + last_e_prompt
                            else:
                                ne_prompt = e_seq

                            # Store for next layer
                            last_e_prompt = e_seq

                            # Concatenate accumulated prompt
                            x = torch.cat([ne_prompt, x], dim=1)
                            x = block(x)
                        else:
                            # For prompt tuning, concatenate prompts to sequence
                            x = torch.cat([e_seq, x], dim=1)
                            x = block(x)

                    # Accumulate similarity losses
                    if reduce_sim is None:
                        reduce_sim = e_out.get("reduce_sim", None)
                        reduce_sim2 = e_out.get("reduce_sim2", None)

                else:
                    # Block without prompts
                    x = block(x)

            # Store last batched prompt if we have any
            if batched_prompt_list:
                res["batched_prompt"] = batched_prompt_list[-1]

            # Apply final layer norm from encoder
            x = self.backbone.encoder.final_layer_norm(x)

        else:
            # No prompts - just iterate through blocks normally
            for block in encoder_blocks:
                x = block(x)
            x = self.backbone.encoder.final_layer_norm(x)

        # Apply head if exists
        if hasattr(self.backbone, 'head') and self.backbone.head is not None:
            x = self.backbone.head(x)

        # Keep sequence representation like VisionTransformer
        features_seq = x  # [B, seq, embed_dim]

        # Update results with all outputs
        res.update({
            "x": features_seq,
            "reduce_sim": reduce_sim,
            "reduce_sim2": reduce_sim2
        })

        # Add x_embed_norm if we have it
        if x_embed_norm is not None:
            res["x_embed_norm"] = x_embed_norm

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