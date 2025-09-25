# ------------------------------------------
# MOMENT Foundation Model backbone
# With continual learning & prompt support
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
        args=None,
        **kwargs
    ):
        super().__init__()

        self.args = args
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.embed_dim = 128  # 🔑 fixed working dim

        # === Backbone ===
        self.backbone = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': 45,
                'num_class': 18,
                'freeze_encoder': False,
                'freeze_embedder': False,
                'freeze_head': False,
            },
        )
        self.backbone.init()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.model.to(device)        self.input_proj = nn.Linear(self.args.num_features, self.embed_dim)

        # === E-Prompt ===
        self.use_e_prompt = getattr(args, "use_e_prompt", True)
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
                num_layers=len(e_prompt_layer_idx),
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

        # === Classifier ===
        self.head = SimpleContinualLinear(self.embed_dim, num_classes)

        print(f"[MomentTransformerL] Initialized with {num_classes} classes, "
              f"{num_tasks} tasks, embed_dim={self.embed_dim}")

    # -------------------------------------------------
    def forward_features(self, x, task_id=-1, train=True, y=None, prompt_mask=None, cls_features=None):
        # normalize shape [B, seq, features]
        if x.dim() == 2:
            x = x.view(x.size(0), self.args.seq_length, self.args.num_features)
        elif x.shape[1] == self.args.num_features and x.shape[2] == self.args.seq_length:
            x = x.permute(0, 2, 1)

        x = x.float()
        x = self.input_proj(x)  # [B, seq_len, embed_dim]

        # === G-Prompt ===
        if self.use_g_prompt:
            g_out = self.g_prompt(x_embed=x, y=y, task_id=task_id,
                                  prompt_mask=prompt_mask, layer_num=0,
                                  cls_features=cls_features)
            if "batched_prompt" in g_out:
                g_seq = g_out["batched_prompt"]
                if g_seq.dim() == 3:
                    # [B, length, embed_dim]
                    pass
                elif g_seq.dim() == 5:
                    B, dual, length, H, D = g_seq.shape
                    g_seq = g_seq.reshape(B, dual * length, H * D)
                x = torch.cat([g_seq, x], dim=1)

        # === E-Prompt ===
        if self.use_e_prompt:
            e_out = self.e_prompt(x_embed=x, y=y, task_id=task_id,
                                  prompt_mask=prompt_mask, layer_num=0,
                                  cls_features=cls_features)
            if "batched_prompt" in e_out:
                e_seq = e_out["batched_prompt"]
                if e_seq.dim() == 3:
                    pass
                elif e_seq.dim() == 5:
                    B, dual, length, H, D = e_seq.shape
                    e_seq = e_seq.reshape(B, dual * length, H * D)
                x = torch.cat([e_seq, x], dim=1)

        # === Backbone ===
        outputs = self.backbone.forward(task_name="classification", x_enc=x)

        # ✅ Extract features
        if hasattr(outputs, "reconstruction") and outputs.reconstruction is not None:
            features = outputs.reconstruction
            if features.dim() == 3:
                features = features.mean(dim=1)
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            features = outputs.logits
        else:
            raise ValueError(f"[MomentTransformerL] Unexpected MOMENT output keys: {outputs.__dict__.keys()}")

        return features

    # -------------------------------------------------
    def forward(self, x, task_id=-1, train=True):
        feats = self.forward_features(x, task_id=task_id, train=train)
        logits = self.head(feats)
        return {"logits": logits, "pre_logits": feats}

    def forwardA1(self, x, target=None, task_id=-1, cls_features=None, train=True):
        feats = self.forward_features(x, task_id=task_id, train=train, cls_features=cls_features)
        logits = self.head(feats)
        return {"logits": logits, "pre_logits": feats}

    def forward2(self, x, max_t=None):
        feats = self.forward_features(x, task_id=-1, train=False)
        logits = self.head(feats)
        return {"logits": logits, "pre_logits": feats}
