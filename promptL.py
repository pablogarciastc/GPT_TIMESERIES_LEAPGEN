import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Gen_Attention2


class LPrompt(nn.Module):
    def __init__(
            self,
            length=5,
            embed_dim=128,
            num_tasks=10,
            num_classes=100,
            embedding_key="mean",
            prompt_init="uniform",
            prompt_pool=True,
            prompt_key=True,
            pool_size=None,
            top_k=1,
            top_k_l=3,
            batchwise_prompt=False,
            prompt_key_init="uniform",
            num_layers=1,
            use_prefix_tune_for_e_prompt=False,
            num_heads=8,
            same_key_value=False,
            prompts_per_task=5,
            text_embed_dim=768,
    ):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.prompts_per_task = prompts_per_task
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.top_k_l = top_k_l
        self.batchwise_prompt = batchwise_prompt
        self.numclasses_per_task = int(self.num_classes / self.num_tasks)

        # Fixed pool size - allocate maximum needed upfront
        self.max_pool_size = self.num_tasks * self.prompts_per_task

        # === Improved prompt keys with proper initialization ===
        self.prompt_key = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.task_key = nn.Parameter(torch.empty(num_tasks, embed_dim))

        if prompt_key_init == "xavier":
            nn.init.xavier_uniform_(self.prompt_key)
            nn.init.xavier_uniform_(self.task_key)
        elif prompt_key_init == "uniform":
            nn.init.uniform_(self.prompt_key, -0.5, 0.5)
            nn.init.uniform_(self.task_key, -0.5, 0.5)
        else:  # zero
            nn.init.zeros_(self.prompt_key)
            nn.init.zeros_(self.task_key)

        # === Projections with dropout for regularization ===
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )
        self.prompt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )

        # === Improved generator allocation ===
        self.generators = nn.ModuleDict()
        for layer_idx in range(num_layers):
            self.generators[f"layer_{layer_idx}"] = nn.ModuleDict()
            for head_idx in range(num_heads):
                self.generators[f"layer_{layer_idx}"][f"head_{head_idx}"] = nn.ModuleDict({
                    "k_gens": nn.ModuleList(),
                    "v_gens": nn.ModuleList()
                })

                # Pre-allocate generators for maximum pool size
                head_dim = embed_dim // num_heads
                for _ in range(self.max_pool_size):
                    k_gen = Gen_Attention2(head_dim, 1, False, 0.1, 0.1)
                    v_gen = Gen_Attention2(head_dim, 1, False, 0.1, 0.1)
                    self.generators[f"layer_{layer_idx}"][f"head_{head_idx}"]["k_gens"].append(k_gen)
                    self.generators[f"layer_{layer_idx}"][f"head_{head_idx}"]["v_gens"].append(v_gen)

        # === Task management ===
        self.task_boundaries = []  # [(start_class, end_class, start_prompt, end_prompt)]
        self.current_task = 0
        self.desc_embeddings = []

        # === Temperature for softmax (learnable) ===
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)

        print(f"[ImprovedLPrompt] Initialized with {num_classes} classes, {num_tasks} tasks")

    def l2_normalize(self, x, dim=None, epsilon=1e-8):
        """Improved normalization with smaller epsilon"""
        return F.normalize(x, p=2, dim=dim, eps=epsilon)

    def add_task(self, task_id, class_indices, desc_embeddings):
        """Properly register a new task with its descriptors"""
        start_class = min(class_indices) if class_indices else 0
        end_class = max(class_indices) + 1 if class_indices else 0
        start_prompt = len(self.desc_embeddings)

        # Project and store descriptors
        with torch.no_grad():
            projected_desc = self.text_proj(desc_embeddings)
            self.desc_embeddings.append(projected_desc)

        end_prompt = start_prompt + len(desc_embeddings)
        self.task_boundaries.append((start_class, end_class, start_prompt, end_prompt))
        self.current_task = max(self.current_task, task_id + 1)

        print(f"[ImprovedLPrompt] Added task {task_id}: classes [{start_class}:{end_class}], "
              f"prompts [{start_prompt}:{end_prompt}]")

    def get_task_boundaries(self, pred_task_id):
        """Get class and prompt boundaries for a predicted task"""
        if not self.task_boundaries or pred_task_id >= len(self.task_boundaries):
            # Fallback for first task or invalid prediction
            return 0, self.numclasses_per_task, 0, min(self.top_k_l, len(self.desc_embeddings))

        start_class, end_class, start_prompt, end_prompt = self.task_boundaries[pred_task_id]
        return start_class, end_class, start_prompt, min(end_prompt, start_prompt + self.top_k_l)

    def predict_task(self, x_embed_norm, training_task_id=None):
        """Improved task prediction with stability"""
        if self.training and training_task_id is not None:
            return training_task_id

        if not self.task_boundaries:
            return 0

        # Use task-level keys for prediction
        task_keys = self.task_key[:len(self.task_boundaries)]
        task_keys_norm = self.l2_normalize(task_keys, dim=-1)

        # Compute similarities with temperature scaling
        similarities = torch.matmul(x_embed_norm, task_keys_norm.t()) * self.temperature
        pred_tasks = torch.argmax(similarities, dim=1)

        # Use mode for batch prediction stability
        return torch.mode(pred_tasks)[0].item()

    def forward(self, x_embed, y=None, task_id=-1, prompt_mask=None, layer_num=0, cls_features=None):
        out = {}
        if not self.prompt_pool or not self.desc_embeddings:
            return out

        # === Compute sequence embedding ===
        if self.embedding_key == "mean":
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == "max":
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == "cls" and cls_features is not None:
            x_embed_mean = cls_features
        else:
            x_embed_mean = torch.mean(x_embed, dim=1)  # fallback

        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)

        # === Task prediction ===
        pred_task_id = self.predict_task(x_embed_norm, task_id if self.training else None)
        out['predicted_task'] = pred_task_id

        # === Get boundaries ===
        start_class, end_class, start_prompt, end_prompt = self.get_task_boundaries(pred_task_id)

        # === Class-level prompt selection ===
        if self.training:
            # During training, use current task classes only
            class_keys = self.prompt_key[start_class:end_class]
        else:
            # During inference, consider all seen classes
            max_seen_class = max(tb[1] for tb in self.task_boundaries[:pred_task_id + 1])
            class_keys = self.prompt_key[:max_seen_class]

        class_keys_norm = self.l2_normalize(class_keys, dim=-1)
        class_similarities = torch.matmul(x_embed_norm, class_keys_norm.t())

        # Top-k selection with proper bounds checking
        available_classes = class_similarities.shape[1]
        actual_k = min(self.top_k_l, available_classes, end_prompt - start_prompt)

        if actual_k > 0:
            top_k_sim, top_k_idx = torch.topk(class_similarities, k=actual_k, dim=1)
            # Apply softmax for weighted combination
            top_k_weights = F.softmax(top_k_sim * self.temperature, dim=1)
        else:
            # Fallback
            top_k_weights = torch.ones(x_embed_norm.shape[0], 1, device=x_embed_norm.device)
            top_k_idx = torch.zeros(x_embed_norm.shape[0], 1, dtype=torch.long, device=x_embed_norm.device)

        # === Construct prompts from descriptors ===
        if pred_task_id < len(self.desc_embeddings):
            task_desc = self.desc_embeddings[pred_task_id]
            # Select relevant descriptors based on top-k classes
            selected_desc_idx = top_k_idx % len(task_desc)  # Handle index bounds
            selected_desc = task_desc[selected_desc_idx]  # [B, K, D]

            # Weighted combination of descriptors
            weighted_prompts = torch.sum(selected_desc * top_k_weights.unsqueeze(-1), dim=1)  # [B, D]

            # Project to prompt space
            batched_prompt = self.prompt_proj(weighted_prompts.unsqueeze(1))  # [B, 1, D]
        else:
            # Fallback: use learned prompt keys
            batched_prompt = torch.zeros(x_embed_norm.shape[0], 1, self.embed_dim, device=x_embed_norm.device)

        # === Generate final prompts ===
        final_prompts = self.generate_prompts(
            batched_prompt, start_prompt, end_prompt, layer_num, top_k_weights
        )
        out["batched_prompt"] = final_prompts

        # === Compute regularization losses ===
        if self.training and y is not None:
            # Pull constraint: encourage diversity in prompt selection
            prompt_keys_norm = self.l2_normalize(self.prompt_key[y], dim=-1)
            reduce_sim = torch.mean(torch.sum(prompt_keys_norm * x_embed_norm, dim=1))
            out['reduce_sim'] = reduce_sim

            # Task-level alignment
            if pred_task_id < len(self.task_boundaries):
                task_key_norm = self.l2_normalize(self.task_key[pred_task_id:pred_task_id + 1], dim=-1)
                task_sim = torch.mean(torch.matmul(x_embed_norm, task_key_norm.t()))
                out['reduce_sim2'] = task_sim

        return out

    # Add these methods to your existing LPrompt class in promptL.py

    def process_new_task(self, old_num_k, new_num_k, new_desc_embed):
        """
        Legacy method for backward compatibility with existing training code.
        Maps to the new add_task method.
        """
        # Calculate task_id from the progression
        current_task_id = len(self.task_boundaries)

        # Determine class indices for this task
        classes_per_task = self.num_classes // self.num_tasks
        start_class = current_task_id * classes_per_task
        end_class = min(start_class + classes_per_task, self.num_classes)
        class_indices = list(range(start_class, end_class))

        # Handle descriptor embeddings
        if current_task_id == 0:
            # First task - use new_desc_embed directly
            desc_embed = new_desc_embed
        else:
            # Subsequent tasks - extract relevant descriptors
            # new_desc_embed contains all descriptors up to current task
            start_desc = old_num_k
            desc_embed = new_desc_embed[start_desc:new_num_k] if len(new_desc_embed.shape) > 1 else new_desc_embed[
                -classes_per_task:]

        print(f"[LPrompt] process_new_task: task_id={current_task_id}, "
              f"classes={class_indices}, desc_shape={desc_embed.shape}")

        # Call the new method
        self.add_task(current_task_id, class_indices, desc_embed)

        # Update tracking variables for compatibility
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k

    def add_task(self, task_id, class_indices, desc_embeddings):
        """
        Properly register a new task with its descriptors
        """
        if len(class_indices) == 0:
            print("Warning: No class indices provided for task", task_id)
            return

        start_class = min(class_indices)
        end_class = max(class_indices) + 1
        start_prompt = len(self.desc_embeddings)

        # Ensure desc_embeddings is properly shaped
        if desc_embeddings.dim() == 1:
            desc_embeddings = desc_embeddings.unsqueeze(0)
        elif desc_embeddings.dim() == 3:
            desc_embeddings = desc_embeddings.squeeze(1)

        # Project and store descriptors
        with torch.no_grad():
            if desc_embeddings.shape[-1] != self.text_proj[0].in_features:
                # If already projected to embed_dim, skip projection
                if desc_embeddings.shape[-1] == self.embed_dim:
                    projected_desc = desc_embeddings
                else:
                    print(
                        f"Warning: Descriptor dimension mismatch. Expected {self.text_proj[0].in_features}, got {desc_embeddings.shape[-1]}")
                    # Try to handle dimension mismatch
                    if desc_embeddings.shape[-1] < self.text_proj[0].in_features:
                        # Pad with zeros
                        padding = torch.zeros(desc_embeddings.shape[0],
                                              self.text_proj[0].in_features - desc_embeddings.shape[-1],
                                              device=desc_embeddings.device)
                        desc_embeddings = torch.cat([desc_embeddings, padding], dim=1)
                    else:
                        # Truncate
                        desc_embeddings = desc_embeddings[:, :self.text_proj[0].in_features]
                    projected_desc = self.text_proj(desc_embeddings)
            else:
                projected_desc = self.text_proj(desc_embeddings)

            self.desc_embeddings.append(projected_desc)

        end_prompt = start_prompt + len(projected_desc)
        self.task_boundaries.append((start_class, end_class, start_prompt, end_prompt))
        self.current_task = max(self.current_task, task_id + 1)

        print(f"[LPrompt] Added task {task_id}: classes [{start_class}:{end_class}], "
              f"prompts [{start_prompt}:{end_prompt}], desc_shape={projected_desc.shape}")

    # Also add this method to handle the missing kmax_list and lmax_list attributes
    def init_tracking_lists(self):
        """Initialize tracking lists for backward compatibility"""
        if not hasattr(self, 'kmax_list'):
            self.kmax_list = []
        if not hasattr(self, 'lmax_list'):
            self.lmax_list = []
        if not hasattr(self, 'old_num_k'):
            self.old_num_k = 0
        if not hasattr(self, 'new_num_k'):
            self.new_num_k = 0
        if not hasattr(self, 'old_num_c'):
            self.old_num_c = 0
        if not hasattr(self, 'new_num_c'):
            self.new_num_c = 0

    # Update the __init__ method - add this at the end of __init__
    def update_init(self):
        """Add this call at the end of your LPrompt.__init__ method"""
        # Initialize backward compatibility attributes
        self.init_tracking_lists()

        print(f"[LPrompt] Initialized with {self.num_classes} classes, {self.num_tasks} tasks, "
              f"embed_dim={self.embed_dim}, max_pool_size={self.max_pool_size}")
        
    def generate_prompts(self, batched_prompt, start_prompt, end_prompt, layer_num, weights):
        """Generate final prompts using attention generators"""
        if batched_prompt.dim() == 2:
            batched_prompt = batched_prompt.unsqueeze(1)

        B, seq_len, D = batched_prompt.shape
        head_dim = D // self.num_heads

        # Reshape for multi-head attention
        prompt_heads = batched_prompt.view(B, seq_len, self.num_heads, head_dim)
        prompt_heads = prompt_heads.permute(2, 0, 1, 3)  # [H, B, seq_len, head_dim]

        k_prompts = torch.zeros_like(prompt_heads)
        v_prompts = torch.zeros_like(prompt_heads)

        layer_key = f"layer_{layer_num}"
        if layer_key not in self.generators:
            layer_num = 0  # Fallback
            layer_key = f"layer_{layer_num}"

        for head_idx in range(self.num_heads):
            head_key = f"head_{head_idx}"
            if head_key not in self.generators[layer_key]:
                continue

            k_gens = self.generators[layer_key][head_key]["k_gens"]
            v_gens = self.generators[layer_key][head_key]["v_gens"]

            prompt_head = prompt_heads[head_idx]  # [B, seq_len, head_dim]

            # Use weighted combination of generators
            for i in range(min(weights.shape[1], end_prompt - start_prompt)):
                gen_idx = start_prompt + i
                if gen_idx < len(k_gens) and gen_idx < len(v_gens):
                    weight = weights[:, i:i + 1].unsqueeze(-1)  # [B, 1, 1]
                    k_prompts[head_idx] += weight * k_gens[gen_idx](prompt_head)
                    v_prompts[head_idx] += weight * v_gens[gen_idx](prompt_head)

        # Combine k and v prompts
        kv_prompts = torch.stack([k_prompts, v_prompts], dim=0)  # [2, H, B, seq_len, head_dim]

        # Expand for prompt length
        kv_prompts = kv_prompts.unsqueeze(3).repeat(1, 1, 1, self.length, 1)  # [2, H, B, length, head_dim]
        kv_prompts = kv_prompts.permute(2, 0, 3, 1, 4)  # [B, 2, length, H, head_dim]

        # Reshape to final form
        B, dual, length, H, head_dim = kv_prompts.shape
        final_prompts = kv_prompts.reshape(B, dual * length, H * head_dim)

        return final_prompts

