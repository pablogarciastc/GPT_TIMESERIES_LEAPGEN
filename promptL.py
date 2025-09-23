import torch
import torch.nn as nn
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

        # pool size = tasks * prompts_per_task
        self.pool_size = self.num_tasks * self.prompts_per_task

        # === Prompt keys ===
        key_shape = (self.num_classes, embed_dim)
        task_shape = (self.num_tasks, embed_dim)
        if prompt_key_init == "zero":
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            self.prompt_key2 = nn.Parameter(torch.zeros(task_shape))
        elif prompt_key_init == "uniform":
            self.prompt_key = nn.Parameter(torch.empty(key_shape))
            self.prompt_key2 = nn.Parameter(torch.empty(task_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
            nn.init.uniform_(self.prompt_key2, -1, 1)
        elif prompt_key_init == "ortho":
            self.prompt_key = nn.Parameter(torch.empty(key_shape))
            self.prompt_key2 = nn.Parameter(torch.empty(task_shape))
            nn.init.orthogonal_(self.prompt_key)
            nn.init.orthogonal_(self.prompt_key2)

        # === Proyecciones ===
        self.text_proj = nn.Linear(text_embed_dim, embed_dim, bias=False)
        self.prompt_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # === Generadores de atención ===
        self.k_comp_gen = nn.ModuleDict()
        self.v_comp_gen = nn.ModuleDict()
        for i in range(num_layers):
            self.k_comp_gen[str(i)] = nn.ModuleDict()
            self.v_comp_gen[str(i)] = nn.ModuleDict()
            for j in range(num_heads):
                self.k_comp_gen[str(i)][str(j)] = nn.ModuleList()
                self.v_comp_gen[str(i)][str(j)] = nn.ModuleList()
                for _ in range(top_k_l):
                    k_comp = Gen_Attention2(embed_dim // num_heads, 1, False, 0.0, 0.0)
                    v_comp = Gen_Attention2(embed_dim // num_heads, 1, False, 0.0, 0.0)
                    self.k_comp_gen[str(i)][str(j)].append(k_comp)
                    self.v_comp_gen[str(i)][str(j)].append(v_comp)

        # === Tracking ===
        self.old_num_k, self.new_num_k = 0, 0
        self.old_num_c, self.new_num_c = 0, 0
        self.kmax_list, self.lmax_list = [], []

        print(
            f"[LPrompt] Initialized | tasks={self.num_tasks}, pool_size={self.pool_size}, "
            f"num_classes={self.num_classes}, top_k={self.top_k}, top_k_l={self.top_k_l}"
        )

    # ---------------------------
    # Helpers
    # ---------------------------
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
        return x * x_inv_norm

    def process_new_task(self, old_num_k, new_num_k, new_desc_embed):
        """Actualizar pool de prompts con nuevas descripciones."""
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k
        self.kmax_list.append(new_num_k)

        # proyectar descriptores de texto → embed_dim
        self.desc_embed = self.text_proj(new_desc_embed)

        self.old_num_c = self.new_num_c
        self.new_num_c += self.num_classes // self.num_tasks
        self.lmax_list.append(self.new_num_c)

        print(f"[LPrompt] Updated | old_num_k={old_num_k}, new_num_k={new_num_k}")
        print("kmax_list:", self.kmax_list, "lmax_list:", self.lmax_list)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(
        self, x_embed, y=None, task_id=-1, prompt_mask=None, layer_num=0, cls_features=None
    ):
        out = dict()
        if not self.prompt_pool:
            return out

        # === Compute sequence embedding ===
        if self.embedding_key == "mean":
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == "max":
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == "mean_max":
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == "cls":
            x_embed_mean = cls_features if cls_features is not None else torch.max(x_embed, dim=1)[0]
        else:
            raise NotImplementedError(f"Unknown embedding_key={self.embedding_key}")

        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)

        # === Similarities ===
        prompt_key_norm = self.l2_normalize(self.prompt_key[: self.lmax_list[-1]], dim=-1)
        similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()).t()
        out["similarity"] = similarity

        # top-k
        available_k = similarity.shape[1]
        actual_k = min(self.top_k_l, available_k)
        sim_top_k, idx = torch.topk(similarity, k=actual_k, dim=1)

        # === Construir prompts a partir de desc_embed ===
        batched_prompt_raw = torch.matmul(sim_top_k.unsqueeze(1), self.desc_embed[idx])
        out["desc_embed"] = batched_prompt_raw.squeeze(1).detach().clone()

        # proyectar si es necesario
        if batched_prompt_raw.shape[-1] == self.embed_dim:
            batched_prompt_raw = self.prompt_proj(batched_prompt_raw)

        # === Final prompts ===
        out["batched_prompt"] = self.compute_att_over_prompt(
            batched_prompt_raw, self.old_num_k, self.new_num_k, layer_num, similarity
        )
        return out

    def compute_att_over_prompt(self, batched_prompt, s, f, layer_num, similarity):
        """Atención sobre prompts seleccionados."""




        if batched_prompt.dim() == 2:  # [B, D]
            B, C = batched_prompt.shape
            head_dim = C // self.num_heads
            batched_prompt = batched_prompt.view(B, self.num_heads, head_dim)

        k_prompt_layer = batched_prompt.permute(1, 0, 2)
        v_prompt_layer = batched_prompt.permute(1, 0, 2)

        n_heads, batch_size, head_dim = k_prompt_layer.shape
        new_k_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=v_prompt_layer.device)

        for h in range(self.num_heads):
            k_comp_gen = self.k_comp_gen[str(layer_num)][str(h)]
            v_comp_gen = self.v_comp_gen[str(layer_num)][str(h)]
            k_prompt_head = k_prompt_layer[h].unsqueeze(1)
            v_prompt_head = v_prompt_layer[h].unsqueeze(1)
            for p in range(s, f):
                new_k_prompt_layer[h] += (
                    k_comp_gen[p](k_prompt_head).squeeze(1) * similarity[:, p - s].unsqueeze(1)
                )
                new_v_prompt_layer[h] += (
                    v_comp_gen[p](v_prompt_head).squeeze(1) * similarity[:, p - s].unsqueeze(1)
                )

        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0)
        new_batched_prompt = new_batched_prompt.unsqueeze(3).repeat(1, 1, 1, self.length, 1)
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4)
        B, dual, length, H, D = new_batched_prompt.shape
        new_batched_prompt = new_batched_prompt.reshape(B, dual * length, H * D)

        # Añadir debug información
        print(f"[DEBUG] compute_att_over_prompt: s={s}, e={e}")
        print(f"[DEBUG] k_comp_gen length: {len(self.k_comp_gen)}")
        print(f"[DEBUG] similarity shape: {similarity.shape}")
        print(f"[DEBUG] k_prompt_head shape: {k_prompt_head.shape}")

        for p in range(s, e):
            print(f"[DEBUG] Processing p={p}")
            if p >= len(self.k_comp_gen):
                print(f"[ERROR] Index {p} out of range for k_comp_gen (length {len(self.k_comp_gen)})")
                # Crear el componente faltante o saltear
                break

        return new_batched_prompt
