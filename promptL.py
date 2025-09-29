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
            text_embed_dim=128,
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
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.numclasses_per_task = int(self.num_classes / self.num_tasks)

        # pool size = tasks * prompts_per_task
        self.pool_size = self.num_tasks * self.prompts_per_task

        # === Prompt keys ===
        key_shape = (self.num_classes, embed_dim)
        task_shape = (self.num_tasks, embed_dim)
        if prompt_key_init == "zero":
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            self.prompt_key2 = nn.Parameter(torch.zeros(task_shape))
        elif prompt_key_init == "uniform":
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            self.prompt_key2 = nn.Parameter(torch.randn(task_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
            nn.init.uniform_(self.prompt_key2, -1, 1)
        elif prompt_key_init == "ortho":
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            self.prompt_key2 = nn.Parameter(torch.randn(task_shape))
            nn.init.orthogonal_(self.prompt_key)
            nn.init.orthogonal_(self.prompt_key2)

        # === Attention generators - start with top_k_l, expand later ===
        self.k_comp_gen = nn.ModuleDict()
        self.v_comp_gen = nn.ModuleDict()
        for i in range(num_layers):
            self.k_comp_gen[str(i)] = nn.ModuleDict()
            self.v_comp_gen[str(i)] = nn.ModuleDict()
            for j in range(num_heads):
                self.k_comp_gen[str(i)][str(j)] = nn.ModuleList()
                self.v_comp_gen[str(i)][str(j)] = nn.ModuleList()
                for _ in range(self.top_k_l):
                    k_comp = Gen_Attention2(embed_dim // num_heads, 1, False, 0.0, 0.0)
                    v_comp = Gen_Attention2(embed_dim // num_heads, 1, False, 0.0, 0.0)
                    self.k_comp_gen[str(i)][str(j)].append(k_comp)
                    self.v_comp_gen[str(i)][str(j)].append(v_comp)

        # === Tracking ===
        self.old_num_k, self.new_num_k = 0, 0
        self.old_num_c, self.new_num_c = 0, 0
        self.kmax_list, self.lmax_list = [], []

        print(
            f"Num Tasks: {self.num_tasks}, pool_size: {self.pool_size}, "
            f"num_classes: {self.num_classes}, top_k: {self.top_k}"
        )
        print("Init prompt finished")

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def process_new_task(self, old_num_k, new_num_k, new_desc_embed):
        """Process new task - expand attention generators like original."""
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k

        print("Old Num K:", self.old_num_k, "New Num K:", self.new_num_k)
        self.kmax_list.append(new_num_k)

        self.desc_embed = new_desc_embed
        self.old_num_c = self.new_num_c
        self.new_num_c = self.new_num_c + int(self.num_classes / self.num_tasks)
        self.lmax_list.append(self.new_num_c)
        print(self.kmax_list)
        print(self.lmax_list)

        # Expand attention generators for new prompts (like original)
        if self.old_num_c > 0:
            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    for k in range(self.old_num_k - self.top_k_l, self.old_num_k):
                        k_comp = Gen_Attention2(int(self.embed_dim / self.num_heads), 1, False, 0., 0.).cuda()
                        self.k_comp_gen[str(i)][str(j)].append(k_comp)

                        v_comp = Gen_Attention2(int(self.embed_dim / self.num_heads), 1, False, 0., 0.).cuda()
                        self.v_comp_gen[str(i)][str(j)].append(v_comp)

            # Freeze old generators
            for i in range(self.num_layers):
                for j in range(self.num_heads):
                    for k in range(self.old_num_k):
                        gen = self.k_comp_gen[str(i)][str(j)][k]
                        for n, p in gen.named_parameters():
                            p.requires_grad = False
                        gen = self.v_comp_gen[str(i)][str(j)][k]
                        for n, p in gen.named_parameters():
                            p.requires_grad = False

    def forward(self, x_embed, y, task_id=-1, prompt_mask=None, layer_num=-1, cls_features=None):
        out = dict()
        if self.prompt_pool:
            # === Compute embedding ===
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s = self.old_num_k
            f = self.new_num_k

            # === Normalizations ===
            prompt_key2_norm = self.l2_normalize(self.prompt_key2[0:len(self.kmax_list)], dim=-1)
            prompt_key_norm = self.l2_normalize(self.prompt_key[0:self.lmax_list[-1]], dim=-1)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)

            prompt_key2_norm_allclass = torch.flatten(
                (prompt_key2_norm.unsqueeze(1).repeat(1, self.numclasses_per_task, 1)),
                start_dim=0, end_dim=1
            )

            # === Task prediction ===
            sim = torch.matmul(prompt_key_norm, x_embed_norm.t()).t()
            sim2 = torch.matmul(prompt_key2_norm_allclass, x_embed_norm.t()).t()
            sim2 = (1000 * sim) * (1000 * sim2)

            (sim2_top_k, idx2) = torch.topk(sim2, k=1, dim=1)
            idx2 = torch.floor(idx2 / self.numclasses_per_task).long()

            if self.training:
                idx2[0:] = task_id
                pred_task_id = task_id
            else:
                pred_task_id = torch.mode(idx2.detach().clone().flatten().cpu()).values.item()

            # === Calculate reduce_sim2 (EXACTLY like original) ===
            batched_key2_norm = prompt_key2_norm[idx2]
            sim2 = batched_key2_norm * x_embed_norm.unsqueeze(1)
            reduce_sim = torch.sum(sim2) / x_embed.shape[0]
            out['reduce_sim2'] = reduce_sim

            # === Determine task boundaries ===
            f = self.kmax_list[pred_task_id]
            fl = self.lmax_list[pred_task_id]
            if task_id == 0 or pred_task_id == 0:
                sl = 0
                s = 0
                f = self.top_k_l
            else:
                sl = self.lmax_list[pred_task_id - 1]
                s = self.kmax_list[pred_task_id - 1]
                f = self.kmax_list[pred_task_id]
            out['max_t'] = pred_task_id + 1

            # === Class-level similarities ===
            if self.training:
                prompt_key = self.prompt_key[sl:fl]
            else:
                prompt_key = self.prompt_key[0:fl]

            prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()).t()
            out['similarity'] = similarity

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k_l, dim=1)

            batched_prompt_raw = torch.matmul(similarity_top_k.unsqueeze(1), self.desc_embed[idx])
            out['desc_embed'] = batched_prompt_raw.squeeze(1).detach().clone()

            # === Calculate reduce_sim (for main loss) ===
            if self.training:
                pkey_norm = self.l2_normalize(self.prompt_key, dim=-1)
                batched_key_norm = pkey_norm[y]
                sim = batched_key_norm * x_embed_norm
                reduce_sim = torch.sum(sim) / x_embed.shape[0]
                out['reduce_sim'] = reduce_sim

            # === Prepare batched prompts ===
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = F.interpolate(batched_prompt_raw, [self.embed_dim // self.num_heads])
                batched_prompt_raw = batched_prompt_raw.repeat(1, self.num_heads, 1)
                batched_prompt_raw = batched_prompt_raw.view(
                    x_embed.shape[0], self.num_heads, self.embed_dim // self.num_heads
                ).unsqueeze(0)
                batched_prompt_raw = torch.cat((batched_prompt_raw, batched_prompt_raw), dim=0)
            else:
                batched_prompt_raw = F.interpolate(batched_prompt_raw, [self.embed_dim // self.num_heads])
                batched_prompt_raw = batched_prompt_raw.repeat(1, self.num_heads, 1)
                batched_prompt_raw = batched_prompt_raw.view(
                    x_embed.shape[0], self.num_heads, self.embed_dim // self.num_heads
                )
                batched_prompt_raw = torch.cat((batched_prompt_raw, batched_prompt_raw), dim=0)

            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

        # === Prepare similarity for attention ===
        similarity_new = torch.zeros_like(similarity, device=similarity.device)
        for c in range(idx.shape[1]):
            similarity_new[:, idx[:, c]] = similarity_top_k[:, c]

        batched_prompt = self.compute_att_over_prompt(batched_prompt_raw, s, f, layer_num, similarity_new)
        out['batched_prompt'] = batched_prompt

        return out

    def compute_att_over_prompt(self, batched_prompt, s, f, layer_num, similarity):
        """Compute attention over prompts (like original)."""
        k_prompt_layer = batched_prompt[0]  # B, num_heads, head_dim
        v_prompt_layer = batched_prompt[1]  # B, num_heads, head_dim

        k_prompt_layer = k_prompt_layer.permute(1, 0, 2)  # num_heads, B, head_dim
        v_prompt_layer = v_prompt_layer.permute(1, 0, 2)  # num_heads, B, head_dim

        n_heads, batch_size, head_dim = k_prompt_layer.shape

        new_k_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, head_dim), device=v_prompt_layer.device)

        for h in range(self.num_heads):
            k_comp_gen = self.k_comp_gen[str(layer_num)][str(h)]
            v_comp_gen = self.v_comp_gen[str(layer_num)][str(h)]

            k_prompt_head = k_prompt_layer[h].unsqueeze(1)  # B, 1, head_dim
            v_prompt_head = v_prompt_layer[h].unsqueeze(1)  # B, 1, head_dim

            for p in range(s, f):
                k_comp_val = k_comp_gen[p]
                v_comp_val = v_comp_gen[p]

                new_k_prompt_layer[h] += k_comp_val(k_prompt_head).squeeze(1) * similarity[:, p - s].unsqueeze(1)
                new_v_prompt_layer[h] += v_comp_val(v_prompt_head).squeeze(1) * similarity[:, p - s].unsqueeze(1)

        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0)
        new_batched_prompt = new_batched_prompt.unsqueeze(3).repeat(1, 1, 1, self.length, 1)
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4)

        return new_batched_prompt