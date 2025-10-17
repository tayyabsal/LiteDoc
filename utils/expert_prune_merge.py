import torch
from collections import defaultdict
from deepseek_vl2.utils.cka_utils import find_similar_experts

def prune_and_merge_experts(
    model,
    cka_matrices: dict,
    similarity_threshold: float = 0.7,
    max_prune_ratio: float = 0.5,
    dynamic_skip_delta: float = 0.1,
    custom_merge_pairs: dict = None,
):
    """
    Prune and merge experts based on CKA similarity.

    If custom_merge_pairs is provided, use it instead of computing similar pairs.

    Args:
        model: DeepseekVLV2ForCausalLM instance.
        cka_matrices: dict[layer_idx] -> torch.Tensor [E, E].
        similarity_threshold: CKA threshold to group experts into clusters.
        max_prune_ratio: maximum fraction of experts to prune per layer.
        dynamic_skip_delta: if max importance - min importance < delta, skip pruning layer.
        custom_merge_pairs: dict[layer_idx] -> list of (i, j) pairs to merge manually.
    """
    layers = model.language.model.layers
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer.mlp, 'experts'):
            continue

        experts = layer.mlp.experts
        E = len(experts)
        if E < 2:
            continue

        sim_mat = cka_matrices.get(layer_idx)
        if sim_mat is None:
            continue

        # 1. Build clusters via union-find based on similarity
        parent = list(range(E))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        if custom_merge_pairs and layer_idx in custom_merge_pairs:
            similar = [(i, j, 1.0) for (i, j) in custom_merge_pairs[layer_idx]]
        else:
            similar = find_similar_experts({layer_idx: sim_mat}, threshold=similarity_threshold).get(layer_idx, [])

        for i, j, _ in similar:
            union(i, j)

        clusters = defaultdict(list)
        for e in range(E):
            clusters[find(e)].append(e)

        # 2. Merge clusters: collapse each cluster into the first member
        merged_map = {}
        for root, members in clusters.items():
            primary = members[0]
            if len(members) > 1:
                state_dicts = [experts[m].state_dict() for m in members]
                merged = {}
                for key in state_dicts[0]:
                    vals = torch.stack([sd[key] for sd in state_dicts], dim=0)
                    merged[key] = vals.mean(dim=0)
                experts[primary].load_state_dict(merged)
            for m in members:
                merged_map[m] = primary

        # 3. Compute importance: mean CKA similarity per expert
        importance = sim_mat.mean(dim=1).cpu()
        imp_max, imp_min = importance.max().item(), importance.min().item()
        if (imp_max - imp_min) < dynamic_skip_delta:
            print(f"Layer {layer_idx}: importance range {imp_max - imp_min:.4f} < {dynamic_skip_delta}, skipping pruning.")
            continue

        # determine prune candidates
        prune_limit = max(1, int(E * max_prune_ratio))
        sorted_idxs = sorted(range(E), key=lambda x: importance[x].item())
        to_prune = []
        for idx in sorted_idxs:
            if merged_map.get(idx, idx) != idx:
                continue
            to_prune.append(idx)
            if len(to_prune) >= prune_limit:
                break

        for idx in to_prune:
            zero_dict = {k: torch.zeros_like(v) for k, v in experts[idx].state_dict().items()}
            experts[idx].load_state_dict(zero_dict)
            if hasattr(layer.mlp, 'expert_dropout_mask'):
                layer.mlp.expert_dropout_mask[idx] = True

        print(f"Layer {layer_idx}: merged {len(clusters)} clusters, pruned {len(to_prune)} experts.")

    print("Pruning and merging complete.")

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
