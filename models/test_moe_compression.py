import torch
from deepseek_vl2.models.modeling_deepseek import DeepseekV2MoE

def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

class DummyConfig:
    # Transformer hidden size (must match what MLP expects)
        vocab_size=102400
        hidden_size=4096
        n_routed_experts = 64
        intermediate_size=11008
        moe_intermediate_size = 1407
        num_hidden_layers=30
        num_attention_heads=32
        num_key_value_heads=32
        n_shared_experts = None
       
        ep_size = 1
        routed_scaling_factor = 1.0
        kv_lora_rank = 512
        q_lora_rank = 1536
        qk_rope_head_dim = 64
        v_head_dim = 128
        qk_nope_head_dim = 128
        topk_method = 'gready'
        n_group = None
        topk_group = None
        num_experts_per_tok = None
        moe_layer_freq = 1
        first_k_dense_replace = 0
        norm_topk_prob = False
        scoring_func = 'softmax'
        aux_loss_alpha = 0.001
        seq_aux = True
        hidden_act="silu"
        max_position_embeddings=2048
        initializer_range=0.02
        rms_norm_eps=1e-6
        use_cache=True
        pad_token_id=None
        bos_token_id=100000
        eos_token_id=100001
        pretraining_tp=1
        tie_word_embeddings=False
        rope_theta=10000.0
        rope_scaling=None
        attention_bias=False
        attention_dropout=0.0
        use_mla=True



# 1) Without compression
cfg_plain = DummyConfig()
cfg_plain.svd_rank = None
model_plain = DeepseekV2MoE(cfg_plain)
plain_params = count_params(model_plain)

# 2) With compression
cfg_merged = DummyConfig()
cfg_merged.svd_rank = 32
model_merged = DeepseekV2MoE(cfg_merged)
merged_params = count_params(model_merged)

print(f"Params w/o merging : {plain_params:,}")
print(f"Params w/ merging  : {merged_params:,}")
print(f"Compression ratio  : {merged_params/plain_params:.2%}")
