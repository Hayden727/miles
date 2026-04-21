import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)


try:
    # nemotron_h (Mamba+Attention hybrid) surfaces >2-tuple at
    # TransformerLayer._forward_attention's unpack. Three-layer defense:
    # (1) SelfAttention.forward (if that's where the extra element appears)
    # (2) SelfAttention.__call__ (covers nn.Module dispatch + hooks)
    # (3) Diagnostic print of actual self_attention class and tuple size
    #     inside TransformerLayer._forward_attention so we can see what's
    #     happening when all else fails.
    import sys as _miles_sys
    _miles_sys.stderr.write(">>> miles nemotron_h attn-shim: installing\n")
    _miles_sys.stderr.flush()

    from megatron.core.transformer.attention import SelfAttention as _MilesSelfAttention
    from megatron.core.transformer.transformer_layer import TransformerLayer as _MilesTL

    _miles_attn_diag_logged = [False]

    _orig_sa_forward = _MilesSelfAttention.forward

    def _miles_sa_forward(self, *args, **kwargs):
        ret = _orig_sa_forward(self, *args, **kwargs)
        if isinstance(ret, tuple) and len(ret) > 2:
            return ret[0], ret[1]
        return ret

    _MilesSelfAttention.forward = _miles_sa_forward

    _orig_sa_call = _MilesSelfAttention.__call__

    def _miles_sa_call(self, *args, **kwargs):
        ret = _orig_sa_call(self, *args, **kwargs)
        if isinstance(ret, tuple) and len(ret) > 2:
            return ret[0], ret[1]
        return ret

    _MilesSelfAttention.__call__ = _miles_sa_call

    _orig_fwd_attn = _MilesTL._forward_attention

    def _miles_fwd_attn(self, *args, **kwargs):
        # nemotron_h / Mamba hybrid: non-attention positions have IdentityOp
        # in self_attention. TransformerLayer._forward_attention's unpack
        # fails on the raw tensor returned. Short-circuit.
        if type(self.self_attention).__name__ == "IdentityOp":
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            return hidden_states, None
        return _orig_fwd_attn(self, *args, **kwargs)

    _MilesTL._forward_attention = _miles_fwd_attn

    _orig_fwd_mlp = _MilesTL._forward_mlp

    def _miles_fwd_mlp(self, hidden_states, *args, **kwargs):
        # Symmetric fix: attention-only layers have IdentityOp in self.mlp.
        # _forward_mlp's `mlp_output, mlp_output_bias = mlp_output_with_bias`
        # unpack fails on IdentityOp return (raw tensor). Short-circuit so
        # the attention output flows through unchanged.
        if type(self.mlp).__name__ == "IdentityOp":
            return hidden_states
        return _orig_fwd_mlp(self, hidden_states, *args, **kwargs)

    _MilesTL._forward_mlp = _miles_fwd_mlp

    # For PP>1: megatron.bridge's param_mapping.broadcast_obj_from_pp_rank
    # calls torch.distributed.broadcast_object_list with the pp_group.
    # miles wraps groups in ReloadableProcessGroup (subclass of
    # torch.distributed.ProcessGroup) which is NOT registered in
    # torch.distributed._world.pg_group_ranks, so get_group_rank raises
    # "Group ... is not registered". Unwrap to the inner real group before
    # the broadcast.
    from miles.utils.reloadable_process_group import ReloadableProcessGroup as _MilesRPG
    from megatron.bridge.models.conversion import param_mapping as _MilesBridgeParamMapping

    _orig_broadcast_obj_from_pp_rank = _MilesBridgeParamMapping.MegatronParamMapping.broadcast_obj_from_pp_rank

    def _miles_broadcast_obj_from_pp_rank(self, obj, name=None):
        if isinstance(self.pp_group, _MilesRPG):
            _orig_pp = self.pp_group
            self.pp_group = _orig_pp.group  # inner real torch ProcessGroup
            try:
                return _orig_broadcast_obj_from_pp_rank(self, obj, name)
            finally:
                self.pp_group = _orig_pp
        return _orig_broadcast_obj_from_pp_rank(self, obj, name)

    _MilesBridgeParamMapping.MegatronParamMapping.broadcast_obj_from_pp_rank = _miles_broadcast_obj_from_pp_rank

except Exception as _e:  # best-effort shim
    logging.warning("nemotron_h attn-unpack shim not applied: %s", _e)
