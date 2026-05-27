import sys

from transformer_engine.pytorch.optimizers import fused_adam

F = fused_adam.__file__
src = open(F).read()
if "_nla_state_restore" in src:
    print(f"already patched: {F}")
    sys.exit(0)
OLD = '''        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})'''
NEW = '''        # NLA v12: super() with state={} WIPES self.state. The loop below then
        # _initialize_state's a fresh ~29GB while distrib_optimizer:814's caller
        # still holds the old 29GB by ref → 2× peak → OOM at PP3 actor (76/79GB).
        # Save+restore so set_scaled_state finds existing tensors and copy_'s in-place.
        _nla_state_restore = dict(self.state)
        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
        for _p, _st in _nla_state_restore.items():
            self.state[_p] = _st'''
assert OLD in src, f"anchor not found in {F}"
open(F, "w").write(src.replace(OLD, NEW, 1))
print(f"patched: {F}")
