"""Child process for audit/regenerate_layouts.py. Run with `python -I` so that nothing but the
worktree given on the command line provides cbf_study. Regenerates the validation and pilot layout
sets with that commit's own code and config, saves the arrays and prints their hashes as JSON."""
import hashlib
import json
import os
import sys

wt, outdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, wt)
import numpy as np                                   # noqa: E402
import cbf_study.scenario as scn                     # noqa: E402
from cbf_study.config import StudyConfig             # noqa: E402
from cbf_study.scenario import sample_layout         # noqa: E402

assert os.path.realpath(scn.__file__).startswith(os.path.realpath(wt)), scn.__file__
cfg = StudyConfig.load(os.path.join(wt, "config", "study.toml"))
fc, sc = cfg.layout_filter_config(), cfg.scenario_config()
os.makedirs(outdir, exist_ok=True)
rows = []
for split, seed_fn, n in (("validation", cfg.validation_seed, 200), ("pilot", cfg.pilot_seed, 500)):
    for K in (0, 1, 2, 3, 5, 8):
        seed = seed_fn(K)
        lay = sample_layout(seed, n, K, fc, sc)
        arrs = dict(init_poses=lay.init_poses, obst_xy=lay.obst_xy, goal=lay.goal, slots=lay.slots)
        np.savez(os.path.join(outdir, f"{split}_K{K}_n{n}.npz"), **arrs)
        h = {f"sha256_{k}": hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrs.items()}
        h["dtypes_shapes"] = ";".join(f"{k}:{v.dtype}{list(v.shape)}" for k, v in arrs.items())
        # the format of the hashes recorded before the sampler change (tests/test_config.py)
        h["sha256_16_init_obst"] = hashlib.sha256(lay.init_poses.tobytes() + lay.obst_xy.tobytes()).hexdigest()[:16]
        rows.append(dict(split=split, density=K, layout_seed=seed, n_layouts=n, **h))
print(json.dumps(dict(scenario_file=scn.__file__, numpy=np.__version__, config_version=getattr(cfg, "version", 1),
                      env_fingerprint=cfg.env_fingerprint(), rows=rows)))
