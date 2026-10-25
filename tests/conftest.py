import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbf_study.cbf_qp import FilterConfig          # noqa: E402
from cbf_study.protocol import ProtocolConfig      # noqa: E402


def make_cfgs(dt=0.1, robot_r=0.1, obst_r=0.2, ell=0.05, v_min=-0.5, v_max=0.5, w_max=1.0,
              alpha=2.0, sensing_range=1.0, comm_range=1.5, max_steps=300, goal_tol=0.1,
              formation_tol=0.1):
    fc = FilterConfig(dt=dt, robot_r=robot_r, obst_r=obst_r, ell=ell, v_min=v_min, v_max=v_max,
                      w_max=w_max, alpha=alpha, sensing_range=sensing_range, comm_range=comm_range)
    pc = ProtocolConfig(dt=dt, robot_r=robot_r, obst_r=obst_r, v_min=v_min, v_max=v_max,
                        w_max=w_max, max_steps=max_steps, goal_tol=goal_tol,
                        formation_tol=formation_tol)
    return fc, pc


@pytest.fixture
def cfgs():
    return make_cfgs()
