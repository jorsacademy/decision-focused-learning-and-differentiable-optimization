"""Run the synthetic trained-model optimization benchmark."""

from __future__ import annotations

import json

import numpy as np

from trained_ml_opt.demo import run_demo


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    print(json.dumps(run_demo(), cls=NumpyEncoder, indent=2))
