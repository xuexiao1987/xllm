# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise both production layout implementations against the same inputs."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from xllm.python.models.deepseek_v41_layout import CSA2LayerPlan  # noqa: E402


def main() -> None:
    cases = [
        ([0, 2, 2, 1, 1, 1], [1, 3], [1, 3, 4], 3),
        ([0, 2, 2, 1, 1, 1], [1, 3], [1, 3, 4], 4),
        ([0, 0], [], [], -1),
        ([1, 1], [0], [0], -1),
        ([2, 2], [0], [0, 1], 1),
        ([0, 2, 0, 2], [1], [1, 3], -1),
        ([0, 2, 2, 1, 1, 1], [1, 3], [1, 3, 4], 1),
        ([0, 2], [1], [1], 2),
        ([0, 2], [1], [1], -2),
        ([0, 2], [1], [1], 0),
        ([2, 1], [0], [0], -1),
        ([2, 2], [1, 0], [0, 1], -1),
        ([2, 2], [0, 0], [0], -1),
        ([2, 2], [0], [0, 0], -1),
        ([0, 0], [0], [0], -1),
        ([2, 2], [0], [], -1),
        ([2, 2], [2], [2], -1),
        ([2, 0], [0], [0, 1], -1),
        ([4, 4], [0], [0], -1),
        ([0, 0] + [2] * 18 + [1] * 20, [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36], 20),
    ]
    for ratios, sources, indices, candidate in cases:
        args = [sys.argv[1], *(",".join(map(str, values)) for values in (ratios, sources, indices)), str(candidate)]
        native = subprocess.run(args, capture_output=True, text=True, timeout=10)
        try:
            plan = CSA2LayerPlan.build(len(ratios), ratios, sources, indices)
            plan.validate_candidate_source(candidate)
        except ValueError:
            assert native.returncode != 0, (args, native.stdout)
            continue
        assert native.returncode == 0, (args, native.stderr)
        lines = native.stdout.splitlines()
        assert tuple(map(int, lines[0].split())) == plan.manager_ratios
        expected = []
        for i in range(len(plan)):
            mode = plan.mode(i)
            expected.append(
                f"{mode.mode} {mode.compress_ratio} {mode.main_kv_source_layer} "
                f"{int(mode.is_full)} {int(mode.is_full and mode.compress_ratio == 2)}"
            )
        assert lines[1:] == expected, (args, lines, expected)
    print(f"{len(cases)} cross-language layout cases passed")


if __name__ == "__main__":
    main()
