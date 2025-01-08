# flake8: noqa: B950
# fmt: off
# This file was generated by AutoHeuristic. Do not modify it manually!
# To regenerate this file, take a look at the steps in the README.md file inside torchgen/_autoheuristic/mm/
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class MMRankingH100(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 232448
            and str(metadata.device_capa) == "(9, 0)"
        )

    def get_confidence_threshold(self) -> float:
        return 0.0

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
        self.choices.append('extern_mm')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=2_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=1')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=1_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=16_numstages=2_numwarps=2')
        self.choices.append('type=triton_BLOCK-M=32_BLOCK-K=64_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=16_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=32_numstages=5_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=2_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=4_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=16_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=32_numstages=5_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=8')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=4_numwarps=4')
        self.choices.append('type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=64_numstages=5_numwarps=4')

    def get_name(self) -> str:
        return 'mm'

    def get_best_choices(self, context: AHContext) -> Optional[List[tuple[float, int]]]:
        if context.get_value('arith_intensity') <= 29.89772129058838:
            if context.get_value('n') <= 34.0:
                if context.get_value('n') <= 18.0:
                    if context.get_value('k*n') <= 432.0:
                        if context.get_value('arith_intensity') <= 7.8700292110443115:
                            return [(0.098, 128), (0.098, 129), (0.098, 127), (0.073, 14), (0.073, 16), (0.073, 12), (0.073, 154), (0.073, 156), (0.073, 157), (0.073, 155), (0.049, 10), (0.049, 94), (0.049, 95), (0.048, 96)]
                        else:
                            return [(0.091, 154), (0.073, 10), (0.073, 15), (0.073, 13), (0.073, 11), (0.073, 17), (0.073, 16), (0.073, 14), (0.073, 12), (0.055, 127), (0.054, 157), (0.054, 156), (0.054, 155), (0.036, 129), (0.036, 128), (0.018, 41), (0.018, 43)]
                    else:
                        if context.get_value('k') <= 40.0:
                            return [(0.070, 39), (0.069, 45), (0.069, 41), (0.069, 43), (0.069, 111), (0.069, 112), (0.056, 38), (0.056, 40), (0.056, 42), (0.056, 44), (0.056, 174), (0.056, 173), (0.056, 175), (0.056, 134), (0.056, 172), (0.056, 135), (0.014, 154), (0.014, 127)]
                        else:
                            return [(0.147, 144), (0.119, 143), (0.087, 142), (0.083, 0), (0.073, 191), (0.059, 69), (0.050, 67), (0.046, 70), (0.041, 1), (0.036, 174), (0.032, 43), (0.032, 123), (0.028, 40), (0.027, 42), (0.027, 173), (0.023, 175), (0.018, 66), (0.014, 192), (0.014, 193), (0.014, 139), (0.014, 68), (0.014, 127)]
                else:
                    if context.get_value('mat1_stride_0') <= 40.0:
                        if context.get_value('mat1_stride_0') <= 20.0:
                            return [(0.109, 23), (0.109, 21), (0.109, 20), (0.088, 0), (0.087, 131), (0.066, 18), (0.065, 130), (0.065, 132), (0.065, 159), (0.065, 160), (0.065, 161), (0.065, 158), (0.022, 22), (0.022, 19)]
                        else:
                            return [(0.065, 46), (0.064, 52), (0.064, 50), (0.064, 48), (0.064, 51), (0.064, 49), (0.064, 47), (0.064, 53), (0.064, 181), (0.064, 177), (0.064, 179), (0.064, 176), (0.038, 130), (0.038, 136), (0.026, 182), (0.026, 178), (0.026, 180), (0.026, 137), (0.025, 158), (0.013, 114), (0.013, 113)]
                    else:
                        if context.get_value('mat1_stride_0') <= 68.0:
                            return [(0.138, 140), (0.125, 195), (0.100, 71), (0.100, 74), (0.100, 196), (0.100, 194), (0.100, 197), (0.075, 75), (0.062, 72), (0.062, 73), (0.012, 180), (0.012, 51), (0.012, 182)]
                        else:
                            return [(0.124, 180), (0.124, 182), (0.114, 75), (0.103, 74), (0.093, 51), (0.093, 71), (0.072, 72), (0.062, 194), (0.052, 145), (0.052, 195), (0.021, 48), (0.021, 50), (0.021, 47), (0.020, 124), (0.010, 147), (0.010, 146), (0.010, 46)]
            else:
                if context.get_value('k') <= 18.0:
                    if context.get_value('m*k') <= 528.0:
                        return [(0.097, 88), (0.087, 92), (0.077, 90), (0.058, 105), (0.058, 103), (0.058, 104), (0.058, 99), (0.058, 100), (0.058, 106), (0.058, 93), (0.057, 91), (0.057, 97), (0.057, 98), (0.057, 101), (0.048, 102), (0.029, 87), (0.029, 89)]
                    else:
                        if context.get_value('n') <= 80.0:
                            return [(0.057, 161), (0.057, 130), (0.057, 24), (0.056, 164), (0.056, 163), (0.056, 166), (0.056, 168), (0.056, 30), (0.056, 28), (0.056, 26), (0.056, 25), (0.056, 27), (0.056, 29), (0.056, 31), (0.042, 131), (0.028, 99), (0.028, 101), (0.028, 100), (0.028, 167), (0.028, 165), (0.028, 133)]
                        else:
                            return [(0.110, 164), (0.108, 163), (0.106, 168), (0.069, 161), (0.066, 151), (0.060, 152), (0.055, 165), (0.050, 27), (0.050, 29), (0.048, 131), (0.043, 153), (0.037, 133), (0.037, 130), (0.028, 8), (0.028, 5), (0.027, 7), (0.026, 26), (0.016, 162), (0.012, 9), (0.007, 4), (0.005, 100), (0.005, 6), (0.005, 24)]
                else:
                    if context.get_value('k') <= 36.0:
                        if context.get_value('n') <= 68.0:
                            return [(0.097, 184), (0.097, 56), (0.086, 186), (0.086, 183), (0.086, 188), (0.086, 58), (0.086, 60), (0.065, 54), (0.043, 187), (0.043, 185), (0.043, 57), (0.043, 61), (0.032, 55), (0.032, 130), (0.032, 59), (0.011, 181), (0.011, 163), (0.011, 136), (0.011, 138)]
                        else:
                            return [(0.117, 184), (0.117, 170), (0.117, 169), (0.107, 183), (0.106, 188), (0.075, 181), (0.064, 130), (0.064, 56), (0.053, 171), (0.032, 57), (0.032, 59), (0.032, 185), (0.011, 163), (0.011, 32), (0.011, 37), (0.011, 34), (0.011, 33), (0.011, 35), (0.011, 36), (0.011, 54)]
                    else:
                        if context.get_value('mat2_stride_0') <= 384.0:
                            return [(0.244, 0), (0.061, 76), (0.061, 79), (0.030, 3), (0.030, 183), (0.030, 189), (0.030, 187), (0.030, 64), (0.030, 190), (0.030, 62), (0.030, 198), (0.030, 201), (0.030, 77), (0.030, 200), (0.030, 80), (0.030, 199), (0.030, 78), (0.030, 184), (0.020, 86), (0.020, 84), (0.020, 120), (0.020, 81), (0.020, 121), (0.020, 85), (0.020, 122), (0.010, 83), (0.010, 118), (0.010, 119), (0.010, 82)]
                        else:
                            return [(0.274, 83), (0.171, 86), (0.152, 0), (0.071, 85), (0.061, 125), (0.050, 84), (0.020, 109), (0.020, 117), (0.020, 81), (0.020, 118), (0.020, 121), (0.020, 108), (0.020, 115), (0.020, 116), (0.010, 110), (0.010, 120), (0.010, 103), (0.010, 107), (0.010, 119), (0.010, 122)]
        else:
            if context.get_value('arith_intensity') <= 56.995582580566406:
                if context.get_value('n') <= 68.0:
                    if context.get_value('k*n') <= 4448.0:
                        if context.get_value('m*n') <= 29626368.0:
                            return [(0.107, 198), (0.107, 200), (0.107, 201), (0.107, 199), (0.106, 76), (0.106, 79), (0.064, 197), (0.063, 56), (0.043, 184), (0.043, 187), (0.042, 80), (0.042, 77), (0.042, 183), (0.021, 78)]
                        else:
                            return [(0.073, 201), (0.073, 198), (0.073, 200), (0.073, 199), (0.073, 197), (0.073, 56), (0.073, 58), (0.073, 79), (0.073, 76), (0.072, 59), (0.072, 78), (0.072, 77), (0.072, 80), (0.018, 184), (0.018, 55), (0.018, 54)]
                    else:
                        if context.get_value('k') <= 348.0:
                            return [(0.206, 76), (0.183, 77), (0.169, 198), (0.160, 199), (0.053, 59), (0.046, 56), (0.038, 3), (0.030, 148), (0.030, 58), (0.030, 187), (0.023, 184), (0.015, 0), (0.008, 55), (0.008, 54)]
                        else:
                            return [(0.146, 198), (0.145, 199), (0.145, 148), (0.126, 0), (0.084, 76), (0.084, 77), (0.042, 80), (0.042, 79), (0.021, 149), (0.021, 150), (0.021, 3), (0.014, 46), (0.014, 74), (0.014, 75), (0.014, 124), (0.014, 194), (0.014, 195), (0.007, 145), (0.007, 146), (0.007, 2), (0.007, 72), (0.007, 147), (0.007, 71)]
                else:
                    if context.get_value('m') <= 3264.0:
                        return [(0.247, 147), (0.115, 197), (0.066, 199), (0.066, 201), (0.066, 198), (0.049, 0), (0.049, 169), (0.049, 171), (0.033, 140), (0.033, 125), (0.033, 114), (0.016, 126), (0.016, 183), (0.016, 184), (0.016, 185), (0.016, 182), (0.016, 188), (0.016, 78), (0.016, 148), (0.016, 138), (0.016, 77), (0.016, 56), (0.016, 59)]
                    else:
                        if context.get_value('k') <= 62.5:
                            return [(0.226, 190), (0.226, 189), (0.122, 62), (0.122, 64), (0.055, 77), (0.055, 78), (0.037, 198), (0.036, 201), (0.036, 33), (0.024, 163), (0.018, 56), (0.018, 35), (0.018, 169), (0.006, 171)]
                        else:
                            return [(0.162, 35), (0.118, 33), (0.096, 189), (0.096, 190), (0.088, 169), (0.074, 62), (0.073, 56), (0.066, 171), (0.051, 198), (0.051, 201), (0.044, 59), (0.037, 64), (0.029, 63), (0.007, 0), (0.007, 77)]
            else:
                if context.get_value('m*n') <= 1097728.0:
                    return [(0.403, 0), (0.179, 141), (0.134, 150), (0.086, 147), (0.051, 148), (0.048, 3), (0.024, 189), (0.020, 199), (0.017, 64), (0.010, 65), (0.010, 77), (0.007, 114), (0.003, 138), (0.003, 59), (0.003, 182)]
                else:
                    if context.get_value('m*n') <= 3244032.0:
                        return [(0.295, 189), (0.176, 64), (0.157, 65), (0.090, 0), (0.069, 62), (0.059, 63), (0.046, 77), (0.039, 169), (0.023, 199), (0.020, 35), (0.013, 33), (0.010, 171), (0.003, 141)]
                    else:
                        if context.get_value('n') <= 136.0:
                            return [(0.197, 189), (0.197, 63), (0.161, 77), (0.157, 62), (0.061, 33), (0.044, 65), (0.039, 35), (0.039, 64), (0.030, 169), (0.026, 0), (0.017, 199), (0.017, 148), (0.009, 56), (0.004, 3)]
                        else:
                            return [(0.460, 0), (0.145, 62), (0.138, 63), (0.081, 35), (0.047, 33), (0.043, 189), (0.023, 64), (0.018, 77), (0.013, 169), (0.009, 65), (0.009, 56), (0.005, 32), (0.005, 59), (0.002, 183), (0.002, 163)]
