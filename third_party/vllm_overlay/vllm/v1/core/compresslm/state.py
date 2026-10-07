# SPDX-License-Identifier: Apache-2.0
"""Per-request state for fixed-block CompressLM decoding."""

from dataclasses import dataclass, field


CONTENT = 0
CODE = 1
REPLAY = 2


@dataclass
class CompressLMBlock:
    block_id: int
    content_positions: list[int] = field(default_factory=list)
    content_token_ids: list[int] = field(default_factory=list)
    prompt_content_positions: list[int] = field(default_factory=list)
    response_content_positions: list[int] = field(default_factory=list)
    code_span: tuple[int, int] | None = None
    code_excluded_content_span: tuple[int, int] | None = None
    replay_positions: list[int] = field(default_factory=list)
    retained_tail_positions: list[int] = field(default_factory=list)


@dataclass
class CompressLMState:
    original_prompt_token_ids: list[int]
    pending_prompt_index: int = 0
    roles: list[int] = field(default_factory=list)
    position_to_block: list[int] = field(default_factory=list)
    blocks: list[CompressLMBlock] = field(default_factory=list)
    current_block_id: int = 0
    held_sampled_token: int | None = None
    visible_output_token_ids: list[int] = field(default_factory=list)
    content_to_expired_replay: dict[int, int] = field(default_factory=dict)
    expired_replay_positions: set[int] = field(default_factory=set)
    content_to_expired_retained_tail: dict[int, int] = field(default_factory=dict)
    expired_retained_tail_positions: set[int] = field(default_factory=set)
    expired_sticky_retained_block_ids: set[int] = field(default_factory=set)
    compacted_block_ids: set[int] = field(default_factory=set)
    worker_token_ids_dirty: bool = False

    @property
    def current_block(self) -> CompressLMBlock:
        return self.blocks[self.current_block_id]

    def reset_compaction_progress(self) -> None:
        self.expired_replay_positions.clear()
        self.expired_retained_tail_positions.clear()
        self.expired_sticky_retained_block_ids.clear()
        self.compacted_block_ids.clear()
