# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side state machine for faithful CompressLM generation."""

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.request import RequestStatus

from .state import CODE, CONTENT, REPLAY, CompressLMBlock, CompressLMState

if TYPE_CHECKING:
    from vllm.config.compresslm import CompressLMConfig
    from vllm.v1.engine import SpanRemovalResult
    from vllm.v1.request import Request


MaskSpan = Callable[..., "SpanRemovalResult"]   # (request_id, start, end, groups=None)
PackCodeSpan = Callable[[str, int, int], None]


class CompressLMProcessor:
    def __init__(self, config: "CompressLMConfig") -> None:
        if not config._initialized:
            raise ValueError("CompressLMConfig token IDs are not initialized")
        self.config = config
        # Physical per-layer quotas: the code quota of every KV cache group (set by the
        # scheduler from the KV cache config). Groups with quota q < K physically drop
        # code slots [q, K) of a block when its content is compacted.
        self.group_quotas: tuple[int, ...] | None = None

    def set_group_quotas(self, group_quotas: Sequence[int] | None) -> None:
        if group_quotas is None:
            self.group_quotas = None
            return
        K = self.config.num_code_tokens
        for quota in group_quotas:
            if not 0 < quota <= K:
                raise ValueError(f"KV cache group quota {quota} outside [1, {K}]")
        self.group_quotas = tuple(int(q) for q in group_quotas)

    def prepare_request(self, request: "Request") -> CompressLMState:
        if request.prompt_token_ids is None:
            raise ValueError("CompressLM decoding requires prompt token IDs")
        if request.prompt_embeds is not None:
            raise ValueError("CompressLM decoding does not support prompt_embeds")
        if request.pooling_params is not None:
            raise ValueError("CompressLM decoding only supports generation requests")
        sampling = request.sampling_params
        if sampling is None:
            raise ValueError("CompressLM decoding requires sampling parameters")
        if getattr(sampling, "frequency_penalty", 0.0) != 0.0:
            raise ValueError("CompressLM decoding requires frequency_penalty=0")
        if getattr(sampling, "presence_penalty", 0.0) != 0.0:
            raise ValueError("CompressLM decoding requires presence_penalty=0")
        if getattr(sampling, "repetition_penalty", 1.0) != 1.0:
            raise ValueError("CompressLM decoding requires repetition_penalty=1")
        if getattr(sampling, "stop", None):
            raise ValueError(
                "CompressLM decoding currently supports token/EOS stops, not stop strings"
            )
        if getattr(sampling, "min_tokens", 0) != 0:
            raise ValueError("CompressLM decoding currently requires min_tokens=0")
        if (
            getattr(sampling, "logprobs", None) is not None
            and not self.config.debug
        ):
            raise ValueError(
                "CompressLM output logprobs are only available in debug mode"
            )
        if getattr(sampling, "prompt_logprobs", None) is not None:
            raise ValueError("CompressLM decoding does not support prompt logprobs")
        if request.use_structured_output:
            raise ValueError("CompressLM decoding does not support structured output")

        banned_singletons = {
            token_ids[0]
            for token_ids in (sampling.bad_words_token_ids or [])
            if len(token_ids) == 1
        }
        missing_bans = set(self.config.code_token_ids) - banned_singletons
        if missing_bans:
            raise ValueError(
                "CompressLM code tokens must be passed as bad_words so internal "
                f"IDs cannot be sampled; missing {len(missing_bans)} IDs"
            )

        prompt = list(request.prompt_token_ids)
        if not prompt:
            raise ValueError("CompressLM decoding requires a non-empty prompt")

        first = prompt[: self.config.block_size]
        state = CompressLMState(
            original_prompt_token_ids=prompt,
            pending_prompt_index=len(first),
            blocks=[CompressLMBlock(block_id=0)],
        )

        request.prompt_token_ids = first
        request.num_prompt_tokens = len(first)
        request._all_token_ids.clear()
        request._all_token_ids.extend(first)
        request._output_token_ids.clear()
        self._record_content(state, 0, first, is_prompt=True)
        return state

    def is_code_ready(self, request: "Request") -> bool:
        state = request.compresslm_state
        if state is None or request.num_computed_tokens >= request.num_tokens:
            return False
        return state.roles[request.num_computed_tokens] == CODE

    def num_code_rows_pending(self, request: "Request") -> int:
        """Length of the CODE run a code-ready request schedules as one unit (0 if it is not code-ready)."""
        if not self.is_code_ready(request):
            return 0
        roles, start = request.compresslm_state.roles, request.num_computed_tokens
        end = start + 1
        while end < request.num_tokens and roles[end] == CODE:
            end += 1
        return end - start

    def limit_num_new_tokens(
        self,
        request: "Request",
        proposed: int,
    ) -> int:
        state = request.compresslm_state
        if state is None or proposed <= 0:
            return proposed
        start = request.num_computed_tokens
        if start >= request.num_tokens:
            return proposed

        role = state.roles[start]
        end = start + 1
        while end < request.num_tokens and state.roles[end] == role:
            end += 1
        remaining = end - start

        if role == CODE:
            return remaining if proposed >= remaining else 0
        limit = min(proposed, remaining)
        if role == CONTENT:
            for position in range(start, start + limit):
                if (
                    position in state.content_to_expired_replay
                    or position in state.content_to_expired_retained_tail
                ):
                    return position - start + 1
        return limit

    def should_skip_sampling(
        self,
        request: "Request",
        num_scheduled_tokens: int,
    ) -> bool:
        state = request.compresslm_state
        if state is None:
            return False
        start = request.num_computed_tokens
        end = start + num_scheduled_tokens
        role = state.roles[start]
        if role != CONTENT:
            return True
        if end < request.num_tokens:
            return True
        return state.pending_prompt_index < len(state.original_prompt_token_ids)

    def update_after_step(
        self,
        request: "Request",
        num_scheduled_tokens: int,
        sampled_token_ids: list[int],
        mask_span: MaskSpan,
        max_model_len: int,
        pack_code_span: PackCodeSpan | None = None,
    ) -> tuple[list[int], bool]:
        state = request.compresslm_state
        if state is None:
            return sampled_token_ids, False
        if len(sampled_token_ids) > 1:
            raise RuntimeError(
                "CompressLM decoding is incompatible with speculative multi-token output"
            )

        end = request.num_computed_tokens
        start = end - num_scheduled_tokens
        role = state.roles[start]
        if any(state.roles[pos] != role for pos in range(start, end)):
            raise RuntimeError("CompressLM scheduled a mixed-role model step")

        if role == CONTENT:
            self._expire_replay_after_content(state, request, start, end, mask_span)
            self._expire_retained_tail_after_content(
                state, request, start, end, mask_span
            )

        sampled = sampled_token_ids[0] if sampled_token_ids else None
        stopped = False
        visible: list[int] = []
        if sampled is not None:
            state.visible_output_token_ids.append(sampled)
            visible.append(sampled)
            stopped = self._check_stop(request, state, sampled, max_model_len)
            if stopped:
                return visible, True

        block = state.blocks[state.position_to_block[start]]
        role_end = end == request.num_tokens or state.roles[end] != role

        if role == CONTENT:
            if len(block.content_positions) == self.config.block_size and (
                end == block.content_positions[-1] + 1
            ):
                if self.config.retain_tail_policy == "sticky":
                    # The prior tail is a content-only bridge. Remove it
                    # after the final content query and before codes run.
                    # This must also run during preemption re-prefill, when
                    # the existing code_span prevents restaging the codes.
                    self._expire_sticky_retained_tail_at_boundary(
                        request, state, block, mask_span
                    )
                if block.code_span is None:
                    internal_tokens = (
                        self.config.num_code_tokens + self.config.replay_tail
                    )
                    if end + internal_tokens >= max_model_len:
                        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                        return visible, True
                    state.held_sampled_token = sampled
                    self._stage_codes(request, state, block)
            elif sampled is not None:
                self._append_content(request, state, [sampled], is_prompt=False)

        elif role == CODE and role_end:
            if not block.replay_positions:
                self._stage_replay(request, state, block)
                if not block.replay_positions:
                    self._finish_block(
                        request,
                        state,
                        block,
                        mask_span,
                        pack_code_span,
                    )

        elif role == REPLAY and role_end:
            self._finish_block(
                request,
                state,
                block,
                mask_span,
                pack_code_span,
            )

        return visible, False

    def reset_for_preemption(self, state: CompressLMState) -> None:
        state.reset_compaction_progress()

    def code_excluded_logical_range(
        self,
        request: "Request",
    ) -> tuple[int, int] | None:
        if not self.config.code_excluded_tail:
            return None
        state = request.compresslm_state
        if state is None:
            raise RuntimeError("CompressLM request has no state")
        block = state.current_block
        if block.code_excluded_content_span is None:
            raise RuntimeError("tail-exclusive code block has no source span")
        return block.code_excluded_content_span

    def code_physical_spans(
        self,
        request: "Request",
        group: int = 0,
        logical_limit: int | None = None,
    ) -> list[tuple[int, int]]:
        """Physical [start, end) of every staged code span (block order) after KV compaction.

        Row k of a span holds code slot k, so a per-layer quota q hides [start + q, end). Includes the
        current block's span during its own code step (codes never see their block's slots >= q either).
        With physical per-layer quotas (group_quotas set) only spans still holding all K rows in `group`'s
        layout are returned: compacted blocks already had their slots >= quota removed, so nothing is left to hide.

        `logical_limit` is the first logical position NOT yet in this request's KV, i.e. num_computed_tokens +
        num_scheduled_tokens. Blocks whose codes sit beyond it are skipped. This matters only after a PREEMPTION:
        the scheduler frees the KV and restarts the request at num_computed_tokens = 0 while deliberately keeping
        the logical structure (state.blocks and their code_span) so re-prefill replays the identical token stream
        without restaging codes. Without the bound we would emit spans for blocks the re-prefill has not reached
        yet, at their original high positions, and the worker rejects them against the rebuilt (short) seq_len.
        """
        state = request.compresslm_state
        if state is None:
            raise RuntimeError("CompressLM request has no state")
        K = self.config.num_code_tokens
        spans: list[tuple[int, int]] = []
        for block in state.blocks:
            if block.code_span is None:
                continue
            if self.group_quotas is not None and block.block_id in state.compacted_block_ids:
                continue
            start, end = block.code_span
            if logical_limit is not None and end > logical_limit:
                continue    # not re-prefilled yet after a preemption; its codes are not in the KV
            if end - start != K:
                raise RuntimeError(f"code span {block.code_span} does not hold {K} rows")
            physical_start = request.logical_to_physical(start, group)
            physical_end = request.logical_to_physical(end - 1, group) + 1
            if physical_end - physical_start != K:
                raise RuntimeError("code span is not contiguous after KV compaction")
            spans.append((physical_start, physical_end))
        return spans

    def _record_content(
        self,
        state: CompressLMState,
        start: int,
        token_ids: list[int],
        is_prompt: bool,
    ) -> None:
        block = state.current_block
        previous = state.blocks[block.block_id - 1] if block.block_id else None
        for offset, token_id in enumerate(token_ids):
            position = start + offset
            within = len(block.content_positions)
            block.content_positions.append(position)
            block.content_token_ids.append(token_id)
            if is_prompt:
                block.prompt_content_positions.append(position)
            else:
                block.response_content_positions.append(position)
            state.roles.append(CONTENT)
            state.position_to_block.append(block.block_id)
            if (
                previous is not None
                and within < len(previous.replay_positions)
            ):
                state.content_to_expired_replay[position] = (
                    previous.replay_positions[within]
                )
            if (
                self.config.retain_tail_policy == "rolling"
                and previous is not None
                and within < len(previous.retained_tail_positions)
            ):
                retained_position = previous.retained_tail_positions[within]
                if retained_position in previous.response_content_positions:
                    state.content_to_expired_retained_tail[position] = (
                        retained_position
                    )

    def _append_content(
        self,
        request: "Request",
        state: CompressLMState,
        token_ids: list[int],
        is_prompt: bool,
    ) -> None:
        if not token_ids:
            return
        start = request.num_tokens
        request.append_output_token_ids(token_ids)
        self._record_content(state, start, token_ids, is_prompt)

    def _stage_codes(
        self,
        request: "Request",
        state: CompressLMState,
        block: CompressLMBlock,
    ) -> None:
        start = request.num_tokens
        code_ids = self.config.code_token_ids
        request.append_output_token_ids(code_ids)
        end = request.num_tokens
        block.code_span = (start, end)
        if self.config.retain_tail_kv:
            block.retained_tail_positions = block.content_positions[
                -self.config.retain_tail_kv :
            ]
        if self.config.code_excluded_tail:
            tail_positions = block.content_positions[
                -self.config.code_excluded_tail :
            ]
            tail_start, tail_end = tail_positions[0], tail_positions[-1] + 1
            if tail_positions != list(range(tail_start, tail_end)):
                raise RuntimeError("exact tail source positions must be contiguous")
            block.code_excluded_content_span = (tail_start, tail_end)
        state.roles.extend([CODE] * len(code_ids))
        state.position_to_block.extend([block.block_id] * len(code_ids))
        state.worker_token_ids_dirty = True
        if self.config.debug:
            print(
                f"[CompressLM] request={request.request_id} block={block.block_id} "
                f"codes=[{start},{end})",
                flush=True,
            )

    def _stage_replay(
        self,
        request: "Request",
        state: CompressLMState,
        block: CompressLMBlock,
    ) -> None:
        tail = self.config.replay_tail
        if not tail:
            return
        replay_ids = block.content_token_ids[-tail:]
        start = request.num_tokens
        request.append_output_token_ids(replay_ids)
        block.replay_positions = list(range(start, start + len(replay_ids)))
        state.roles.extend([REPLAY] * len(replay_ids))
        state.position_to_block.extend([block.block_id] * len(replay_ids))
        state.worker_token_ids_dirty = True

    def _finish_block(
        self,
        request: "Request",
        state: CompressLMState,
        block: CompressLMBlock,
        mask_span: MaskSpan,
        pack_code_span: PackCodeSpan | None,
    ) -> None:
        if self.config.retain_tail_policy == "sticky":
            self._expire_sticky_retained_tail_at_boundary(
                request, state, block, mask_span
            )
        self._compact_response_content(
            request,
            state,
            block,
            mask_span,
            pack_code_span,
        )

        # During preemption recovery, later logical tokens already exist.
        replay_end = (
            block.replay_positions[-1] + 1
            if block.replay_positions
            else block.code_span[1]
        )
        if replay_end < request.num_tokens:
            return

        next_block = CompressLMBlock(block_id=len(state.blocks))
        state.blocks.append(next_block)
        state.current_block_id = next_block.block_id

        if state.pending_prompt_index < len(state.original_prompt_token_ids):
            remaining = state.original_prompt_token_ids[state.pending_prompt_index :]
            chunk = remaining[: self.config.block_size]
            state.pending_prompt_index += len(chunk)
            self._append_content(request, state, chunk, is_prompt=True)
            state.worker_token_ids_dirty = True
            return

        if state.held_sampled_token is None:
            raise RuntimeError("CompressLM block completed without continuation token")
        held = state.held_sampled_token
        state.held_sampled_token = None
        self._append_content(request, state, [held], is_prompt=False)
        state.worker_token_ids_dirty = True

    def _compact_response_content(
        self,
        request: "Request",
        state: CompressLMState,
        block: CompressLMBlock,
        mask_span: MaskSpan,
        pack_code_span: PackCodeSpan | None,
    ) -> None:
        if block.block_id in state.compacted_block_ids:
            return
        if self.config.uses_selective_history:
            state.compacted_block_ids.add(block.block_id)
            return
        positions = block.response_content_positions
        if self.config.retain_tail_kv:
            retained = set(block.retained_tail_positions)
            positions = [position for position in positions if position not in retained]
        if self.config.uses_packed_kv:
            if pack_code_span is None:
                raise RuntimeError("packed KV decoding has no code capture callback")
            if block.code_span is None:
                raise RuntimeError("packed KV block has no completed code span")
            code_start, code_end = block.code_span
            pack_code_span(request.request_id, code_start, code_end)
            if positions and positions != list(range(positions[0], positions[-1] + 1)):
                raise RuntimeError("packed KV response content must be contiguous")
            if self.config.retain_tail_kv:
                # Static tail: the block's final retain_tail_kv content rows stay in the paged cache for the rest
                # of the request (visible to every later query, like the codes in the sidecar). Remove the
                # summarized response content and the code rows as two spans around it (the scheduler queues
                # the two compactions in order).
                if self.config.retain_tail_policy != "static":
                    raise RuntimeError("packed KV retained tails must use the static policy")
                if positions:
                    self._mask_or_raise(
                        mask_span,
                        request.request_id,
                        positions[0],
                        positions[-1] + 1,
                    )
                self._mask_or_raise(
                    mask_span,
                    request.request_id,
                    code_start,
                    code_end,
                )
                state.compacted_block_ids.add(block.block_id)
                return
            removal_start = positions[0] if positions else code_start
            if positions and positions[-1] + 1 != code_start:
                raise RuntimeError(
                    "packed KV response content and code span are not contiguous"
                )
            self._mask_or_raise(
                mask_span,
                request.request_id,
                removal_start,
                code_end,
            )
            state.compacted_block_ids.add(block.block_id)
            return
        if positions:
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise RuntimeError("CompressLM response content must be contiguous")
            self._mask_or_raise(
                mask_span,
                request.request_id,
                positions[0],
                positions[-1] + 1,
            )
        if self.group_quotas is not None:
            # Physical per-layer quotas: every KV cache group whose quota q < K drops the block's
            # code slots [q, K) from its own layout (the nested-prefix layout makes slots >= q
            # invisible to that group's layers forever). Done after the content removal so each
            # group's second removal is computed on the layout its first one left.
            if block.code_span is None:
                raise RuntimeError("physical per-layer quotas need a completed code span")
            code_start, code_end = block.code_span
            K = self.config.num_code_tokens
            for quota in sorted(set(self.group_quotas)):
                if quota >= K:
                    continue
                groups = [g for g, q in enumerate(self.group_quotas) if q == quota]
                self._mask_or_raise(
                    mask_span,
                    request.request_id,
                    code_start + quota,
                    code_end,
                    groups=groups,
                )
        state.compacted_block_ids.add(block.block_id)

    def _expire_sticky_retained_tail_at_boundary(
        self,
        request: "Request",
        state: CompressLMState,
        block: CompressLMBlock,
        mask_span: MaskSpan,
    ) -> None:
        if not block.block_id:
            return
        previous = state.blocks[block.block_id - 1]
        if previous.block_id in state.expired_sticky_retained_block_ids:
            return
        if self.config.uses_selective_history:
            state.expired_sticky_retained_block_ids.add(previous.block_id)
            return

        response_positions = set(previous.response_content_positions)
        positions = [
            position
            for position in previous.retained_tail_positions
            if position in response_positions
        ]
        if positions:
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise RuntimeError(
                    "sticky retained response tail must be contiguous"
                )
            self._mask_or_raise(
                mask_span,
                request.request_id,
                positions[0],
                positions[-1] + 1,
            )
        state.expired_sticky_retained_block_ids.add(previous.block_id)

    def _expire_replay_after_content(
        self,
        state: CompressLMState,
        request: "Request",
        start: int,
        end: int,
        mask_span: MaskSpan,
    ) -> None:
        for position in range(start, end):
            replay_position = state.content_to_expired_replay.get(position)
            if (
                replay_position is None
                or replay_position in state.expired_replay_positions
            ):
                continue
            if self.config.uses_selective_history:
                state.expired_replay_positions.add(replay_position)
                continue
            self._mask_or_raise(
                mask_span,
                request.request_id,
                replay_position,
                replay_position + 1,
            )
            state.expired_replay_positions.add(replay_position)

    def _expire_retained_tail_after_content(
        self,
        state: CompressLMState,
        request: "Request",
        start: int,
        end: int,
        mask_span: MaskSpan,
    ) -> None:
        for position in range(start, end):
            retained_position = state.content_to_expired_retained_tail.get(
                position
            )
            if (
                retained_position is None
                or retained_position in state.expired_retained_tail_positions
            ):
                continue
            if self.config.uses_selective_history:
                state.expired_retained_tail_positions.add(retained_position)
                continue
            self._mask_or_raise(
                mask_span,
                request.request_id,
                retained_position,
                retained_position + 1,
            )
            state.expired_retained_tail_positions.add(retained_position)

    @staticmethod
    def _mask_or_raise(
        mask_span: MaskSpan,
        request_id: str,
        start: int,
        end: int,
        groups: Sequence[int] | None = None,
    ) -> None:
        result = mask_span(request_id, start, end) if groups is None else mask_span(request_id, start, end, groups=groups)
        if not result.success:
            raise RuntimeError(
                f"CompressLM KV compaction failed for [{start}, {end}): "
                f"{result.error_message}"
            )

    @staticmethod
    def _check_stop(
        request: "Request",
        state: CompressLMState,
        token_id: int,
        max_model_len: int,
    ) -> bool:
        sampling = request.sampling_params
        assert sampling is not None
        visible_count = len(state.visible_output_token_ids)
        if visible_count < sampling.min_tokens:
            return False
        if not sampling.ignore_eos and token_id == request.eos_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        if token_id in (sampling.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = token_id
            return True
        if request.num_tokens >= max_model_len or visible_count >= request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        return False
