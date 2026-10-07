"""Single source of truth for the eval/train user-prompt format.

CRITICAL INVARIANT: the user-turn content fed to the model at EVAL must match
TRAINING byte-for-byte, or the SFT model is prompted out-of-distribution. The
data and eval scripts build the user turn through here so they can never drift.

The "boxed instruction" is OFF by default. Runs that want the instruction set it in
BOTH places:
  - training: data config `boxed_instruction: true`  (-> tokenization cache key changes)
  - eval:     env `BOXED_INSTRUCTION=1`
Keep them in lockstep; mismatch is a silent accuracy bug.
"""
import os

# The exact instruction string. Change in ONE place; both train and eval pick it up.
BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def format_user_content(problem: str, use_instruction: bool) -> str:
    """Build the user-turn content. use_instruction=False -> raw problem (training-default)."""
    if not use_instruction:
        return problem
    return f"{problem}\n\n{BOXED_INSTRUCTION}"


def boxed_instruction_enabled() -> bool:
    """Eval-side default: read the BOXED_INSTRUCTION env flag (0/1). Off by default."""
    return os.environ.get("BOXED_INSTRUCTION", "0") == "1"
