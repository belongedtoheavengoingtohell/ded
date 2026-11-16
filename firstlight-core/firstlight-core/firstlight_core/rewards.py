"""The "points" the brain is trained on, as plain maths (no game engine needed). From FirstLight Zero's reward.py.

The brain gets:
  * +1 for winning, -1 for losing at the end,
  * a small bonus/penalty as towers lose or keep hit points (`shaping`),
  * a penalty for wasting elixir by sitting at 10 and letting it overflow,
  * a penalty if its King tower gets woken up.
The tower and King parts need the live game state, so only the elixir-waste maths is reproduced here.
"""
import math

REWARD_DEFAULTS = dict(
    terminal_reward_scale=1., reward_gamma=.9997, shaping=.2,
    personal_elixir_overflow_penalty=.0225, unilateral_elixir_overflow_penalty=.075,
    personal_elixir_overflow_grace=4., elixir_overflow_step_penalty_cap=.1,
    king_activation_penalty=.1,
)


def newly_wasted_elixir(before_elixir, after_elixir, generated_elixir):
    """How much elixir was thrown away during one step because the bar was already full."""
    if any(not math.isfinite(v) or v < 0. for v in (before_elixir, after_elixir, generated_elixir)):
        raise ValueError('elixir inputs must be finite and nonnegative')
    if after_elixir < 10. - 1e-6:
        return 0.
    return generated_elixir if before_elixir >= 10. - 1e-6 else max(0., before_elixir + generated_elixir - 10.)


def overflow_penalty_step(personal_wasted, personal_accrued, unilateral_wasted, unilateral_accrued,
                          before_elixir, after_elixir, generated_elixir, both_full,
                          personal_coefficient=REWARD_DEFAULTS['personal_elixir_overflow_penalty'],
                          personal_grace=REWARD_DEFAULTS['personal_elixir_overflow_grace'],
                          unilateral_coefficient=REWARD_DEFAULTS['unilateral_elixir_overflow_penalty'],
                          step_cap=REWARD_DEFAULTS['elixir_overflow_step_penalty_cap']):
    """Advance the two "wasted elixir" clocks by one step. Returns the new state and the penalty paid this step:
    (personal_wasted, personal_accrued, unilateral_wasted, unilateral_accrued, personal_penalty, unilateral_penalty).

    personal    : you wasted a lot of elixir in total (after a free allowance of `personal_grace`).
    unilateral  : you are overflowing while your opponent is not (this grows faster, squared).
    """
    if after_elixir < 10. - 1e-6:
        return 0., 0., 0., 0., 0., 0.
    newly = newly_wasted_elixir(before_elixir, after_elixir, generated_elixir)
    next_personal = personal_wasted + newly
    personal_delta = max(0., personal_coefficient * max(0., next_personal - personal_grace) - personal_accrued)
    if both_full:
        next_uni, unilateral_accrued, uni_delta = 0., 0., 0.
    else:
        next_uni = unilateral_wasted + newly
        uni_delta = max(0., unilateral_coefficient * next_uni * next_uni - unilateral_accrued)
    combined = personal_delta + uni_delta
    scale = 1. if combined <= step_cap else step_cap / combined
    return (next_personal, personal_accrued + personal_delta * scale, next_uni,
            unilateral_accrued + uni_delta * scale, personal_delta * scale, uni_delta * scale)
