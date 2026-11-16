"""Poke the brain with a FAKE board, offline. No game, no engine, no internet.

    python demo_brain.py

It loads the trained model (v1450), shows it a made-up battle situation, and prints what it would do.
The board here is invented, so the move is only a demo of how the brain "talks"; it is not real play.
"""
from pathlib import Path
import torch

from firstlight_core.model import XbowNet
from firstlight_core.spec import (CARD_FEATURES, MAX_UNITS, NO_CARD, OPP_CARD_FEATURES, OPTIONS, OPTION_FEATURES,
                                  SCALARS, UNIT_FEATURES, HEIGHT, WIDTH, CARD_NAMES, ABILITY_OPTION)

CHECKPOINT = Path(__file__).parent / 'checkpoints' / 'firstlight-zero-v1450.pt'


def load_brain():
    net = XbowNet(channels=64)                       # the published model uses 64 channels (see checkpoints README)
    net.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'), strict=True)
    return net.eval()


def fake_situation(n=1):
    """One made-up moment: 10 elixir, all four cards playable anywhere on our half of the board."""
    x = dict(
        scal=torch.zeros(n, SCALARS), cards=torch.zeros(n, 8, CARD_FEATURES), opp_cards=torch.zeros(n, 8, OPP_CARD_FEATURES),
        option_card=torch.tensor([[0, 4, 6, 7, NO_CARD]]).repeat(n, 1),     # X-Bow, Skeletons, Fireball, Log, (no ability)
        opt_feat=torch.rand(n, OPTIONS, OPTION_FEATURES),
        unit_type=torch.zeros(n, MAX_UNITS, dtype=torch.long), unit_feat=torch.zeros(n, MAX_UNITS, UNIT_FEATURES))
    x['scal'][:, 4] = 1.0                                                   # my_elixir_over_10 = full
    place = torch.zeros(n, 4, HEIGHT, WIDTH, dtype=torch.bool); place[:, :, HEIGHT // 2:, :] = True
    opt_mask = torch.tensor([[True, True, True, True, False]]).repeat(n, 1)
    return x, dict(opt_mask=opt_mask, place=place, force=torch.zeros(n, dtype=torch.bool))


if __name__ == '__main__':
    torch.manual_seed(0)
    net = load_brain()
    x, legal = fake_situation()
    state = net.initial_state(1)
    out, _ = net.act(x, legal, state, greedy=False, temperature=0.5)       # 0.5 is the temperature the authors recommend
    gate, option, cell = int(out['gate']), int(out['option']), int(out['cell'])
    print('How sure the brain thinks it is of winning from here (value):', round(float(out['value']), 3))
    if gate == 0:
        print('Decision: WAIT (do nothing this moment)')
    elif option == ABILITY_OPTION:
        print('Decision: use the Hero Knight ability')
    else:
        row, col = divmod(cell, WIDTH)
        print(f'Decision: play {CARD_NAMES[int(x["option_card"][0, option])]} at tile column {col}, row {row}')
