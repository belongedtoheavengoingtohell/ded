# FirstLight Core 🧠

A small, easy-to-read piece of **FirstLight Zero**: a Clash Royale AI that taught itself by playing against itself, with no human games to copy.
This folder keeps only the important parts. Everything runs **offline**.

## The three parts (think of a robot learning a game)

| Part | Where | What it does |
|---|---|---|
| 👀 **The eyes** | `firstlight_core/spec.py` | Lists *what the AI can see*: troops and where they are, elixir, towers, cards in hand. |
| 🧠 **The brain** | `firstlight_core/model.py` | A small neural network (1.17 million numbers). It looks at the board and picks: **wait**, or **play a card here**. |
| 🏋️ **The practice gym** | `firstlight_core/ppo.py` | How it improves. After each game it asks "which moves worked?", then nudges the brain to do more of those. This is called **PPO**. |
| 🏅 **The points** | `firstlight_core/rewards.py` | What counts as "good": +1 for winning, -1 for losing, bonuses for tower health, a penalty for wasting elixir. |
| 📼 **Game recordings** | `firstlight_core/store.py` | The file format used to save a game so the brain can study it. |

## What's included

- `checkpoints/firstlight-zero-v1450.pt`: the trained brain (the authors' best version).
- `demo_brain.py`: shows the brain a **made-up** board and prints what it would do.
- `LICENSE` and `NOTICE`: Apache-2.0. Please keep them if you share this.

## Try it

You need Python 3.12 and `pip install -r requirements.txt`.

```
python demo_brain.py
```

It prints the brain's guess of how likely it is to win, and the move it picks. The board is invented, so the move is only a demo of how the brain "talks".

## What is NOT here, on purpose

- **The game engine and the eyes' wiring** (`native_runner/` and `features.py` in the original). They read real battles from a special offline copy of the game that you must build yourself from your own game files.
- **Anything that connects to the real game or plays the official ladder.** The original project doesn't do this either, and neither does this folder.

So this folder is for **learning and experimenting**: reading the code, running the brain on fake boards, and studying how self-play training works.

## Ideas to improve it (good for a school project)

1. **Make the points smarter.** Edit `rewards.py` and think about what else should count as "good play".
2. **Add a "why" printout.** Show which card the brain wanted most and how sure it was (the option scores inside `model.py`).
3. **Compare brain versions.** The original repo has checkpoints every 50 rounds. See how a young brain (v100) differs from an old one (v1450) on the same fake board.
4. **Try a bigger or smaller brain.** Change `channels` in `XbowNet(...)` and see what breaks.

## Honest notes

- This brain only knows **one fixed deck** (the X-Bow mirror) at one card level. It can't play other decks.
- I could not run PyTorch in my workspace, so I checked the code for syntax errors and confirmed the checkpoint's layer names, but I have **not** run `demo_brain.py` myself. If it errors, send me the message and I'll fix it.
- Credit: all the real work is by the FirstLight Zero author, https://github.com/Jaasssoooonnnnn/FirstLight-Zero (Apache-2.0). This is a trimmed copy.
