#!/usr/bin/env python3
"""scripts/continue_train.py — Correctly continue training from a checkpoint.

BUG this replaces: scripts/02_train.py's --resume flag builds the model via
YOLO(MODEL_YAML) (a fresh, randomly-initialized graph) and then passes
model=<last.pt path> as a kwarg to .train(). Ultralytics does NOT use that
kwarg to load weights into an already-constructed YOLO instance — the model
is already bound at YOLO() construction time. Combined with pretrained=False
(forced whenever --resume is passed), this means --resume silently trains
from scratch. Verified by an absent "Transferred N/437 items from pretrained
weights" line in that run's log, and by its per-epoch metrics being
bit-identical to an earlier from-scratch run (same seed=42 +
deterministic=True reproduces identical trajectories from identical random
initialization).

v4 — TRUE resume, verified by reading ultralytics/engine/trainer.py directly
(installed package, 8.0.236) rather than assuming:

  YOLO(checkpoint_path).train(resume=True)

is the correct call. Model.train() (engine/model.py:348) rewrites the
resume kwarg to the checkpoint path and skips manually rebuilding the model
graph when resuming. BaseTrainer.check_resume() (engine/trainer.py:584)
then REBUILDS self.args entirely from the checkpoint's own embedded train
args (only imgsz/batch survive from caller overrides — NOT workers, NOT any
other kwarg), which is why the startup banner printed misleading
resume=False/workers=2 in an earlier attempt: that banner reflects the old
checkpoint's stored args, not the live resume state. The actual behavioral
flag (self.resume) is set correctly and independently on trainer.py:607,
and resume_training() (line 609) genuinely restores epoch count, optimizer
(AdamW) momentum, and EMA state from the checkpoint. That earlier attempt
was killed before reaching resume_training()'s log line, which is why it
looked broken but very likely wasn't.

Because workers can't be overridden through this path, OOM mitigation for
resumed runs has to happen some other way — closing background apps /
monitoring free RAM — not via a workers= kwarg here.

No other kwargs are passed: check_resume() would discard them anyway, and
the checkpoint already has the right data/epochs/imgsz/batch/augmentation
config embedded from when it was first trained.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "aerial_guardian_p2_v2" / "weights" / "last.pt"

if __name__ == "__main__":
    print(f"Resuming from: {SOURCE_WEIGHTS}")
    model = YOLO(str(SOURCE_WEIGHTS))
    model.train(resume=True)
