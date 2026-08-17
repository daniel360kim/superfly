# ATTEMPTS — ledger of what's been tried

Condensed record of approaches and their verdicts. Check before re-trying
anything. Verdicts: `REJECTED` / `ESTABLISHED-NEGATIVE` / `SHIPPED` /
`SUPERSEDED` / `PENDING`.

## 2026-08-17 — DiffAero planar checkpoints are unrecoverable — ESTABLISHED-NEGATIVE

`checkpoints/DiffAero/planar_{cnn,mlp,rcnn}_sr0.9*` were committed as
**symlinks** pointing at `/home/ubuntu/superfly/diffaero/outputs/train/2026-07-03/...`.
Only the symlink text ever entered git; the targets lived in a gitignored
Hydra output tree on a checkout that no longer exists. The weights are gone.
The three symlinks and the `diffaero_vel_planar` registry entry that pointed
at them are deleted. Any future planar run must be retrained from scratch
(`scripts/train_diffaero.py`, planar config) — do not go looking for these
files on other boxes; the 2026-07-03 output tree survives nowhere.
