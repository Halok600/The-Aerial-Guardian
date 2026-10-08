# Tracking, HOTA per sequence (VisDrone2019-MOT-test-dev)

Same run as [mot_table.md](mot_table.md) (`outputs/eval/mot/mot_per_seq.md`).

**person** (HOTA per sequence)

| Sequence | Ultralytics ByteTrack (default config) | Ultralytics ByteTrack (our thresholds) | Ours (no camera-motion comp.) | Ours + ECC camera-motion comp. |
|---|---|---|---|---|
| uav0000009_03358_v | 21.0 | 20.5 | 20.5 | 35.0 |
| uav0000073_00600_v | 41.9 | 42.7 | 42.7 | 49.6 |
| uav0000073_04464_v | 43.6 | 44.1 | 44.2 | 44.6 |
| uav0000077_00720_v | 29.2 | 28.8 | 29.5 | 40.1 |
| uav0000088_00290_v | 28.9 | 30.2 | 31.0 | 42.5 |
| uav0000119_02301_v | 31.8 | 34.3 | 34.3 | 34.3 |
| uav0000120_04775_v | 32.7 | 33.6 | 34.5 | 38.9 |
| uav0000161_00000_v | 32.2 | 26.0 | 25.1 | 25.2 |
| uav0000188_00000_v | 18.2 | 16.0 | 16.1 | 17.5 |
| uav0000201_00000_v | 4.3 | 4.3 | 6.1 | 13.6 |
| uav0000249_00001_v | 24.7 | 24.8 | 25.9 | 31.3 |
| uav0000249_02688_v | 17.6 | 23.2 | 23.6 | 25.8 |
| uav0000297_00000_v | 30.3 | 33.3 | 33.7 | 39.6 |
| uav0000297_02761_v | 34.8 | 43.0 | 43.4 | 43.4 |
| uav0000306_00230_v | 18.2 | 20.7 | 20.9 | 28.2 |
| uav0000355_00001_v | 28.6 | 38.8 | 37.6 | 41.2 |
| uav0000370_00001_v | n/a | n/a | n/a | n/a |

n/a: the sequence has no ground-truth objects of this group

**vehicle** (HOTA per sequence)

| Sequence | Ultralytics ByteTrack (default config) | Ultralytics ByteTrack (our thresholds) | Ours (no camera-motion comp.) | Ours + ECC camera-motion comp. |
|---|---|---|---|---|
| uav0000009_03358_v | 47.0 | 47.7 | 48.1 | 66.5 |
| uav0000073_00600_v | 38.4 | 0.0 | 0.0 | 0.0 |
| uav0000073_04464_v | 30.9 | 42.5 | 53.0 | 53.0 |
| uav0000077_00720_v | 49.6 | 50.5 | 50.9 | 62.1 |
| uav0000088_00290_v | n/a | n/a | n/a | n/a |
| uav0000119_02301_v | 63.6 | 67.6 | 67.4 | 67.4 |
| uav0000120_04775_v | 58.6 | 59.6 | 60.6 | 65.9 |
| uav0000161_00000_v | 71.3 | 72.6 | 73.0 | 73.2 |
| uav0000188_00000_v | 55.5 | 56.2 | 55.8 | 59.9 |
| uav0000201_00000_v | 41.1 | 42.9 | 43.2 | 49.8 |
| uav0000249_00001_v | 50.5 | 51.3 | 51.0 | 61.8 |
| uav0000249_02688_v | 43.8 | 45.2 | 45.0 | 57.2 |
| uav0000297_00000_v | 52.1 | 52.8 | 52.7 | 63.2 |
| uav0000297_02761_v | 60.2 | 64.2 | 64.3 | 64.4 |
| uav0000306_00230_v | 51.4 | 54.2 | 54.2 | 57.6 |
| uav0000355_00001_v | 49.8 | 55.3 | 55.9 | 56.2 |
| uav0000370_00001_v | 37.1 | 38.1 | 38.2 | 39.1 |

n/a: the sequence has no ground-truth objects of this group
