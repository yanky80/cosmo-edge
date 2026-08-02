# RK3588 preview benchmark

This is a reproducible capacity report, not an FPS acceptance gate. It uses
one fixed board, model, and video; rerun the command below after changing any
of them and publish a new report instead of comparing unlike runs.

## Fixed inputs

| Item | Value |
| --- | --- |
| Board | BOZZ SW3588-MB-V10-EC11-B0 |
| Kernel | Linux 6.1.118, aarch64 |
| Model | `/home/YTHC/convert/yolo26n.rknn` (YOLO26, six INT8 raw-head outputs) |
| Video | `/home/YTHC/cosmo-edge-issue10/Safety Helmet.mp4` (H.264, 1920x1080, 24 FPS, 15.018667 s) |
| Run | 100 decoded frames, warm-up included, no host-frame check |

The public repository contains the schema and summary only. Board-local model
and video paths are intentionally not copied into the repository.

## Reproduction

Build the standalone acceptance smoke on the board using the command in
[`test/rk3588/detector_task_smoke.cc`](../../../test/rk3588/detector_task_smoke.cc),
then run:

```sh
./detector_task_smoke \
  --model /home/YTHC/convert/yolo26n.rknn \
  --h264 '/home/YTHC/cosmo-edge-issue10/Safety Helmet.mp4' \
  --frames 100 --skip-host-check --skip-error-check
```

Run board commands under the repository lock:

```sh
flock -w 1800 /tmp/cosmo-edge-rk3588-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 6022 YTHC@120.48.106.5 "<command>"'
```

Record `uname -a`, `/proc/device-tree/model`, decoder selection, frame count,
detector count, elapsed time, peak RSS, and the source/destination DMA-BUF fd
pair. Do not add an FPS pass/fail threshold.

## Report 2026-08-02

The measured one-instance run is in [`report.json`](report.json). The smoke
also creates one detector per task and rejects the fourth, proving the
configured three-instance ceiling. The current standalone smoke does not
expose independent per-stage timers or peak-RSS sampling, so those fields are
explicitly marked `not_instrumented` rather than inferred from end-to-end time.
Add stage timing only when the runtime exposes it without changing the
zero-copy path.

| Detector instances | Decode | RGA | RKNN | Postprocess | End-to-end | Memory | Result |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | hardware path verified | fd 29 → 8 | six outputs verified | 168 detections / 100 frames | 19.66 FPS | not instrumented | PASS |
| 3 | pool allocation/cap verified | not instrumented | not instrumented | not instrumented | not instrumented | not instrumented | PASS |

"PASS" here means the run completed and the contracts held; it is not a
performance target.
