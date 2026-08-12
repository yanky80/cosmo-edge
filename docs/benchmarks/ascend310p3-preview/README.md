# Ascend 310P3 YOLO26 validation and benchmark

This is a reproducible validation and capacity report for the completed YOLO26
task path on the shared 310P3 host, not an FPS acceptance gate. It uses one
fixed host, model, and video; rerun the commands below after changing any of
them and publish a new report instead of comparing unlike runs.

## Fixed inputs

| Item | Value |
| --- | --- |
| Host | `root@35623rcqc768.vicp.fun:1022` (shared 310P3 test host, see `docs/development/ascend310p3-test-host-baseline.md`) |
| NPU | Ascend 310P3, device 0 (`npu-smi 24.1.1.1`, driver 24.1.1.1, firmware 7.5.0.5.220) |
| Model (OM) | `/opt/convert/bjsubway-yolo26/model.om` (FP16, SHA256 `e8a14da4…d76f8`) |
| Model (ONNX FP32 baseline) | `/opt/convert/bjsubway-yolo26/best.onnx` (SHA256 `e5406bd4…8863a`) |
| Video | `/tmp/subway_test_h264_x2.mp4` (1280x720 H.264 25 fps, 150 frames = the fixed 75-frame subway clip concatenated twice so the first 100 valid frames fit) |
| Task smoke | `test/ascend310p3/ascend_task_smoke.cc` (built on host, command in the file header) |
| Thresholds | conf 0.25, matched-box IoU >= 0.98, confidence absolute error <= 0.01 |

The public repository contains the report and commands only; host-local model
and video paths are intentionally not copied into the repository.

## Validation: first 100 valid frames vs ONNX FP32 baseline

`ascend_task_smoke --dump-detections` writes per-frame detections as JSONL
(`{"frame":N,"dets":[{"class","score","cx","cy","w","h"}]}` in 960x960
letterbox pixels). `test/ascend310p3/onnx_fp32_compare.py` runs the ONNX FP32
baseline with the same letterbox contract and compares the two dumps
(class match, greedy same-class IoU matching, IoU >= 0.98, confidence
absolute error <= 0.01). Frames are "valid" when the baseline has at least one
detection at the confidence threshold; the first 100 valid frames are compared.

## Benchmark: decode / transfer / DVPP / ACL / postprocess / e2e

The task smoke prints per-stage timings for every frame
(`frame=N e2e_graph=… image_to_tensor=… acl=… postprocess=… dets=…` and the
per-node `Forward:` lines with upload/dvpp/download/normalize and
h2d/execute+sync/d2h breakdowns). `decode` is reported as the
`source session done: decoded=N in T s (X fps incl. decode)` line. CPU and
peak RSS come from `/usr/bin/time -v`; NPU utilization and device memory come
from `npu-smi info` sampled during the run.

The checked-in `report.json` also records the means parsed from those logs:
DVPP image upload/download/normalization, DVPP processing, and ACL input H2D,
execute+sync, and output D2H+conversion for both detector counts. The
device-frame path retains the small ACL H2D and D2H values needed by the
model buffers; these are not host-video-frame copies.

## Soak: 30+ minutes, 1 and 3 detectors

Run the task smoke with enough `--frames × --rounds` for at least 30 minutes
of wall time (rounds are the product task start/stop lifecycle on one
persistent decoder). Sample process RSS and `npu-smi` device memory every
20 seconds during the run; after the process exits, `npu-smi` device memory
must return to the idle baseline (no unreleased device memory after task
teardown).

## Reproduction

All hardware commands run under the shared serialization lock:

```sh
flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun "<command>"'
```

Build the task smoke on the host (`bash build_task_smoke.sh`, command in the
file header), then:

```sh
# 100-frame validation
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video /tmp/subway_test_h264_x2.mp4 --frames 150 --conf 0.25 \
  --dump-detections /tmp/ascend_dets.jsonl
python3 test/ascend310p3/onnx_fp32_compare.py --mode baseline \
  --onnx /opt/convert/bjsubway-yolo26/best.onnx \
  --video /tmp/subway_test_h264_x2.mp4 --frames 150 --conf 0.25 \
  --out /tmp/baseline.jsonl
python3 test/ascend310p3/onnx_fp32_compare.py --mode compare \
  --baseline /tmp/baseline.jsonl --ascend /tmp/ascend_dets.jsonl \
  --out /tmp/compare_report.json

# benchmarks (1 and 3 instances)
/usr/bin/time -v -o /tmp/bench1.time ./ascend_task_smoke --om … --frames 300 --instances 1
/usr/bin/time -v -o /tmp/bench3.time ./ascend_task_smoke --om … --frames 300 --instances 3

# soaks (>= 30 min each)
./ascend_task_smoke --om … --frames 1000 --instances 1 --rounds 48
./ascend_task_smoke --om … --frames 350 --instances 3 --rounds 36
```

Record `uname -a`, `npu-smi info`, decoder selection, frame count, detector
count, elapsed time, peak RSS, per-stage timings, NPU utilization, and device
memory before/after teardown. Do not add an FPS pass/fail threshold.
