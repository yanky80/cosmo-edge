#!/usr/bin/env python3
"""Issue #34: ONNX FP32 baseline + per-frame comparison for the Ascend 310P3
YOLO26 FP16 OM validation.

Two modes:

  baseline:
    Decode the first --frames frames of --video (OpenCV, presentation order,
    same frame index as the h264_ascend/h265_ascend task smoke) and run the
    ONNX FP32 model with the OM letterbox contract (aspect-fit to 960x960,
    centered 114 padding, BGR->RGB, /255). Ultralytics end2end exports emit
    xyxy in 960x960 letterbox pixels; detections are dumped in that same
    letterbox space, matching `ascend_task_smoke --dump-detections` (the
    engine's yolo_e2e output stays in letterbox pixels; IoU is invariant
    under the uniform-scale + translation back-mapping to original pixels).
    JSONL schema:
      {"frame":N,"dets":[{"class":..,"score":..,"cx":..,"cy":..,"w":..,"h":..},...]}
    cxcywh is in 960x960 letterbox pixels; N is 0-based.

  compare:
    Match baseline detections against the Ascend dump per frame (same class,
    greedy max IoU). A frame passes when every baseline detection has a
    matched-box IoU >= --min-iou and confidence absolute error <=
    --max-conf-err, and the Ascend side has no unmatched extra detections.
    Frames are "valid" when the baseline has at least one detection at the
    confidence threshold; the first --valid-frames valid frames are compared.

Requires on the test host: python3, opencv-python-headless, numpy,
onnxruntime (1.x), the exported ONNX FP32 model, and the same fixed video
used for the Ascend smoke.

Example (310P3 test host, under the flock lock):
  python3 test/ascend310p3/onnx_fp32_compare.py --mode baseline \
      --onnx /opt/convert/bjsubway-yolo26/best.onnx \
      --video /tmp/subway_test_h264_x2.mp4 --frames 100 --out /tmp/baseline.jsonl
  ./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
      --video /tmp/subway_test_h264_x2.mp4 --frames 100 --conf 0.25 \
      --dump-detections /tmp/ascend_dets.jsonl
  python3 test/ascend310p3/onnx_fp32_compare.py --mode compare \
      --baseline /tmp/baseline.jsonl --ascend /tmp/ascend_dets.jsonl \
      --out /tmp/compare_report.json
"""

import argparse
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 960
PAD_COLOR = 114.0


def letterbox(frame_hw):
    """Aspect-fit to INPUT_SIZE with centered padding, mirroring the OM
    image_to_tensor (DVPP) contract and Ultralytics auto letterbox."""
    h, w = frame_hw
    r = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    return r, pad_x, pad_y


def run_baseline(args):
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r, pad_x, pad_y = letterbox((height, width))
    out = open(args.out, "w")
    frames_dumped = 0
    while frames_dumped < args.frames:
        ok, bgr = cap.read()
        if not ok:
            sys.exit(f"video ended after {frames_dumped} frames, need {args.frames}")
        resized = cv2.resize(bgr, (int(round(width * r)), int(round(height * r))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_COLOR, dtype=np.uint8)
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None]  # NCHW 0..1
        out_raw = sess.run(None, {in_name: blob})[0]  # [1, 300, 6] xyxy score cls
        dets = []
        for row in out_raw[0]:
            # xyxy is already in 960x960 letterbox pixels.
            x1 = min(INPUT_SIZE, max(0.0, float(row[0])))
            y1 = min(INPUT_SIZE, max(0.0, float(row[1])))
            x2 = min(INPUT_SIZE, max(0.0, float(row[2])))
            y2 = min(INPUT_SIZE, max(0.0, float(row[3])))
            score, cls = float(row[4]), float(row[5])
            if score < args.conf:
                continue
            dets.append({
                "class": cls,
                "score": score,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": x2 - x1,
                "h": y2 - y1,
            })
        out.write(json.dumps({"frame": frames_dumped, "dets": dets}) + "\n")
        frames_dumped += 1
    out.close()
    print(f"baseline done: {frames_dumped} frames -> {args.out} (model {args.onnx})")


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1, bx2, by2 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def run_compare(args):
    def load(path):
        by_frame = {}
        for line in open(path):
            obj = json.loads(line)
            by_frame[obj["frame"]] = obj["dets"]
        return by_frame

    base = load(args.baseline)
    asc = load(args.ascend)
    valid_frames = [f for f in sorted(base) if base[f]]
    compared = valid_frames[: args.valid_frames]
    worst_iou, max_conf_err = 1.0, 0.0
    frames_passed = 0
    pairs = 0
    detail = []
    for f in compared:
        b_dets = sorted(base[f], key=lambda d: -d["score"])
        a_dets = asc.get(f, [])
        used = [False] * len(a_dets)
        frame_ok = True
        for b in b_dets:
            best_i, best_iou = -1, 0.0
            for i, a in enumerate(a_dets):
                if used[i] or a["class"] != b["class"]:
                    continue
                iou = _iou(b, a)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i < 0 or best_iou < args.min_iou:
                frame_ok = False
                detail.append({"frame": f, "ok": False,
                               "reason": f"baseline det class={b['class']} unmatched (best IoU {best_iou:.4f})"})
                continue
            used[best_i] = True
            conf_err = abs(a_dets[best_i]["score"] - b["score"])
            pairs += 1
            worst_iou = min(worst_iou, best_iou)
            max_conf_err = max(max_conf_err, conf_err)
            if conf_err > args.max_conf_err:
                frame_ok = False
                detail.append({"frame": f, "ok": False,
                               "reason": f"conf err {conf_err:.4f} > {args.max_conf_err}"})
        for i, a in enumerate(a_dets):
            if not used[i]:
                frame_ok = False
                detail.append({"frame": f, "ok": False,
                               "reason": f"extra ascend det class={a['class']} score={a['score']:.4f}"})
        if frame_ok:
            frames_passed += 1
        detail.append({"frame": f, "ok": frame_ok, "baseline_dets": len(b_dets),
                       "ascend_dets": len(a_dets)})
    passed = frames_passed == len(compared) and len(compared) > 0
    report = {
        "schema": "cosmo-edge.ascend310p3-validate/v1",
        "baseline": args.baseline,
        "ascend": args.ascend,
        "min_iou": args.min_iou,
        "max_conf_err": args.max_conf_err,
        "conf_threshold": args.conf,
        "valid_frames_requested": args.valid_frames,
        "frames_compared": len(compared),
        "frames_passed": frames_passed,
        "matched_pairs": pairs,
        "worst_iou": worst_iou,
        "max_conf_abs_err": max_conf_err,
        "classes_match": passed,
        "result": "pass" if passed else "fail",
        "frame_detail": detail,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("frames_compared", "frames_passed", "matched_pairs",
                                             "worst_iou", "max_conf_abs_err", "classes_match",
                                             "result")}, indent=2))
    if not passed:
        sys.exit("comparison FAILED: see frame_detail in " + args.out)


def main():
    ap = argparse.ArgumentParser(description="ONNX FP32 baseline + Ascend FP16 OM comparison")
    ap.add_argument("--mode", choices=["baseline", "compare"], required=True)
    ap.add_argument("--onnx")
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--ascend")
    ap.add_argument("--min-iou", type=float, default=0.98)
    ap.add_argument("--max-conf-err", type=float, default=0.01)
    ap.add_argument("--valid-frames", type=int, default=100)
    args = ap.parse_args()
    if args.mode == "baseline":
        for k in ("onnx", "video"):
            if not getattr(args, k):
                ap.error(f"--mode baseline requires --{k}")
        run_baseline(args)
    else:
        for k in ("baseline", "ascend"):
            if not getattr(args, k):
                ap.error(f"--mode compare requires --{k}")
        run_compare(args)


if __name__ == "__main__":
    main()
