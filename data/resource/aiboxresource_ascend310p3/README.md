# Ascend 310P3 Resource Root

This resource root is selected by the `COSMO_TARGET_PLATFORM=ascend310p3`
package profile and installed as the package `resource/` directory.

Model packages must contain a single explicit `.om` artifact (CANN AscendCL)
matching the `ASCEND310P3` chip contract described in
`docs/development/ascend310p3-adaptation-plan.md`. The reproducible ATC/AIPP
conversion record lives in `docs/development/ascend310p3-yolo26-om-atc.md`.
`model_template/yolo26_det.json` is the minimal YOLO26 package template used by
the UI add-model flow and import validation. CANN, driver, firmware and custom
FFmpeg binaries stay external to the repository; only platform-neutral metadata
and algorithm resources live here.
