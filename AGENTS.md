# Agent instructions

## RK3588 hardware validation

- The shared RK3588 test host is available with:
  `ssh -o ConnectTimeout=10 -p 6022 YTHC@120.48.106.5`
- Use the board for issue acceptance checks that require real RK3588 hardware.
  Keep local/unit tests as the first gate.
- Serialize board access across worktrees. Run remote checks under the local lock,
  for example:
  `flock -w 1800 /tmp/cosmo-edge-rk3588-hw.lock -c 'ssh -o ConnectTimeout=10 -p 6022 YTHC@120.48.106.5 "<command>"'`
- Before testing, confirm the target with `uname -a` and
  `cat /proc/device-tree/model`.
- Do not reboot, provision, install system packages, change persistent system
  configuration, or stop unrelated processes without explicit user approval.
- Do not store passwords, private keys, tokens, or device-specific secrets in
  the repository. Do not assume passwordless `sudo`.
- Record the exact board-test commands and relevant results in the pull request.

## Huawei Ascend 310P3 hardware validation

- The shared Huawei Ascend 310P3 test host is available with:
  `ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun`
- Use the host for acceptance checks that require real 310P3 hardware.
  Keep local/unit tests as the first gate.
- Serialize host access across worktrees. Run remote checks under the local lock,
  for example:
  `flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c 'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun "<command>"'`
- Before testing, confirm the target with `uname -a` and `npu-smi info`.
- Do not reboot, provision, install system packages, change persistent system
  configuration, or stop unrelated processes without explicit user approval.
- Do not store passwords, private keys, tokens, or device-specific secrets in
  the repository.
- Record the exact host-test commands and relevant results in the pull request.
