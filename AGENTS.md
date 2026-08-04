# Agent instructions

## Code search

- When exploring or searching this codebase, use the project `code-review-graph`
  MCP first, especially `get-minimal-context-tool`,
  `semantic-search-nodes-tool`, `query-graph-tool`, and
  `traverse-graph-tool`.
- Use `/home/yanky/work/projects/cosmo-edge` as `repo_root` when calling the
  MCP tools.
- Fall back to `rg` or other text search only when the graph MCP is unavailable
  or does not provide enough detail.

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

@/home/yanky/.codex/RTK.md
