# runtime.md — probed 2026-10-13 (session: Qoder agent, this repo)

Every row says how it was checked. Tiers: H = frontier reasoning, M = solid coder, L = fast/cheap.

## Models
| model | tier | available? | how checked |
|---|---|---|---|
| this agent (Qoder) | H-capable, single context | yes | running now; large context, full tool set |
| Claude Code headless | H (`opus`) / M (`sonnet`) / L (`haiku`) | CLI present, **2.1.296** | `claude --version`; `claude -p --model haiku "Reply RUNTIME_OK"` → `RUNTIME_OK` (exit 0, ~20 s) |
| Gemini / Antigravity | M–H ideation | CLI present, **1.1.15** | `agy --version` (a real `agy -p` call was not run this session — treat ideation availability as UNPROBED) |
| subagents (in-app) | M | yes | `Agent` tool: Explore / Plan / general-purpose; SearchWorkspace from the main agent only |

**Mode: multi-tier.** `research/config.yaml:72-77` routing still applies (replan/spec/attack = opus, build = sonnet).
Rule §2.3 does not trigger: no open `tier: L` D3 file predates this session except the audit handoff, which is H-grade work.

## Tools
| tool | state | how checked |
|---|---|---|
| shell | Git Bash on win32; Python **3.13.13** | `python --version` |
| file read/edit/write | present | used this session |
| web search/fetch | present (WebSearch/WebFetch) | available, not exercised |
| long context | large; still use file handoffs (§2.4) | n/a |

## Services
| service | state | how checked |
|---|---|---|
| Kaggle CLI | **2.2.4**, auth works | `kaggle --version`; `poll_lb.py` listed submissions |
| competition slug | `enveda-CASMI26-molecule-id-mass-spectra` — `enveda-casmi26` returns **403 Forbidden** on ListSubmissions | both called this session |
| exploit worktree | exists, branch `line/exploit`, HEAD `052c96f`; **uncommitted**: `casmi26-e006-engine2-fppair-pv88.ipynb` untracked, `build/test_patch_engine2_fppair.py` modified | `git -C E:/casmi/lines/exploit status --short` |
| loop tests | **green**: `test_bin.py` 52/52, `test_campaign.py` ok, `harness/test_suite.py` ok, `preflight.py --selftest` ok | run this session |
| concurrent agents | 2× `claude.exe` resident (~110–120 MB) | `tasklist` — campaign state untouched this session |

## Missing / unresolved
- `research/runtime.md` did not exist before this file (§1 deliverable was never written).
- `research/audit/` does not exist: the loop-audit deliverable (`loop_audit.md` §6) was never produced.
- LB of cycle-3 probe `syedtoubur/casmi-engine2-pv88` (G4 PASS 2026-10-12 21:58) is **unknown to code** —
  `poll_lb.py` shows an UNMATCHED submission at 2026-10-12 17:55 UTC with an empty score (W13 recurrence).
- `agy -p` headless ideation unprobed; Scout's web access unconfirmed.

## Qoder — probed 2026-10-14 00:31 BST (2026-10-13 23:31 UTC), campaign 20261014-qoder-structural
| item | value | how checked |
|---|---|---|
| model | Qoder agent; model name/family not exposed to the agent. Context: large (auto-compacted). **Tier recorded: M** (conservative: identity unverifiable, so tier L/M safeguards of qoder_prompt §4.3 apply) | runtime self-report |
| shell | Git Bash (win32), PowerShell 5.1 available | Bash tool |
| file read/edit, web search/fetch | present | tool list |
| python | 3.13.13 | `python --version` |
| kaggle CLI | 2.2.4 | `kaggle --version` |
| exploit worktree | branch `line/exploit`, ahead of origin by 4; many unstaged modifications (other agents' work, untouched) | `git -C E:/casmi/lines/exploit status` |
| GPU (2026-W42) | 1.22 GPU-h used (control 0.34, instrument 0.89) | `python research/bin/gpu.py` |
| harness battery | C1_npx, C1_xlib, C2, C2_hard, C2_np all have `post_dump/state.pkl` | `ls research/harness/dumps/*/post_dump/state.pkl` |
| concurrent agents | 2x claude.exe + 1 python.exe resident | `tasklist` |
| submit autonomy | granted in chat for this session only (qoder_prompt §10) | human message |
