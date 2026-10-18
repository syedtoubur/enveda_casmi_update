---
name: smart-zone-handoff
description: >-
  Generates a comprehensive, high-fidelity handoff dossier when context reaches the 140k Smart Zone ceiling or when ending a session, saving state to .scratch/handoffs/latest.md so a fresh session can immediately continue.
---

# Smart Zone Handoff Skill

This skill creates a compact, high-density bridge between an expiring agent session (which has reached or exceeded the **140k token Smart Zone**) and a fresh session.

The goal is to allow the user to start a fresh chat with only ~3,000–5,000 tokens of clean context, restoring 100% of model reasoning and attention sharpness without losing any decisions, files, or progress.

---

## Process

### 1. Ensure Storage Directory
Ensure the directory `.scratch/handoffs/` exists in the repository root.

### 2. Synthesize the Handoff Dossier
Write a structured Markdown document containing the following 6 mandatory sections:

```markdown
# Session Handoff Dossier
**Generated:** <YYYY-MM-DD HH:MM:SS>
**Source Conversation ID:** <conversation_id if known>
**Context Size:** ~<estimated_tokens> tokens (Smart Zone Exit)

## 1. High-Level Mission & Active Objective
State the overarching goal and the specific problem tackled in this session.

## 2. Settled Decisions (Do NOT Re-Litigate)
Bullet points of technical choices, thresholds, parameters, and design decisions finalized during this session.
Format:
- **Decision:** <what was chosen>
  - *Rationale:* <why this was picked over alternatives>
  - *Impact:* <how downstream components behave>

## 3. Modified & Created Files
List all files touched or generated during the session, formatted as clickable links:
- [`path/to/file`](file:///absolute/path/to/file) - <1-line summary of what it does>

## 4. Current Work in Flight
- What was being worked on right before the session ended.
- Any open PRs, drafts, or unverified changes.

## 5. The Immediate Frontier (Next Steps)
List the next 2-4 concrete, prioritized actions for the new agent:
1. <Step 1>
2. <Step 2>
3. <Step 3>

## 6. Recommended Skills & Boot Command
Specify which skills the next agent should call (e.g., `/implement`, `/tdd`, `/code-review`, `/grill-with-docs`).

### Copy-Paste Boot Prompt for New Session:
> "Resume work from [latest.md](file:///<repo_root>/.scratch/handoffs/latest.md). Read the Settled Decisions and immediately proceed with Step 1 of the Immediate Frontier."
```

### 3. File Outputs
Write the synthesized dossier to two locations:
1. `.scratch/handoffs/latest.md` (Always overwritten so the user has a fixed target to point to).
2. `.scratch/handoffs/handoff-<YYYYMMDD-HHMMSS>.md` (Timestamped archive for auditing).

### 4. Direct the User
Once the handoff files are saved:
1. Present the high-level summary to the user.
2. Provide the clickable link to [`.scratch/handoffs/latest.md`](file:///e:/ENVEDA_casmi/.scratch/handoffs/latest.md).
3. Instruct the user:
   *"Open a fresh conversation session and run:*
   `@.scratch/handoffs/latest.md continue`"
