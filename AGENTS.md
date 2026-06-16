# Repository Agent Rules

This file governs the entire repository tree.

## Documentation-Only By Default

Agents working in this repo must default to documentation-only work unless the user explicitly asks for implementation, repair, refactoring, restructuring, test changes, dependency changes, or other non-documentation edits.

Allowed by default:

- Inspect files, run non-destructive read-only commands, and gather evidence.
- Write or update documentation, notes, plans, status reports, handoffs, TODOs, and issue summaries.
- Report problems found in code, structure, tests, dependencies, environment setup, or project organization.

Not allowed by default:

- Automatically fix discovered code problems.
- Automatically restructure directories or rename files.
- Automatically change tests, dependencies, configuration, generated artifacts, or runtime behavior.
- Treat a discovered issue as permission to modify non-documentation files.

When an agent discovers a problem, it should report the issue with file references, evidence, likely impact, and a recommended next step. It should not fix the issue unless the user has explicitly requested that fix or has clearly authorized non-documentation edits for the current task.

If a request is ambiguous, preserve the documentation-only default and ask only when the next action would require non-documentation changes.

## Git Is Read-Only For Agents

Agents may inspect Git state in this repo when it helps produce an accurate report or handoff.

Allowed Git actions by default:

- Read-only inspection such as status, diff, log, show, branch listing, and file history.
- Reporting what Git inspection shows.

Not allowed unless the user explicitly asks for the exact Git action:

- Staging, committing, pushing, pulling, branching, merging, rebasing, restoring, resetting, tagging, amending, changing remotes, changing Git config, or any other operation that modifies repository state or history.

Git modifications are a manual human responsibility in this repo. Agents should inspect and report, not mutate Git state.
