# Timed workbench pilot

> **Status:** parked until *Release readiness* ([Roadmap](Roadmap.md)). Do not schedule participants until the owner requests release preparation.

The pilot measures how quickly a participant reaches a correct conclusion about the primary slice and the normal-image guardrail. Use the same frozen reference workspace and CPU runtime for both methods. Keep holdout closed during this exercise.

## Session

1. Give the participant the reference workspace, Desktop, the runner CLI commands in the README, and two tasks. For task A, zero activation channel 17 in block 2. For task B, zero weight channel 17 in block 2. Each task asks whether the edit improves blurred true nines and whether normal accuracy worsens.
2. Alternate the assigned method and task order across participants. One sequence starts with task A on Desktop and then task B using the script. The other starts with task A using the script and then task B on Desktop. Use fresh output directories and the same baseline cache condition for each participant.
3. Start timing when the task is shown. Stop when the participant states both conclusions and identifies one changed sample. Record errors, requests for assistance and incomplete attempts.
4. Record an anonymous participant ID, sequence, task, method, elapsed seconds, run ID, primary and guardrail answers, and changed sample ID. Keep the answer key from participants until both tasks finish.

## Scoring

Check each answer against the published run report and comparison. Count correct primary-slice conclusions, correct guardrail conclusions and valid changed sample IDs separately. Report time and errors by participant and method together with participant count. The [reference measurements](m1-measurements.md) contain the CPU answer key for these two tasks.
