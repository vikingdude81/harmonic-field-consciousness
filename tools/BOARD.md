# Project boards

Two private GitHub Project boards track the research program, split by paper:

- **bearer** = BEARERS THEORY (https://github.com/users/vikingdude81/projects/1)
  The formal theory manuscripts, the CCA experiment arc, the escape/observability
  frameworks, and the open levers. Has a **Prediction** field (P1..P7, PSD).
- **lbl** = Life Before Language (https://github.com/users/vikingdude81/projects/2)
  The 7-level framework paper, the probe program (Gabriel's Horn, construction-order
  on NanoGPT, the answerability battery), venue/outreach, and the related-work note.
  No Prediction field.

Items that belong to the Life Before Language umbrella (the LBL paper, social
threads, the probe-matrix experiments, the literature note) live on **lbl**. Items
specific to the formal theory and its CCA tests live on **bearer**.

## Fields

Both boards: **Status** (Todo/In Progress/Done), **Type** (Paper/Experiment/
Framework/Lever/Doc/Infra), **Result** (Confirmed/Null/Partial/Open/N/A), **Detail**
(text pointer). The **bearer** board additionally has **Prediction**.

## Keeping them current: board.sh

`tools/board.sh` wraps the `gh` CLI. It needs the `gh` token to carry the `project`
scope (one time: `gh auth refresh -s project`). First argument selects the board.

    ./tools/board.sh bearer ls
    ./tools/board.sh lbl ls

    # move a lever as work starts / finishes
    ./tools/board.sh bearer update "Calibrate viability" status "In Progress"
    ./tools/board.sh bearer update "Calibrate viability" result Partial

    # add a new item when an experiment lands
    ./tools/board.sh bearer add "New probe X" Experiment Open Todo cca/RESULTS_X.md P2
    ./tools/board.sh lbl    add "New LBL probe" Experiment Open Todo PROBE_MATRIX.md

The add signature is: `add "<title>" <Type> <Result> <Status> <detail> [Prediction]`.
Prediction is optional and only applied on the bearer board. The `update` title is a
case-insensitive substring; first match wins. Update fields: type, result, status,
detail, and prediction (bearer only).

## Creating the views (must be done in the GitHub UI)

The API cannot create or reorder project views. On each board page use the
"+ New view" tab.

- **Board by Status** (kanban): New view -> Board -> Group by Status.
- **Table by Result** (the empirical scoreboard): New view -> Table -> Group by
  Result. Confirmed / Null / Partial / Open sections.
- **bearer only, Table by Prediction** (coverage against the manuscript): Group by
  Prediction to see which P1..P7/PSD have experiments behind them.

## Field and option IDs (for scripting)

    -- bearer (project 1) --
    project      PVT_kwHOBb3Bwc4BfZjO
    Status       PVTSSF_lAHOBb3Bwc4BfZjOzhZst_U   Todo f75ad846  In Progress 47fc9ee4  Done 98236657
    Type         PVTSSF_lAHOBb3Bwc4BfZjOzhZt1W0   Paper e7ba1490  Experiment 1af755d5  Framework 30cc875e  Lever 1f8984e8  Doc/Infra b2b6936e
    Result       PVTSSF_lAHOBb3Bwc4BfZjOzhZt1Xw   Confirmed 8b8382b3  Null af73802f  Partial b0da3269  Open a35e2c48  N/A 9798f52b
    Prediction   PVTSSF_lAHOBb3Bwc4BfZjOzhZt1ag   P1 da37ea47  P2 c6c441fd  P3 fef6134a  P4 e955f577  P5 6cbebe94  P7 1ff2ea3a  PSD f729f46d  None b8e5e5c1
    Detail       PVTF_lAHOBb3Bwc4BfZjOzhZt1bM     (text)

    -- lbl (project 2) --
    project      PVT_kwHOBb3Bwc4BfZje
    Status       PVTSSF_lAHOBb3Bwc4BfZjezhZsuNs   Todo f75ad846  In Progress 47fc9ee4  Done 98236657
    Type         PVTSSF_lAHOBb3Bwc4BfZjezhZt9es   Paper 6e4b2019  Experiment aa52dc9b  Framework 97b50325  Lever d051082d  Doc/Infra 0810f584
    Result       PVTSSF_lAHOBb3Bwc4BfZjezhZt9ew   Confirmed 273a7cbe  Null 89a1b10e  Partial b924f24a  Open dd8901ed  N/A 14b765c3
    Detail       PVTF_lAHOBb3Bwc4BfZjezhZt9e0     (text)
