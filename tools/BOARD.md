# BEARERS THEORY project board

The board lives at https://github.com/users/vikingdude81/projects/1 (private).
It tracks the paper deliverables, the CCA experiment arc, the framework notes, and
the open levers for Bearer Theory / Life Before Language.

## Fields

- **Status** (built in): Todo, In Progress, Done
- **Type**: Paper, Experiment, Framework, Lever, Doc/Infra
- **Result**: Confirmed, Null, Partial, Open, N/A
- **Prediction**: P1, P2, P3, P4, P5, P7, PSD, None
- **Detail**: free text, a pointer to the RESULTS doc or script

## Keeping it current: board.sh

`tools/board.sh` wraps the `gh` CLI so single updates are one line. It needs the
`gh` token to carry the `project` scope (one time: `gh auth refresh -s project`).

    # show everything, grouped as status | result | title
    ./tools/board.sh ls

    # move a lever as work starts / finishes
    ./tools/board.sh update "Calibrate viability" status "In Progress"
    ./tools/board.sh update "Calibrate viability" result Partial

    # add a new item when an experiment lands
    ./tools/board.sh add "New probe X" Experiment Open P2 Todo cca/RESULTS_X.md

The title argument to `update` is a case-insensitive substring; the first match
wins. Field names for update are: type, result, prediction, status, detail.

## Creating the views (must be done in the GitHub UI)

The API cannot create or reorder project views yet, so these are a few clicks each
in the browser. On the board page, use the "+ New view" tab at the top.

1. **Board by Status** (the kanban you asked for)
   - New view -> Board layout.
   - "Group by" -> Status.
   - Result: Todo / In Progress / Done columns. The five levers sit in Todo, the
     rest in Done.

2. **Table by Result** (the empirical arc at a glance)
   - New view -> Table layout.
   - "Group by" -> Result.
   - Result: Confirmed / Null / Partial / Open sections. This is the honest
     scoreboard: two Confirmed frameworks, one Confirmed P1 counterfactual, a row
     of Null forager designs, the Partial curriculum and Horn runs, the Open next
     steps.

3. **Table by Prediction** (coverage against Bearer Theory)
   - New view -> Table layout.
   - "Group by" -> Prediction.
   - Result: which manuscript predictions (P1..P7, PSD) have experiments behind
     them and which are thin. Good for spotting gaps before submission.

Optional: in any table view, use the field visibility control to show Type,
Result, Prediction, and Detail as columns and hide the ones you do not want.

## Field and option IDs (for scripting)

    project      PVT_kwHOBb3Bwc4BfZjO
    Status field PVTSSF_lAHOBb3Bwc4BfZjOzhZst_U   Todo f75ad846  In Progress 47fc9ee4  Done 98236657
    Type field   PVTSSF_lAHOBb3Bwc4BfZjOzhZt1W0   Paper e7ba1490  Experiment 1af755d5  Framework 30cc875e  Lever 1f8984e8  Doc/Infra b2b6936e
    Result field PVTSSF_lAHOBb3Bwc4BfZjOzhZt1Xw   Confirmed 8b8382b3  Null af73802f  Partial b0da3269  Open a35e2c48  N/A 9798f52b
    Pred field   PVTSSF_lAHOBb3Bwc4BfZjOzhZt1ag   P1 da37ea47  P2 c6c441fd  P3 fef6134a  P4 e955f577  P5 6cbebe94  P7 1ff2ea3a  PSD f729f46d  None b8e5e5c1
    Detail field PVTF_lAHOBb3Bwc4BfZjOzhZt1bM     (text)
