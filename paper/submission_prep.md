# Submission kit: Life Before Language to PhilArchive and PhilSci-Archive

*Prepared 2026-08-08. Everything you paste into the two open-access philosophy
preprint servers, plus the step-by-step. No endorsement required at either. No
em-dashes.*

---

## Venue routing

- **PhilArchive** (philarchive.org, the open archive behind PhilPapers). Best fit
  for a philosophy-of-mind paper. Anyone can deposit, it is indexed by PhilPapers,
  and it reaches the phil-of-mind audience directly. Start here.
- **PhilSci-Archive** (philsci-archive.pitt.edu). Philosophy of science, which
  covers philosophy of cognitive science and of AI. The experiments-in-a-
  philosophy-paper framing fits. Deposit here as well for the science-of-AI audience.

Both are free, no endorsement, and allow later replacement of the file with a new
version. Cross-posting to both is normal and fine.

## One decision before you submit

The paper's abstract reports the pre-2026-08 state: "Bearer-identity in the
continuant probe becomes load-bearing exactly when outcomes stop being facts about
the world and become facts of the bearer. Nothing measured crosses the wall." The
Constitutive GPT results from this session confirm and strengthen the load-bearing
claim (self versus wrong-bearer, Prediction 2, plus Prediction 4 and scale). Two
options:

1. **Submit v0.2 as is.** It is self-contained and advisor-approved, and the GPT
   results live in the separate addendum. Fastest.
2. **Fold a short paragraph of the GPT confirmation into section 8 first.** Makes a
   stronger submission and points readers to the public code. About an hour.

Recommendation: option 2 if you want the strongest first impression, since the new
result is the positive payoff the original paper could only gesture at. Either way
the kit below applies.

## Metadata to paste (both venues)

**Title:**
Life Before Language: Separating the Engineering Gap from the Metaphysical Gap in
Debates on Machine Consciousness

**Author:** Alex Bone (Independent Researcher, Long Beach, CA)

**Abstract:** (use the paper's abstract verbatim; a lightly trimmed ~180-word
version is below if a shorter field is wanted)

> Contemporary debates about artificial intelligence and consciousness routinely
> compare human and machine cognition at their endpoints, fluent language and
> symbolic reasoning, rather than at their origins. Human consciousness does not
> begin with language. It begins with a pre-rational developmental interface of
> appetite, attachment, sensation, consequence, and temporal continuity, through
> which reasoning later emerges as the act of a self. Large language models invert
> this ordering, beginning at the symbolic layer with no developmental substrate
> beneath it. The paper formalizes a seven-level hierarchy of the developmental
> interface, distinguishes an engineering gap (bridgeable in principle) from a
> metaphysical gap (possibly not), and reports a two-workstream empirical program.
> A construction-order experiment yields three stable regimes: inference-time
> overlays are a tax, fine-tuned retrofits are a crutch, and constraints present
> from the first training step are internalized, removable at almost no cost with
> their discipline persisting in the weights. This refines the framework into a
> graded ladder, modulation to internalization to the wall to consequence to
> subjectivity. Bearer-identity becomes load-bearing exactly when outcomes stop
> being facts about the world and become facts of the bearer. The experiments
> locate the wall; they do not dissolve it.

**Keywords:** philosophy of mind, machine consciousness, developmental cognition,
hallucination, embodiment, construction order, internalization, Leibniz, Boethius

**Data and code availability** (add this line to the paper and the deposit note):
> Code for two of the probe systems is public under MIT: the entropy-decay decoder
> at github.com/vikingdude81/gabriels-horn and the bearer-indexed continuant agent
> at github.com/vikingdude81/constitutive-gpt.

## Subject categories

**PhilArchive / PhilPapers taxonomy** (pick 2 to 3):
- Philosophy of Cognitive Science > Philosophy of Artificial Intelligence
- Philosophy of Mind > Consciousness > Machine Consciousness (or the nearest
  Consciousness subcategory offered)
- Philosophy of Cognitive Science > Philosophy of Cognitive Science, Misc

**PhilSci-Archive subjects** (pick 2 to 3):
- Specific Sciences > Cognitive Science > Artificial Intelligence
- Specific Sciences > Cognitive Science
- General Philosophy of Science, if a third is wanted

## The file

Both servers want a PDF. Use the rendered submission PDF
(Life_Before_Language_v1_submission.pdf, or re-render from the current markdown with
paper/render_submission.py so the version matches). Confirm Figure 1 renders in the
PDF. If you take option 2 above, re-render after adding the GPT paragraph.

## Step by step

**PhilArchive:**
1. Create an account at philpapers.org (free), then go to philarchive.org and choose
   Submit.
2. Enter title, author, abstract, keywords, and the categories above.
3. Upload the PDF. Set license to the default open-access, or CC-BY if offered.
4. Add the data-and-code line in the notes field. Submit. Moderation is light and
   usually clears within a day or two.

**PhilSci-Archive:**
1. Register at philsci-archive.pitt.edu, then Deposit.
2. Type: Preprint. Enter title, authors, abstract, keywords, subjects.
3. Upload the PDF, set the date, submit for the editors' quick check.

## After both are up

- Add the PhilArchive and PhilSci-Archive links to each other and to the GitHub
  repos' About fields, so the paper, the code, and the archives point at each other.
- These give the conceptual paper a citable home immediately, with no endorsement
  gate, while the empirical addendum takes the arXiv or Zenodo route separately.

## The second paper

Bearer Theory (the formal manuscript) can follow the same route once it is in
submission shape. It is the formal companion to this conceptual paper, and the same
metadata pattern applies with a more formal keyword set (formal epistemology,
theory of intelligence, developmental experience).
