"""
Create Zenodo DRAFT deposition(s) for the Bearer Theory papers, upload the PDF, and set
metadata. It STOPS at draft and prints the edit URL. It never publishes; minting the DOI
stays a human click in the Zenodo web UI (or a deliberate --publish run).

Usage (token stays in your environment, never in this file):
  # Windows PowerShell:
  #   $env:ZENODO_TOKEN = "<your token>"
  # Git Bash:
  #   export ZENODO_TOKEN="<your token>"

  python zenodo_publish.py --paper both            # SANDBOX dry-run (safe, fake DOIs)
  python zenodo_publish.py --paper both --prod      # real Zenodo, still only a DRAFT
  python zenodo_publish.py --paper convergence --prod

Token scopes needed: deposit:write  (and deposit:actions only if you later add --publish).
Get one at https://zenodo.org/account/settings/applications/tokens/new/
(or https://sandbox.zenodo.org/account/settings/applications/tokens/new/ for sandbox).
"""
import os, sys, argparse, json, requests

PAPER_DIR = r"C:\Users\akbon\OneDrive\Documents\GitHub\harmonic-field-consciousness\paper"

CREATORS = [{"name": "Bone, Alex", "affiliation": "Independent Researcher, Long Beach, CA",
             "orcid": "0009-0003-4252-7556"}]

CONVERGENCE_DESC = (
    "An incorporated developmental state, a bearer that both biases behavior and is updated by "
    "that behavior's own outcomes, is central to Bearer Theory's account of when a system has an "
    "identity that is its own. We ask when that state actually does causal work rather than sitting "
    "idle, and we propose a design law, supported by cross-substrate evidence, that predicts three "
    "regimes. When the task admits a non-incorporated route to viability, the bearer is idle, and "
    "increasing capacity increases the escape rather than the incorporation. When the route is "
    "foreclosed but the bearer cannot adapt to carry the required state, the system fails, the "
    "difficulty valley. When the route is foreclosed and the bearer is adaptive, the incorporated "
    "state becomes load-bearing and bearer-specific. We give operational definitions of these terms "
    "and show the mechanism across three substrates that share nothing but the loop: a "
    "reinforcement-learning forager, a transformer with a fast-weight memory, and a "
    "competitive-fatigue dynamical system. A self versus wrong-bearer gap of 0.51, replicated with "
    "tight error bars, dissociated from attention, and holding at scale, establishes load-bearing "
    "non-transferability where the escape is foreclosed with an adaptive bearer, and the same result "
    "reproduces under gradient descent in a language model. A counterintuitive corollary follows for "
    "scaling: a more capable optimizer does not discover its bearer identity, it exploits the escape "
    "better. The claim is structural and agnostic on the phenomenal question by design.\n\n"
    "Code and experiments: https://github.com/vikingdude81/constitutive-gpt"
)

LBL_DESC = (
    "Contemporary debates about artificial intelligence and consciousness routinely commit a category "
    "error: they compare human and machine cognition at their endpoints (fluent language, symbolic "
    "reasoning) rather than at their origins. Human consciousness does not begin with language. It "
    "begins with a pre-rational developmental interface of appetite, attachment, sensation, "
    "consequence, and temporal continuity, through which reasoning later emerges as the act of a self. "
    "Large language models invert this ordering: they begin at the symbolic layer and possess no "
    "developmental substrate beneath it. This paper formalizes a seven-level hierarchy of the "
    "developmental interface and locates current AI within it; distinguishes an engineering gap "
    "(bridgeable in principle) from a metaphysical gap (possibly not); proposes and executes a "
    "two-workstream research program with four implemented probe systems and a flagship "
    "construction-order experiment; and reports the results. Inference-only overlays are a tax; "
    "fine-tuned retrofits are a crutch; constraints present from the first training step are "
    "internalized. Bearer-identity in the continuant probe becomes load-bearing exactly when outcomes "
    "stop being facts about the world and become facts of the bearer. Nothing measured crosses the "
    "wall. The experiments locate the wall; they do not dissolve it.\n\n"
    "Companion probe repositories: https://github.com/vikingdude81/constitutive-gpt , "
    "https://github.com/vikingdude81/gabriels-horn"
)

PAPERS = {
    "convergence": {
        "file": os.path.join(PAPER_DIR, "When_Identity_Is_Load_Bearing_v1_submission.pdf"),
        "metadata": {
            "upload_type": "publication",
            "publication_type": "preprint",
            "title": "When Identity Is Load-Bearing: A Proposed Design Law for Incorporated Developmental State",
            "creators": CREATORS,
            "description": CONVERGENCE_DESC,
            "keywords": ["developmental state", "fast weights", "meta-learning", "personal identity",
                         "substrate independence", "incorporated memory", "non-transferability",
                         "machine consciousness", "Bearer Theory"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "language": "eng",
            "publication_date": "2026-08-09",
        },
    },
    "lbl": {
        "file": os.path.join(PAPER_DIR, "Life_Before_Language_v1_submission.pdf"),
        "metadata": {
            "upload_type": "publication",
            "publication_type": "preprint",
            "title": "Life Before Language: Separating the Engineering Gap from the Metaphysical Gap in Debates on Machine Consciousness",
            "creators": CREATORS,
            "description": LBL_DESC,
            "keywords": ["philosophy of mind", "machine consciousness", "developmental cognition",
                         "hallucination", "embodiment", "construction order", "internalization",
                         "Leibniz", "Boethius", "Bearer Theory"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "language": "eng",
            "publication_date": "2026-07-20",
        },
    },
}


def run(key, base, token, orcid, do_publish):
    p = PAPERS[key]
    fpath = p["file"]
    if not os.path.exists(fpath):
        print(f"  [!] file not found: {fpath}"); return
    md = json.loads(json.dumps(p["metadata"]))  # deep copy
    if orcid:
        for c in md["creators"]:
            c["orcid"] = orcid
    h = {"Authorization": f"Bearer {token}"}

    # 1. create empty draft
    r = requests.post(f"{base}/api/deposit/depositions", json={}, headers=h, timeout=60)
    if r.status_code >= 300:
        print(f"  [!] create failed {r.status_code}: {r.text[:300]}"); return
    dep = r.json(); dep_id = dep["id"]; bucket = dep["links"]["bucket"]

    # 2. upload the PDF into the bucket
    fname = os.path.basename(fpath)
    with open(fpath, "rb") as fh:
        ru = requests.put(f"{bucket}/{fname}", data=fh, headers=h, timeout=600)
    if ru.status_code >= 300:
        print(f"  [!] upload failed {ru.status_code}: {ru.text[:300]}"); return

    # 3. attach metadata
    rm = requests.put(f"{base}/api/deposit/depositions/{dep_id}",
                      json={"metadata": md}, headers=h, timeout=60)
    if rm.status_code >= 300:
        print(f"  [!] metadata failed {rm.status_code}: {rm.text[:500]}"); return

    html = rm.json()["links"].get("html", f"{base}/deposit/{dep_id}")
    print(f"  DRAFT created: {md['title'][:60]}...")
    print(f"    file: {fname}")
    print(f"    review / publish here: {html}")

    if do_publish:
        rp = requests.post(rm.json()["links"]["publish"], headers=h, timeout=60)
        if rp.status_code >= 300:
            print(f"    [!] publish failed {rp.status_code}: {rp.text[:300]}")
        else:
            print(f"    PUBLISHED. DOI: {rp.json().get('doi')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=["convergence", "lbl", "both"], default="both")
    ap.add_argument("--prod", action="store_true", help="use real Zenodo (default: sandbox)")
    ap.add_argument("--orcid", default=os.environ.get("ORCID", ""), help="ORCID iD for the author")
    ap.add_argument("--publish", action="store_true",
                    help="ALSO publish (mints a permanent DOI). Omit to stop at draft.")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        print("ERROR: set ZENODO_TOKEN in your environment first."); sys.exit(1)
    base = "https://zenodo.org" if args.prod else "https://sandbox.zenodo.org"
    print(f"target: {base}  ({'PRODUCTION' if args.prod else 'sandbox dry-run'})")
    if args.publish and args.prod:
        print("!!! --publish on PRODUCTION mints a PERMANENT DOI. Ctrl-C now to abort. !!!")
    keys = ["convergence", "lbl"] if args.paper == "both" else [args.paper]
    for k in keys:
        run(k, base, token, args.orcid.strip() or None, args.publish)


if __name__ == "__main__":
    main()
