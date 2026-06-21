"""Extract the REQUIRED experimental procedure from each .docx manual.

This is the requirements side of the procedural grader. For each technique
(microscopy / flow cytometry / western blot) a dedicated extraction tool flattens
its .docx manual and makes one structured LLM call that pulls out:

  - key_factors : the critical experimental choices the assignment requires
                  (cell lines, treatments/conditions, antibodies, channels,
                  gating, controls), each with a short `why`.
  - sequence    : the ordered steps the student is expected to perform.

Output: requirements/<rubric_id>.json

    uv run python scripts/extract_requirements.py

Needs GEMINI_API_KEY (read from .env or the environment).
"""
import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_rubrics import extract_docx  # reuse the docx flattener

from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REQUIREMENTS = ROOT / "requirements"
DEFAULT_MODEL = os.environ.get("EXTRACT_MODEL", "gemini-3.1-flash-lite")

# rubric_id -> (technique, docx filename)
TARGETS = {
    "entry_microscopy": (
        "microscopy",
        "Entry Microscopy for Training_With Data_Answers.docx",
    ),
    "entry_flow_cytometry": (
        "flow_cytometry",
        "Entry Flow Cytometry for Training_With Data_Answers.docx",
    ),
    "entry_western_blot": (
        "western_blot",
        "Entry Western Blot for Training_With Data_Answers.docx",
    ),
}

# Per-technique guidance — the "different tool" for each experiment type. Tells
# the model which key factors and which canonical step sequence to look for.
TECHNIQUE_GUIDANCE = {
    "microscopy": """TECHNIQUE: Fluorescence microscopy.
Key factors to capture:
- samples/cell lines that MUST be imaged (the unknowns to identify + any labelled
  reference organelles used as controls).
- fluorescence channels that must be enabled (blue / green / red) and whether the
  laser must be on.
Canonical sequence: set up the samples -> prepare the slide -> enable the
required fluorescence channel(s) + laser -> image every required sample (and the
reference/control organelles).""",
    "flow_cytometry": """TECHNIQUE: Flow cytometry (FACS) for cell-cycle analysis.
Key factors to capture:
- the cell samples / treatments (drug conditions) that MUST be run.
- the DNA-content channel / stain used.
- the cell-cycle GATING / analysis that must be performed (gating into G0/G1, S,
  G2/M) to read out the distribution.
Canonical sequence: treat/select the cells -> prepare & stain the sample -> run
it on the cytometer -> gate and analyse the cell-cycle distribution.""",
    "western_blot": """TECHNIQUE: Western blot.
Key factors to capture:
- the cell lines AND treatment conditions that MUST be loaded as lanes (e.g.
  wildtype vs mutant, minus-ligand vs plus-ligand).
- the PRIMARY antibodies required, including any phospho-specific antibody needed
  to assess phosphorylation, AND a housekeeping/loading-control antibody.
- the matching secondary antibody/detector.
Canonical sequence: treat the cells -> prepare lysate -> load lanes + size marker
-> run the gel + transfer -> probe with the required primary + secondary
antibody(ies) -> detect/measure the bands.""",
}

SCHEMA = """{
  "rubric_id": "<given>",
  "technique": "<given>",
  "title": "<assignment title from the manual>",
  "summary": "<one sentence: the experimental goal>",
  "key_factors": [
    {
      "factor": "<short name, e.g. 'cell lines', 'treatments', 'primary antibodies', 'channels', 'gating', 'controls'>",
      "required": ["<each specific item the student must use, in the manual's wording>"],
      "why": "<why this is required to do the experiment correctly>"
    }
  ],
  "sequence": [
    {"step": "<one required step, in order>", "why": "<what it accomplishes>"}
  ]
}"""

SYSTEM_PROMPT = """You read a virtual cell-biology lab manual / answer key (a .docx
worksheet for the StarCellBio simulator) and extract the REQUIRED EXPERIMENTAL
PROCEDURE the student must carry out. You are NOT grading written answers or
numeric results — only the procedure: which cell lines, treatments, antibodies,
channels, gating and controls the student must use, and in what order.

Emit ONLY a single JSON object in exactly this schema (no prose, no fences):

%s

Rules:
- Capture only what the manual actually requires; do not invent items.
- Be specific and use the manual's own wording for `required` items so they can be
  matched against the simulator session.
- Always include necessary CONTROLS as key factors (e.g. reference organelles to
  image, a housekeeping loading-control antibody) — these are part of doing the
  experiment correctly.
- Keep `why` short and concrete.
- Output valid JSON only.""" % SCHEMA


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


def extract_one(client, model, rubric_id):
    technique, docx = TARGETS[rubric_id]
    src = DATA / docx
    if not src.exists():
        raise FileNotFoundError(src)
    doc_text = extract_docx(src)
    user = (
        f"rubric_id: {rubric_id}\n"
        f"technique: {technique}\n\n"
        f"{TECHNIQUE_GUIDANCE[technique]}\n\n"
        f"Manual:\n{doc_text}"
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=32768,
        ),
    )
    req = parse_json(resp.text)
    req["rubric_id"] = rubric_id
    req["technique"] = technique
    u = resp.usage_metadata
    if u is not None:
        print(f"  tokens: in={u.prompt_token_count} out={u.candidates_token_count}")
    return req


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="*", help="rubric_ids (default: all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set (.env or environment).", file=sys.stderr)
        return 1

    targets = args.targets or list(TARGETS)
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        print(f"ERROR: unknown target(s): {bad}. Known: {list(TARGETS)}", file=sys.stderr)
        return 1

    REQUIREMENTS.mkdir(exist_ok=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"model: {args.model}\n")
    for rubric_id in targets:
        print(f"[{rubric_id}] <- {TARGETS[rubric_id][1]}")
        try:
            req = extract_one(client, args.model, rubric_id)
        except Exception as e:  # noqa: BLE001 - surface per-doc failures, keep going
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        dest = REQUIREMENTS / f"{rubric_id}.json"
        dest.write_text(json.dumps(req, indent=2, ensure_ascii=False) + "\n")
        nk = len(req.get("key_factors", []))
        ns = len(req.get("sequence", []))
        print(f"  wrote {dest.relative_to(ROOT)}  ({nk} key factors, {ns} steps)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
