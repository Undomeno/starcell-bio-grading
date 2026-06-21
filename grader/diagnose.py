"""Diagnose whether a student's EXECUTED procedure matches the REQUIRED one.

A single LLM call. Given (a) the docx-derived requirements (key factors +
sequence), (b) the student's executed procedure from the session, and (c) the
session's template vocabulary (to bridge the manual's wording and the
simulator's machine names), it judges whether the student is doing the right
thing and points out what's missing or wrong.
"""
import json
import os
import re

from google import genai
from google.genai import types

DEFAULT_MODEL = os.getenv("DIAGNOSE_MODEL", "gemini-3.1-flash-lite")

SYSTEM_PROMPT = """You are a teaching assistant reviewing a student's virtual
cell-biology lab session on the StarCellBio simulator. You are given:

1. REQUIREMENTS — the procedure this experiment requires (key factors the student
   must use, and the expected sequence of steps), extracted from the lab manual.
2. EXECUTED — what the student ACTUALLY ran in the simulator (only executed runs
   are included; things merely set up but never run are excluded).
3. VOCABULARY — maps the simulator's machine names to readable names, so you can
   tell that e.g. the manual's "minus-ligand / plus-ligand" corresponds to the
   simulator drugs "Growth Media" / "Growth Media + Ligand", or that an antibody
   in the session is the manual's phospho-specific antibody.

Your job: judge whether the student is performing the experiment CORRECTLY at the
procedural level. Match by MEANING, not exact strings — use the vocabulary to map
manual wording onto what the student ran. A requirement counts as satisfied only
if the student actually executed it.

Focus on procedure only (which cell lines, treatments, antibodies, channels,
gating, controls, and the order of steps). Do NOT grade written answers or numeric
results. Do NOT assign any score or points.

Be specific and fair: do not flag something as missing if the vocabulary shows the
student did an equivalent thing under a different name. Flag genuine gaps (a
required condition never run, a required antibody never used, a required control
omitted, required channels/gating missing, steps done out of a necessary order).

IMPORTANT — observability: only flag a requirement as missing if it is the kind of
thing the EXECUTED data could actually record. The simulator session captures
selected cell lines/treatments, loaded lanes, antibodies probed, fluorescence
channels imaged, gating, and prep flags — but it does NOT capture some manual
steps such as adding a DNA stain (e.g. Propidium iodide), pipetting, or incubation
times. If a required step has no corresponding field in the EXECUTED data, treat it
as not-observable and do NOT report it as a student error (you may note it
separately only if clearly relevant). Judge the student only on what the session
can actually show.

Emit ONLY a JSON object in this schema (no prose, no fences):
{
  "on_track": <true|false>,        // is the student doing the experiment correctly so far?
  "summary": "<2-3 sentence plain-language verdict>",
  "issues": [
    {
      "factor": "<which requirement/step, e.g. 'primary antibodies', 'gating', 'sequence'>",
      "problem": "<short: what is missing or wrong>",
      "explanation": "<why it matters / what the student should have done, grounded in what they did run>"
    }
  ],
  "did_well": ["<required things the student correctly executed>"]
}
If there are no issues, return an empty "issues" list and on_track=true."""


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # The model occasionally appends trailing text after the JSON object; decode
    # the first object and ignore the rest.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model response: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def diagnose(requirements, executed_runs, vocab, model=DEFAULT_MODEL, client=None):
    """Return the diagnosis dict for one technique."""
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        client = genai.Client(api_key=api_key)

    user = (
        "REQUIREMENTS:\n"
        + json.dumps(requirements, indent=2, ensure_ascii=False)
        + "\n\nEXECUTED (what the student actually ran):\n"
        + json.dumps(executed_runs, indent=2, ensure_ascii=False)
        + "\n\nVOCABULARY (simulator id/name -> readable name):\n"
        + json.dumps(vocab, indent=2, ensure_ascii=False)
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=8192,
        ),
    )
    return _parse_json(resp.text)
