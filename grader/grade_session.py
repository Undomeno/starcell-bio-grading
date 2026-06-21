"""Procedural grader entrypoint: session(s) -> diagnosis report.

Reads one or more StarCellBio session dumps, extracts what the student actually
ran, loads the matching docx-derived requirements, asks the LLM to diagnose
whether the procedure is correct, and prints a Markdown report per session.

    python grader/grade_session.py session1.json session2.json ...
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from session_procedure import parse_scb, extract_procedure  # noqa: E402
from diagnose import diagnose  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REQ_DIR = ROOT / "requirements"

# session assignmentName -> requirement_id (the requirements file basename).
ASSIGNMENT_MAP = {
    "Entry - Microscopy": "entry_microscopy",
    "Entry - Flow Cytometry": "entry_flow_cytometry",
    "Entry - Western Blot": "entry_western_blot",
}


def resolve_requirement_id(name):
    """Map a session's assignmentName to its requirement_id (or None)."""
    return ASSIGNMENT_MAP.get((name or "").strip())


def render(report):
    out = []
    out.append(f"# Procedural review — {report['assignment']}")
    out.append(f"_course {report['course']} · technique: {report['technique']}_\n")
    d = report["diagnosis"]
    verdict = "✅ On track" if d.get("on_track") else "⚠️ Needs correction"
    out.append(f"**{verdict}**\n")
    out.append(d.get("summary", "").strip() + "\n")

    issues = d.get("issues") or []
    if issues:
        out.append("## Issues")
        for i in issues:
            out.append(f"- **{i.get('factor','')}** — {i.get('problem','')}")
            exp = i.get("explanation", "").strip()
            if exp:
                out.append(f"  - {exp}")
        out.append("")
    else:
        out.append("No procedural issues found.\n")

    did = d.get("did_well") or []
    if did:
        out.append("## Done correctly")
        for x in did:
            out.append(f"- {x}")
        out.append("")
    return "\n".join(out)


def grade_one(path):
    """Grade a single session file and print its report. Returns 0 ok, 1 on error."""
    session = parse_scb(path)
    procedure = extract_procedure(session)

    requirement_id = resolve_requirement_id(session.get("name"))
    req_path = REQ_DIR / f"{requirement_id}.json"
    if not req_path.exists():
        print(f"ERROR [{path}]: {req_path} not found. Run scripts/extract_requirements.py first.",
              file=sys.stderr)
        return 1
    requirements = json.loads(req_path.read_text())
    technique = requirements["technique"]
    executed = procedure["techniques"].get(technique, [])
    diagnosis = diagnose(requirements, executed, procedure["template_vocab"])

    print(render({
        "assignment": procedure["assignment"],
        "course": procedure["course"],
        "technique": technique,
        "diagnosis": diagnosis,
    }))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="+", help="paths to StudentAssignment.data dumps")
    args = ap.parse_args()

    rc = 0
    for i, path in enumerate(args.sessions):
        if i:
            print("\n" + "=" * 72 + "\n")
        rc |= grade_one(path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
