"""Extract data from StarCellBio db.

This cleans up the student data: which samples were imaged/run/loaded, which
fluorescence channels were on, which antibodies were probed, whether
gating/analysis happened, plus the prep flags that mark a run as carried out.

Usage:
    python grader/session_procedure.py data/sessions/entry_western_blot__id603.json
"""
import ast
import json
import re
import sys


# parsing
def parse_scb(path):
    """StudentAssignment.data is a Python-2 repr() string, not JSON.

    Single quotes, u'...' prefixes, and Py2 long literals like 174L. Strip the
    trailing L from integers, then literal_eval. (Mirrors poc/grade.py.)
    """
    raw = open(path, encoding="utf-8", errors="replace").read()
    return ast.literal_eval(re.sub(r"(?<=\d)L\b", "", raw))


# replace ids to human readable names
def template_vocab(session):
    t = session.get("template", {}) or {}

    def names(d):
        out = {}
        for k, v in (d or {}).items():
            if k == "order":
                continue
            if isinstance(v, dict) and "name" in v:
                out[k] = v["name"]
        return out

    return {
        "drugs": names(t.get("drugs")),
        "cell_lines": names(t.get("cell_lines")),
        "primary_anti_body": names(t.get("primary_anti_body")),
        "secondary_anti_body": names(t.get("secondary_anti_body")),
    }


def _ct_index(session):
    """Map cell_treatment id -> {strain, protocol, drugs[]} across all experiments.

    id == name == the key used in is_cell_treatment_enabled and lanes' cell_treatment_id.
    """
    index = {}
    for exp in session.get("experiments", {}).get("list", []):
        for ct in exp.get("cell_treatment_list", {}).get("list", []):
            drugs = []
            for tr in ct.get("treatment_list", {}).get("list", []):
                for dr in tr.get("drug_list", {}).get("list", []):
                    name = dr.get("drug_name")
                    if name and name not in drugs:
                        drugs.append(name)
            index[ct.get("id")] = {
                "strain": ct.get("strain"),
                "protocol": (ct.get("protocol") or "").strip(),
                "drugs": drugs,
            }
    return index


def _enabled_samples(run, ct_index):
    """Resolve the `checked` cell-treatment ids of a run to readable conditions."""
    out = []
    for ct_id, state in (run.get("is_cell_treatment_enabled") or {}).items():
        if state == "checked" and ct_id in ct_index:
            out.append(ct_index[ct_id])
    return out


# ── per-technique extractors ─────────────────────────────────────────────────
def extract_microscopy(session, ct_index, vocab):
    runs = []
    for exp in session.get("experiments", {}).get("list", []):
        for r in exp.get("microscopy_list", {}).get("list", []):
            # Captured images live in lanes_list: each lane = one imaged sample in
            # one fluorescence channel (lens_map.if_type). The run-level
            # *_enabled toggles are only the current editor state, not what was
            # actually captured, so derive both samples and channels from lanes.
            lanes = r.get("lanes_list", {}).get("list", [])
            channels, imaged, seen = [], [], set()
            for lane in lanes:
                ch = (lane.get("lens_map", {}) or {}).get("if_type")
                if ch and ch not in channels:
                    channels.append(ch)
                ct_id = lane.get("cell_treatment_id")
                if ct_id in ct_index and ct_id not in seen:
                    seen.add(ct_id)
                    imaged.append(ct_index[ct_id])
            if not (imaged or channels):
                continue  # nothing actually captured in this tab
            runs.append({
                "run": r.get("name"),
                "created_at": r.get("created_at"),
                "slide_prepared": bool(r.get("slide_prepared")),
                "samples_finished": bool(r.get("samples_finished")),
                "laser_on": bool(r.get("laser_on")),
                "channels_imaged": channels,
                "samples_imaged": imaged,
            })
    return runs


def extract_flow_cytometry(session, ct_index, vocab):
    runs = []
    for exp in session.get("experiments", {}).get("list", []):
        for r in exp.get("facs_list", {}).get("list", []):
            samples = _enabled_samples(r, ct_index)
            lanes = r.get("lanes_list", {}).get("list", [])
            gated = any(
                lane.get("bisector_gate_created")
                or (lane.get("canvas_metadata_analysis", {}) or {}).get("ranges")
                for lane in lanes
            )
            if not (samples or lanes):
                continue
            runs.append({
                "run": r.get("name"),
                "created_at": r.get("created_at"),
                "sample_prepared": bool(r.get("sample_prepared")),
                "sample_analysis": bool(r.get("sample_analysis")),
                "gate_count": r.get("gate_count"),
                "gating_performed": bool(gated),
                "samples_run": samples,
            })
    return runs


def extract_western_blot(session, ct_index, vocab):
    pab = vocab["primary_anti_body"]
    runs = []
    for exp in session.get("experiments", {}).get("list", []):
        for r in exp.get("western_blot_list", {}).get("list", []):
            # lanes loaded -> conditions
            loaded = []
            for lane in r.get("lanes_list", {}).get("list", []):
                ct_id = lane.get("cell_treatment_id")
                if ct_id in ct_index:
                    loaded.append(ct_index[ct_id])
            # antibodies probed (per gel)
            antibodies = []
            for gel in r.get("gel_list", {}).get("list", []):
                primary = gel.get("primary_anti_body")
                antibodies.append({
                    "primary": pab.get(primary, primary),
                    "secondary": gel.get("secondary_anti_body"),
                })
            if not (loaded or antibodies):
                continue
            runs.append({
                "run": r.get("name"),
                "created_at": r.get("created_at"),
                "lysate_prepared": bool(r.get("lysate_prepared")),
                "wells_loaded": bool(r.get("wells_loaded")),
                "marker_loaded": bool(r.get("marker_loaded")),
                "is_transfered": bool(r.get("is_transfered")),
                "lanes_loaded": loaded,
                "antibodies_probed": antibodies,
            })
    return runs


TECHNIQUES = {
    "microscopy": ("microscopy_list", extract_microscopy),
    "flow_cytometry": ("facs_list", extract_flow_cytometry),
    "western_blot": ("western_blot_list", extract_western_blot),
}


def techniques_present(session):
    """Which techniques the student actually has runs for."""
    present = []
    for tech, (list_key, _) in TECHNIQUES.items():
        for exp in session.get("experiments", {}).get("list", []):
            if exp.get(list_key, {}).get("list"):
                present.append(tech)
                break
    return present


def extract_procedure(session):
    ct_index = _ct_index(session)
    vocab = template_vocab(session)
    out = {
        "course": session.get("course"),
        "assignment": session.get("name"),
        "techniques": {},
        "template_vocab": vocab,
    }
    for tech in techniques_present(session):
        extractor = TECHNIQUES[tech][1]
        out["techniques"][tech] = extractor(session, ct_index, vocab)
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    session = parse_scb(sys.argv[1])
    print(json.dumps(extract_procedure(session), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
