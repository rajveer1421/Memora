"""Build results/RESULTS.md from the results/*.json files written by the evaluation scripts.
Every metric quoted in the README comes from this file.     python eval/report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, digits=1):
    return "–" if v is None else f"{100 * v:.{digits}f}%"


def ci(v):
    return "–" if not v else f"{100 * v[0]:.1f}–{100 * v[1]:.1f}"


def face_section(out):
    sel, base, ops, rob = load("face_select.json"), load("face_baseline_original.json"), load("face_openset.json"), load("face_robustness.json")
    if not sel:
        return
    out += ["## Face recognition (LFW)", "",
            "Thresholds and calibration fitted on LFW DevTrain; all numbers below are on DevTest (1,000 pairs, "
            "identities disjoint from DevTrain). Scripts: `eval/face_eval.py`, `eval/baseline_original_pipeline.py`.", "",
            "| Pipeline | Accuracy (95% CI) | ROC-AUC | TAR / FAR at accept threshold | ms / face (embed) | Size | Licence |",
            "|---|---|---|---|---|---|---|"]
    if base:
        out.append(f"| Original Memora (DeepFace ArcFace-R34, Haar cascade, cosine 0.68) | {pct(base['accuracy_at_0.68'])} ({ci(base['accuracy_95ci'])}) "
                   f"| {base['roc_auc']:.3f} | {pct(base['TAR_at_0.68'])} / {pct(base['FAR_at_0.68'])} | {1000 * base['median_latency_s']:.0f} (detect+embed) | – | – |")
    names = {"arcface_r34": "Same ArcFace-R34 + YuNet + 5-point alignment", "sface": "**SFace + YuNet (default)**",
             "w600k_mbf": "InsightFace MobileFaceNet + YuNet", "w600k_r50": "InsightFace ResNet-50 + YuNet"}
    for k in ("arcface_r34", "sface", "w600k_mbf", "w600k_r50"):
        m = sel["models"].get(k)
        if m:
            out.append(f"| {names[k]} | {pct(m['test_accuracy'])} ({ci(m['test_accuracy_95ci'])}) | {m['test_roc_auc']:.3f} | "
                       f"{pct(m['test_TAR_at_accept_threshold'])} / {pct(m['test_FAR_at_accept_threshold'])} | {m['embed_ms_median']:.0f} "
                       f"| {m['file_mb']:.0f} MB | {m['license']} |")
    out += ["", f"Detector: {sel['detector']}, {sel['detector_ms_per_image']} ms/image; detection failures on DevTest: "
                f"{sel['detection_failures']['test']}. Calibrated-probability ECE (SFace): {sel['models']['sface']['test_ECE_calibrated_prob']}.", ""]
    if ops:
        out += ["### Open-set identification (how the app uses it)", "",
                f"{ops['setup']['enrolment']}; {ops['setup']['genuine_probes']}; {ops['setup']['strangers']}; "
                f"{ops['setup']['draws']} random draws per gallery size (95% range over draws).", "",
                "| Model | Threshold policy | Gallery | Correct match | Wrong person matched | Unsure (asks / cascades) | Stranger matched as someone | Stranger 'unsure' |",
                "|---|---|---|---|---|---|---|---|"]
        policies = ops["setup"].get("policies", {})
        for k, res in ops["models"].items():
            for policy, rows in (res.items() if "scaled" in res else [("fixed", res)]):
                label = "scaled with gallery (app)" if policy == "scaled" else "fixed (ablation)"
                for g, s in rows.items():
                    out.append(f"| {k} | {label} | {g.split('_')[1]} | {pct(s['DIR_correct_match']['mean'])} | {pct(s['genuine_wrong_person_match']['mean'])} | "
                               f"{pct(s['genuine_unsure']['mean'])} | {pct(s['stranger_false_match']['mean'], 2)} | {pct(s['stranger_unsure']['mean'])} |")
        if policies:
            out.append("\nPolicies: " + "; ".join(f"**{k}**: {v}" for k, v in policies.items()))
        out.append("")
    gate = load("face_gate.json")
    if gate:
        out += ["### Quality floors (tuned on DevTrain, never DevTest)", "",
                f"{gate['rule']}; {gate['n_probes']} degraded DevTrain probes. Tuned floors: {gate['tuned_floors']}. "
                f"Genuine acceptance by degradation strength: {gate['acceptance_by_degradation']}.", ""]
    if rob and "TAR_explain_policy" in next(iter(next(iter(rob["models"].values())).values())):
        out += ["### Robustness: block vs explain on low-quality photos", "",
                "**block** = refuse dark/blurry photos (the original gate); **explain** = always recognise and turn a "
                "non-match on a dark/blurry photo into a 'retake' prompt (the app). Probe degraded; DevTest pairs.", "",
                "| Model | Condition | Flagged dark/blurry | TAR explain | TAR block | FAR explain | FAR block | Failures given a retake prompt |",
                "|---|---|---|---|---|---|---|---|"]
        for k, res in rob["models"].items():
            for cond, s in res.items():
                out.append(f"| {k} | {cond} | {pct(s['soft_flagged_dark_or_blurry'])} | {pct(s['TAR_explain_policy'])} | {pct(s['TAR_block_policy'])} | "
                           f"{pct(s['FAR_explain_policy'], 2)} | {pct(s['FAR_block_policy'], 2)} | {pct(s['genuine_failures_given_retake_prompt'])} |")
        out.append("")


def intent_section(out):
    it = load("intent.json")
    if not it:
        return
    out += ["## Ask Memora intent classifier", "", f"{it['n_test']} held-out test questions ({it['data_note']}) "
            f"Script: `eval/intent_eval.py`.", "",
            "| Model | Accuracy (95% CI) | Macro-F1 (95% CI) | need-help recall | false need-help |", "|---|---|---|---|---|"]
    for k, s in it["models"].items():
        out.append(f"| {k} | {pct(s['accuracy'])} ({ci(s['accuracy_95ci'])}) | {s['macro_f1']:.3f} ({ci(s['macro_f1_95ci'])}) | "
                   f"{pct(s['need_help_recall'], 0)} | {s['false_need_help']} |")
    ours = it["models"]["minilm_logreg (ours)"]["accuracy_by_style"]
    out += ["", "Accuracy by phrasing style (ours): " + ", ".join(f"{k} {v['accuracy']:.0%} (n={v['n']})" for k, v in ours.items()), ""]


def distress_section(out):
    d = load("distress.json")
    if not d:
        return
    out += ["## Check-in distress / emotion classifier", "",
            f"Care-context set: {d['care_set']['n']} sentences ({d['care_set']['n_distress']} distress; {d['care_set']['note']}). "
            f"GoEmotions test: {d['goemotions_test']['n']} comments. Script: `eval/distress_eval.py`.", "",
            "| Model | Care: distress P / R | Care: distress F1 (95% CI) | Care: false alarms | Care: emotion acc. | GoEmotions: distress F1 |",
            "|---|---|---|---|---|---|"]
    for k, s in d["models"].items():
        c, g = s["care"], s["goemotions_test"]
        gt = f"{g['distress_f1']:.3f}" if isinstance(g, dict) else "n/a (in its training data)"
        out.append(f"| {k} | {c['distress_precision']:.2f} / {c['distress_recall']:.2f} | {c['distress_f1']:.3f} ({ci(c['distress_f1_95ci'])}) | "
                   f"{c['false_alarms']} | {pct(c['emotion_accuracy'])} | {gt} |")
    out.append("")
    for k, s in d["models"].items():
        tags = s["care"].get("distress_by_tag")
        if tags:
            out.append(f"- {k}: " + ", ".join(f"{t} {v['distress_accuracy']:.0%}" for t, v in tags.items()))
    out.append("")


def objects_section(out):
    o = load("objects.json")
    out += ["## Object finder", ""]
    if not o:
        out += ["Not evaluated yet: needs your own photos (see `eval/OBJECTS_PROTOCOL.md`). The thresholds in use are "
                "provisional defaults until then.", ""]
        return
    out += [f"{o['n_scenes']} scenes, {o['n_objects']} items; OWLv2 CPU time per frame (median): {o['owlv2_seconds_per_frame_cpu_median']} s.", "",
            "| Method | Precision | Recall | F1 (95% CI) | AP |", "|---|---|---|---|---|"]
    for k, s in o["methods"].items():
        out.append(f"| {k} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.3f} ({ci(s['f1_95ci'])}) | {s['average_precision']} |")
    if o.get("fires_on_other_instance"):
        out.append(f"\nFires on someone else's item of the same kind: {o['fires_on_other_instance']}")
    if o.get("vlm_human_rating"):
        out.append(f"\nLocation sentences (hand-rated): {o['vlm_human_rating']}")
    out.append("")


def main():
    out = ["# Memora evaluation results", "", "_Generated by `eval/report.py` from `results/*.json`. Do not edit by hand._", ""]
    for section in (face_section, intent_section, distress_section, objects_section):
        section(out)
    (R / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
