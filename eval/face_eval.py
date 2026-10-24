"""Face recognition evaluation on LFW (public benchmark; images stay local).

Protocol (no tuning on test data):
  * LFW View 1: DevTrain and DevTest have disjoint identities.
  * Thresholds and Platt calibration are fitted on DevTrain only, using every same-identity pair and
    200k different-identity pairs among DevTrain images (fine-grained false-accept rates).
  * `select`     : model shoot-out on the 1,000 DevTest pairs (accuracy with bootstrap CI, AUC, TAR@FAR).
  * `openset`    : 1:N identification like the app does it (gallery of enrolled people + strangers),
                   using identities that never appear in DevTrain, repeated over random draws.
  * `robustness` : DevTest pairs with the probe darkened / blurred / downscaled / JPEG-compressed,
                   with and without the quality gate.
Writes results/face_*.json and memora/ml/calibration/<embedder>.json.

Usage: python eval/face_eval.py select|openset|robustness|all [--embedders sface,w600k_r50]
"""
import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memora import config  # noqa: E402
from memora.ml import face, models  # noqa: E402

DATA = Path(os.environ.get("MEMORA_EVAL_DATA", ROOT / "eval_data"))
LFW = DATA / "lfw_home"
RESULTS = ROOT / "results"
ALL_EMBEDDERS = ["sface", "arcface_r34", "w600k_mbf", "w600k_r50"]
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------------------- data
def ensure_lfw():
    if not (LFW / "pairsDevTest.txt").exists():
        from sklearn.datasets import fetch_lfw_pairs
        for subset in ("train", "test"):
            fetch_lfw_pairs(subset=subset, funneled=True, data_home=str(DATA), download_if_missing=True)


def img_path(name, k):
    return LFW / "lfw_funneled" / name / f"{name}_{int(k):04d}.jpg"


def read_pairs(fname):
    pairs = []
    with open(LFW / fname) as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) == 3:
                pairs.append((img_path(p[0], p[1]), img_path(p[0], p[2]), 1))
            elif len(p) == 4:
                pairs.append((img_path(p[0], p[1]), img_path(p[2], p[3]), 0))
    return pairs


def identity(path):
    return Path(path).parent.name


# ------------------------------------------------------------------------- detection + embedding
_det_cache = {}


def detect_center(path, degrade=None):
    """Detect faces and keep the one nearest the image centre (LFW convention). Cached per image."""
    key = (str(path), degrade)
    if key not in _det_cache:
        img = cv2.imread(str(path))
        if degrade:
            img = DEGRADATIONS[degrade](img)
        faces = face.detect(img)
        if not faces:
            _det_cache[key] = None
        else:
            h, w = img.shape[:2]
            f = min(faces, key=lambda f: (f["box"][0] + f["box"][2] / 2 - w / 2) ** 2 + (f["box"][1] + f["box"][3] / 2 - h / 2) ** 2)
            crop = face.align(img, f["landmarks"])
            _det_cache[key] = (crop, face.quality(f, crop))
    return _det_cache[key]


def embed_all(paths, name, degrade=None):
    """Return {path: embedding or None}, plus mean embedding latency in ms."""
    emb_fn = face.embedder(name)
    out, times = {}, []
    for p in paths:
        d = detect_center(p, degrade)
        if d is None:
            out[str(p)] = None
            continue
        t = time.perf_counter()
        out[str(p)] = face.normalize(emb_fn(d[0]))
        times.append(time.perf_counter() - t)
    return out, 1000 * float(np.median(times)) if times else None


# ---------------------------------------------------------------------------------- statistics
def bootstrap_ci(values, n=2000):
    values = np.asarray(values, dtype=float)
    idx = RNG.integers(0, len(values), (n, len(values)))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0, c - h)), 4), round(float(min(1, c + h)), 4)]


def threshold_at_far(neg_sims, far):
    """Smallest threshold whose false-accept rate on negatives is <= far."""
    neg = np.sort(np.asarray(neg_sims))[::-1]
    k = int(np.floor(far * len(neg)))
    return float(neg[k]) + 1e-6 if k < len(neg) else float(neg[-1])


def fit_platt(pos, neg):
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([pos, neg])[:, None]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([np.full(len(pos), 0.5 / len(pos)), np.full(len(neg), 0.5 / len(neg))]) * len(y)
    lr = LogisticRegression(C=100).fit(x, y, sample_weight=w)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def ece(probs, labels, bins=10):
    probs, labels = np.asarray(probs), np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return round(float(total), 4)


# ------------------------------------------------------------------------------ calibration fit
def calibrate(name, train_emb):
    """Fit thresholds + Platt scaling on DevTrain identities only."""
    by_id = defaultdict(list)
    for p, e in train_emb.items():
        if e is not None:
            by_id[identity(p)].append(e)
    pos = [float(a @ b) for embs in by_id.values() for a, b in itertools.combinations(embs, 2)]
    ids = list(by_id)
    flat = [(i, e) for i, embs in by_id.items() for e in embs]
    neg = []
    while len(neg) < 200_000:
        a, b = RNG.integers(0, len(flat), 2)
        if flat[a][0] != flat[b][0]:
            neg.append(float(flat[a][1] @ flat[b][1]))
    a, b = fit_platt(np.array(pos), np.array(neg))
    calib = {
        "embedder": name,
        "fitted_on": "LFW DevTrain (identities disjoint from DevTest)",
        "n_positive_pairs": len(pos),
        "n_negative_pairs": len(neg),
        "n_identities": len(ids),
        # Accept as a match only above the 0.1% false-accept point; below the 1% point the face is
        # treated as unknown; in between is the 'unsure' band (shown as uncertain, or sent to the
        # Rekognition fallback in cascade mode).
        "accept_threshold": round(threshold_at_far(neg, 1e-3), 4),
        "reject_threshold": round(threshold_at_far(neg, 1e-2), 4),
        "verification_threshold": None,
        "platt_a": a,
        "platt_b": b,
    }
    # Threshold for any per-comparison false-accept rate, so the app can keep the *open-set* false-match
    # rate constant as the gallery grows (per-comparison FAR = target / gallery size).
    calib["far_curve"] = [[far, round(threshold_at_far(neg, far), 4)]
                          for far in (1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5)]
    # Best single threshold for pair verification (max accuracy on DevTrain-like balanced pairs).
    ths = np.linspace(-0.2, 1.0, 1201)
    pos_a, neg_a = np.array(pos), np.array(neg[:len(pos) * 5])
    acc = [0.5 * ((pos_a >= t).mean() + (neg_a < t).mean()) for t in ths]
    calib["verification_threshold"] = round(float(ths[int(np.argmax(acc))]), 4)
    return calib


# ------------------------------------------------------------------------------------- select
def run_select(embedders):
    train_pairs, test_pairs = read_pairs("pairsDevTrain.txt"), read_pairs("pairsDevTest.txt")
    train_imgs = sorted({str(p) for a, b, _ in train_pairs for p in (a, b)})
    test_imgs = sorted({str(p) for a, b, _ in test_pairs for p in (a, b)})
    t = time.perf_counter()
    for p in train_imgs + test_imgs:
        detect_center(p)
    det_ms = 1000 * (time.perf_counter() - t) / (len(train_imgs) + len(test_imgs))
    report = {"protocol": __doc__.strip().split("\n\n")[1], "detector": "YuNet 2023mar (OpenCV Zoo)",
              "detector_ms_per_image": round(det_ms, 1),
              "detection_failures": {"train": sum(_det_cache[(p, None)] is None for p in train_imgs),
                                     "test": sum(_det_cache[(p, None)] is None for p in test_imgs)},
              "n_test_pairs": len(test_pairs), "models": {}}
    for name in embedders:
        train_emb, _ = embed_all(train_imgs, name)
        calib = calibrate(name, train_emb)
        (face.CALIBRATION_DIR / f"{name}.json").write_text(json.dumps(calib, indent=2))
        test_emb, emb_ms = embed_all(test_imgs, name)
        sims, labels = [], []
        for a, b, y in test_pairs:
            ea, eb = test_emb[str(a)], test_emb[str(b)]
            sims.append(float(ea @ eb) if ea is not None and eb is not None else -1.0)  # failure -> "different"
            labels.append(y)
        sims, labels = np.array(sims), np.array(labels)
        correct = ((sims >= calib["verification_threshold"]) == labels).astype(float)
        from sklearn.metrics import roc_auc_score
        probs = [face.match_probability(s, calib) for s in sims]
        acc_ms = (sims[labels == 1] >= calib["accept_threshold"])
        report["models"][name] = {
            "license": models.FILES[name]["license"],
            "file_mb": round(models.path(name).stat().st_size / 2**20, 1),
            "embed_ms_median": round(emb_ms, 1),
            "test_accuracy": round(float(correct.mean()), 4),
            "test_accuracy_95ci": bootstrap_ci(correct),
            "test_roc_auc": round(float(roc_auc_score(labels, sims)), 4),
            "test_TAR_at_accept_threshold": round(float(acc_ms.mean()), 4),
            "test_FAR_at_accept_threshold": round(float((sims[labels == 0] >= calib["accept_threshold"]).mean()), 4),
            "test_ECE_calibrated_prob": ece(probs, labels),
            "thresholds_from_devtrain": {k: calib[k] for k in ("verification_threshold", "accept_threshold", "reject_threshold")},
        }
        print(name, json.dumps(report["models"][name]), flush=True)
    baseline = RESULTS / "face_baseline_original.json"
    if baseline.exists():
        report["original_memora_pipeline"] = json.loads(baseline.read_text())
    return report


# ------------------------------------------------------------------------------------ open set
def run_openset(embedders, gallery_sizes=(10, 50, 200), draws=20, max_probes_per_person=4):
    train_ids = {identity(p) for a, b, _ in read_pairs("pairsDevTrain.txt") for p in (a, b)}
    people = {}
    for d in sorted((LFW / "lfw_funneled").iterdir()):
        if d.is_dir() and d.name not in train_ids:
            people[d.name] = sorted(d.glob("*.jpg"))
    multi = [n for n, imgs in people.items() if len(imgs) >= 2]
    single = [n for n, imgs in people.items() if len(imgs) == 1]
    rng = np.random.default_rng(1)
    strangers = list(rng.choice(single, size=min(1000, len(single)), replace=False))
    max_g = max(gallery_sizes)
    enrol_pool = list(rng.choice(multi, size=min(len(multi), max_g * 2), replace=False))
    paths = [str(p) for n in enrol_pool for p in people[n][: 1 + max_probes_per_person]]
    paths += [str(people[n][0]) for n in strangers]
    report = {"setup": {"identities_excluded": "all DevTrain identities (used for thresholds)",
                        "enrolment": "1 photo per person (first image)",
                        "genuine_probes": f"up to {max_probes_per_person} other photos per enrolled person",
                        "strangers": f"{len(strangers)} people with a single LFW photo, never enrolled",
                        "draws": draws}, "models": {}}
    for name in embedders:
        calib = face.calibration(name)
        emb, _ = embed_all(paths, name)
        res = {"scaled": {}, "fixed": {}}
        for g in gallery_sizes:
            stats = {"scaled": defaultdict(list), "fixed": defaultdict(list)}
            for _ in range(draws):
                enrolled = list(rng.choice(enrol_pool, size=g, replace=False))
                gallery = [(n, emb[str(people[n][0])]) for n in enrolled if emb[str(people[n][0])] is not None]
                gen = [(n, emb[str(p)]) for n in enrolled for p in people[n][1: 1 + max_probes_per_person]]
                stranger_probes = [emb[str(people[n][0])] for n in rng.choice(strangers, size=len(gen), replace=False)]
                for policy in ("scaled", "fixed"):
                    scale = policy == "scaled"
                    outcomes = defaultdict(int)
                    for true_id, e in gen:
                        if e is None:
                            outcomes["gen_no_face"] += 1
                            continue
                        r = face.identify(e, gallery, name, scale_with_gallery=scale)
                        outcomes[f"gen_{r['decision']}_{'right' if r['person_id'] == true_id else 'wrong'}"] += 1
                    for e in stranger_probes:
                        if e is not None:
                            outcomes["str_" + face.identify(e, gallery, name, scale_with_gallery=scale)["decision"]] += 1
                    n_gen, n_str = len(gen), len(stranger_probes)
                    s = stats[policy]
                    s["DIR_correct_match"].append(outcomes["gen_match_right"] / n_gen)
                    s["genuine_wrong_person_match"].append(outcomes["gen_match_wrong"] / n_gen)
                    s["genuine_unsure"].append((outcomes["gen_unsure_right"] + outcomes["gen_unsure_wrong"]) / n_gen)
                    s["genuine_unknown"].append((outcomes["gen_unknown_right"] + outcomes["gen_unknown_wrong"] + outcomes["gen_no_face"]) / n_gen)
                    s["stranger_false_match"].append(outcomes["str_match"] / n_str)
                    s["stranger_unsure"].append(outcomes["str_unsure"] / n_str)
            for policy in ("scaled", "fixed"):
                res[policy][f"gallery_{g}"] = {k: {"mean": round(float(np.mean(v)), 4),
                                                   "95ci_over_draws": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
                                               for k, v in stats[policy].items()}
        report["models"][name] = res
        print(name, json.dumps({p: {g: v["stranger_false_match"]["mean"] for g, v in r.items()} for p, r in res.items()}), flush=True)
    report["setup"]["policies"] = {"scaled": f"per-comparison FAR = {config.FACE_OPEN_SET_FAR} / gallery size (the app)",
                                   "fixed": "fixed per-comparison FAR 0.1% (ablation)"}
    return report


# ---------------------------------------------------------------------------------- robustness
def _dark(img):
    return np.clip(((img / 255.0) ** 2.2) * 255 * 0.6, 0, 255).astype(np.uint8)


def _blur(img):
    return cv2.GaussianBlur(img, (0, 0), 2.5)


def _lowres(img):
    h, w = img.shape[:2]
    return cv2.resize(cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h))


def _jpeg(img):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 12])[1], cv2.IMREAD_COLOR)


DEGRADATIONS = {"dark": _dark, "blur": _blur, "lowres": _lowres, "jpeg": _jpeg}


def run_robustness(embedders):
    """Compare two ways of using the quality floors on degraded probes (DevTest; floors tuned on DevTrain):
      block   : refuse to recognise when the photo is dark/blurry (the original gate)
      explain : always recognise; a dark/blurry photo that doesn't match gets a "retake" prompt (the app)
    Hard issues (no face, low detector score, tiny face) are refused under both policies."""
    pairs = read_pairs("pairsDevTest.txt")
    report = {"setup": "DevTest pairs; the second image of each pair is degraded; threshold = accept_threshold; "
                       "quality floors from results/face_gate.json (tuned on DevTrain)",
              "floors": {"brightness": config.FACE_MIN_BRIGHTNESS, "sharpness": config.FACE_MIN_SHARPNESS,
                         "size_px": config.FACE_MIN_SIZE_PX, "det_score": config.FACE_MIN_DET_SCORE},
              "models": {}}
    for name in embedders:
        calib = face.calibration(name)
        clean, _ = embed_all({str(a) for a, _, _ in pairs}, name)
        res = {}
        for deg in [None] + list(DEGRADATIONS):
            probe, _ = embed_all({str(b) for _, b, _ in pairs}, name, deg)
            rows = []
            for a, b, y in pairs:
                ea, eb = clean[str(a)], probe[str(b)]
                d = detect_center(b, deg)
                hard = d is None or bool(face.hard_issues(d[1]))
                soft = d is not None and bool(face.soft_issues(d[1]))
                sim = float(ea @ eb) if ea is not None and eb is not None else -1.0
                rows.append((y, (sim >= calib["accept_threshold"]) and not hard, hard, soft))
            y, acc, hard, soft = np.array(rows, dtype=float).T
            gen, imp = y == 1, y == 0
            block_acc = acc * (1 - soft)
            failed = gen & (acc == 0) & (hard == 0)
            res[deg or "clean"] = {
                "hard_rejected": round(float(hard.mean()), 4),
                "soft_flagged_dark_or_blurry": round(float(soft.mean()), 4),
                "TAR_explain_policy": round(float(acc[gen].mean()), 4),
                "TAR_block_policy": round(float(block_acc[gen].mean()), 4),
                "FAR_explain_policy": round(float(acc[imp].mean()), 4),
                "FAR_block_policy": round(float(block_acc[imp].mean()), 4),
                "genuine_failures_given_retake_prompt": round(float(soft[failed].mean()), 4) if failed.any() else None,
            }
        report["models"][name] = res
        print(name, json.dumps(res), flush=True)
    return report


def _graded(kind, level):
    if kind == "dark":
        return lambda img: np.clip(((img / 255.0) ** (1 + 1.5 * level)) * 255 * (1 - 0.2 * level), 0, 255).astype(np.uint8)
    if kind == "blur":
        return lambda img: cv2.GaussianBlur(img, (0, 0), 1.0 + 2.0 * level)
    f = 2 + 2 * level
    return lambda img: cv2.resize(cv2.resize(img, (int(img.shape[1] / f), int(img.shape[0] / f)), interpolation=cv2.INTER_AREA),
                                  (img.shape[1], img.shape[0]))


def run_gate_tuning(embedders, min_tar=0.90, min_bin=40):
    """Tune the quality-gate floors on DevTrain (never DevTest): degrade the probe of each genuine pair
    at graded strengths, record the probe's quality metrics and whether the pair is still accepted, and
    set each floor to the highest metric value below which acceptance falls under `min_tar`. A floor of
    0 means the model copes with that degradation and the gate should not reject on it."""
    name = embedders[0]
    calib = face.calibration(name)
    pairs = [p for p in read_pairs("pairsDevTrain.txt") if p[2] == 1]
    ref, _ = embed_all({str(a) for a, _, _ in pairs}, name)
    emb_fn = face.embedder(name)
    rows = []
    for kind in ("dark", "blur", "lowres"):
        for level in (0, 1, 2, 3):
            fn = _graded(kind, level)
            for a, b, _ in pairs:
                img = fn(cv2.imread(str(b)))
                faces = face.detect(img)
                if not faces or ref[str(a)] is None:
                    continue
                f = faces[0]
                crop = face.align(img, f["landmarks"])
                q = face.quality(f, crop)
                accepted = float(ref[str(a)] @ face.normalize(emb_fn(crop))) >= calib["accept_threshold"]
                rows.append({**q, "accepted": accepted, "kind": kind, "level": level})
    # Each floor is derived only from the degradations that move that metric (a bright but blurred photo
    # must not count against brightness). Isotonic regression fits a monotone acceptance-vs-metric curve;
    # the floor is the lowest metric value where the fitted acceptance reaches `min_tar`.
    from sklearn.isotonic import IsotonicRegression
    family = {"brightness": ("dark",), "sharpness": ("blur", "lowres")}
    floors, curves = {}, {}
    for metric, kinds in family.items():
        fam = [r for r in rows if r["kind"] in kinds]
        x = np.array([r[metric] for r in fam], dtype=float)
        y = np.array([r["accepted"] for r in fam], dtype=float)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        grid = np.unique(np.quantile(x, np.linspace(0, 1, 201)))
        fitted = iso.predict(grid)
        ok = grid[fitted >= min_tar]
        floors[metric] = round(float(ok[0]), 2) if len(ok) and ok[0] > grid[0] else 0.0
        curves[metric] = [[round(float(g), 2), round(float(f), 3)] for g, f in zip(grid[::20], fitted[::20])]
    floors["size_px"] = config.FACE_MIN_SIZE_PX      # not simulated (downscaling keeps the box size): kept as configured
    floors["det_score"] = config.FACE_MIN_DET_SCORE  # detector confidence: kept as configured
    by_level = {f"{k}_{l}": round(float(np.mean([r["accepted"] for r in rows if r["kind"] == k and r["level"] == l])), 3)
                for k in ("dark", "blur", "lowres") for l in (0, 1, 2, 3)}
    report = {"embedder": name, "data": "LFW DevTrain genuine pairs, probe degraded at graded strengths (never DevTest)",
              "rule": f"floor = lowest metric value where isotonic-fitted genuine acceptance >= {min_tar}",
              "n_probes": len(rows), "acceptance_by_degradation": by_level, "tuned_floors": floors,
              "fitted_acceptance_curves": curves}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["select", "openset", "robustness", "gate", "all"])
    ap.add_argument("--embedders", default=",".join(ALL_EMBEDDERS))
    args = ap.parse_args()
    embedders = args.embedders.split(",")
    ensure_lfw()
    RESULTS.mkdir(exist_ok=True)
    face.CALIBRATION_DIR.mkdir(exist_ok=True)
    tasks = ["select", "openset", "gate", "robustness"] if args.task == "all" else [args.task]
    for task in tasks:
        report = {"select": run_select, "openset": run_openset, "robustness": run_robustness, "gate": run_gate_tuning}[task](embedders)
        report["config"] = {"embedders": embedders, "min_face_px": config.FACE_MIN_SIZE_PX,
                            "min_det_score": config.FACE_MIN_DET_SCORE, "min_sharpness": config.FACE_MIN_SHARPNESS,
                            "min_brightness": config.FACE_MIN_BRIGHTNESS}
        (RESULTS / f"face_{task}.json").write_text(json.dumps(report, indent=2))
        print("wrote", RESULTS / f"face_{task}.json")


if __name__ == "__main__":
    main()
