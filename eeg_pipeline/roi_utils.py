"""
ROI utilities: centralized channel name canonicalization and ROI mapping.

This module avoids duplicated implementations across scripts. It uses the
configured ROI regex patterns when available, with robust channel name
canonicalization to handle vendor-specific labels (e.g., "EEG Fp1-Ref").
"""

from __future__ import annotations

import re
from typing import Dict, List

import mne

try:  # prefer package-relative import to avoid ambiguity
    from .config_loader import load_config  # type: ignore
except Exception:  # pragma: no cover - when executed as a script
    from config_loader import load_config

_config = load_config()


def canonicalize_ch_name(ch: str) -> str:
    """Return a canonical channel name suitable for regex/prefix matching.

    - Strip whitespace
    - Remove leading 'EEG' and separators
    - Drop any suffix after '-' or '/' (e.g., references)
    - Remove spaces
    - Remove common trailing reference tokens (Ref, LE, RE, M1, M2, A1, A2, AVG/AVE)
    """
    s = ch.strip()
    try:
        s = re.sub(r"^(EEG[ \-_]*)", "", s, flags=re.IGNORECASE)
        s = re.split(r"[-/]", s)[0]
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"(Ref|LE|RE|M1|M2|A1|A2|AVG|AVE)$", "", s, flags=re.IGNORECASE)
    except Exception:
        return ch
    return s


def _roi_definitions() -> Dict[str, List[str]]:
    return dict(
        _config.get(
            "time_frequency_analysis.rois",
            {
                # --- Peri-Rolandic / primary sensorimotor (pain-critical) ---
                # Hand/forearm representation is best approximated around C4/CP4 on EasyCap M1.
                "S1_R":       [r"^(C4|C6|CP4|CP6)$"],              # contralateral primary somatosensory (right)
                "S1_L":       [r"^(C3|C5|CP3|CP5)$"],              # ipsilateral (left) control
                "M1_R":       [r"^(FC4|C4|CP4)$"],                 # contralateral primary motor/premotor proxy
                "M1_L":       [r"^(FC3|C3|CP3)$"],

                # --- Secondary somatosensory / insula proxies (EEG-accessible scalp correlates) ---
                "S2_R":       [r"^(T8|TP8|FT8|CP6)$"],             # contralateral S2/insula proxy (right)
                "S2_L":       [r"^(T7|TP7|FT7|CP5)$"],

                # --- Midline cingulate / ACC proxies (pain intensity & control) ---
                "MCC_ACC":    [r"^(FCz|Cz|Fz)$"],

                # --- Posterior parietal integration (often tracks ratings/decision) ---
                "Parietal_R": [r"^(P2|P4|P6|PO4|PO8)$"],
                "Parietal_L": [r"^(P1|P3|P5|PO3|PO7)$"],

                # --- Frontal control/anticipation (useful for ramp & expectancy) ---
                "Frontal_Mid":[r"^(AFz|Fz|F2|F4)$"],
                "Frontal":    [r"^(Fpz|Fp[12]|AFz|AF[3-8]|Fz|F[1-8])$"],  # keep your broad set

                # --- Visual/occipital control (isolate visual or general arousal confounds) ---
                "Occipital":  [r"^(Oz|O[12]|POz|PO[3-8])$"],

                # --- Your original broad regions (kept for compatibility) ---
                "Central":    [r"^(Cz|C[1-6])$"],
                "Parietal":   [r"^(Pz|P[1-8])$"],
                "Temporal":   [r"^(T7|T8|TP7|TP8|FT7|FT8)$"],

                # --- A compact, pain-focused composite for diagonals (nice for group summaries) ---
                "PeriRolandic_R":[r"^(FC4|C4|C6|CP4|CP6)$"],       # union S1_R + M1_R
                "PeriRolandic_L":[r"^(FC3|C3|C5|CP3|CP5)$"],

                # --- Your previous "Sensorimotor" preserved, but consider using the lateralized ones above ---
                "Sensorimotor":[r"^(FC[234]|FCz)$", r"^(C[234]|Cz)$", r"^(CP[234]|CPz)$"],
            },
        )
    )



def find_roi_channels(info: mne.Info, patterns: List[str]) -> List[str]:
    chs = info["ch_names"]
    out: List[str] = []
    canon_map = {ch: canonicalize_ch_name(ch) for ch in chs}
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for ch in chs:
            cn = canon_map.get(ch, ch)
            if rx.match(ch) or rx.match(cn):
                out.append(ch)
    # Preserve original order and deduplicate
    seen = set()
    ordered: List[str] = []
    for ch in chs:
        if ch in out and ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return ordered


def build_rois_from_info(info: mne.Info) -> Dict[str, List[str]]:
    rois = {}
    roi_defs = _roi_definitions()
    for roi, pats in roi_defs.items():
        chans = find_roi_channels(info, pats)
        if chans:
            rois[roi] = chans
    return rois
