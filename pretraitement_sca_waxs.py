# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:32:12 2026

Prétraitement SAXS corrigé avec le Factor lu dans Sca WAXS

Logique :
- lire Factor dans : Sca_..._WAXS{...}
- corriger       : Sub_...
- formule        : I_corr = I(Sub SAXS) * Factor(Sca WAXS correspondant)

@author: bara.fall
"""

import os
import re
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def log(msg: str):
    print(msg)


# =========================================================
# 1) Types de fichiers
# =========================================================
def est_fichier_sca_waxs(filename: str) -> bool:
    return filename.lower().endswith(".dat") and filename.startswith("Sca_")


def est_fichier_sub_saxs(filename: str) -> bool:
    return filename.lower().endswith(".dat") and filename.startswith("Sub_")


# =========================================================
# 2) Extraction de l'identifiant
#    Exemples compatibles :
#    - KM225_2_00055
#    - km87_2_iso_00046
# =========================================================
def extraire_identifiant(filename: str) -> Optional[str]:
    patterns = [
        r"(km\d+_\d+_\d+)",       # ex: KM225_2_00055
        r"(km\d+_\d+_[^_]+_\d+)", # ex: km87_2_iso_00046
    ]

    for pat in patterns:
        m = re.search(pat, filename, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    return None


# =========================================================
# 3) Clés de correspondance
#    On match sur :
#    - identifiant
#    - {i,j}
# =========================================================
def extraire_cle_sca_waxs(filename: str) -> Optional[Tuple[str, int, int]]:
    """
    Exemple :
    Sca_4240_Sub_3764_km87_2_iso_00046_WAXS{95,0}_AzInt_Q_2630.dat
    -> ('km87_2_iso_00046', 95, 0)
    """
    ident = extraire_identifiant(filename)
    if ident is None:
        return None

    m = re.search(r"_WAXS\{(\d+),(\d+)\}", filename, re.IGNORECASE)
    if not m:
        return None

    return ident, int(m.group(1)), int(m.group(2))


def extraire_cle_sub_saxs(filename: str) -> Optional[Tuple[str, int, int]]:
    """
    Exemple :
    Sub_3764_km87_2_iso_00046_WAXS{95,0}_AzInt_Px_2630.dat
    -> ('km87_2_iso_00046', 95, 0)
    """
    ident = extraire_identifiant(filename)
    if ident is None:
        return None

    m = re.search(r"_WAXS\{(\d+),(\d+)\}", filename, re.IGNORECASE)
    if not m:
        m = re.search(r"\{(\d+),(\d+)\}", filename, re.IGNORECASE)

    if not m:
        return None

    return ident, int(m.group(1)), int(m.group(2))


# =========================================================
# 4) Lecture du Factor dans Sca WAXS
# =========================================================
RE_FACTOR = re.compile(
    r"Factor\s*=\s*([+-]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE
)


def lire_factor_depuis_sca(path: str) -> Optional[float]:
    encodings = ["utf-8", "latin-1", "utf-16"]

    for enc in encodings:
        facteurs = []
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for line in f:
                    m = RE_FACTOR.search(line)
                    if not m:
                        continue

                    txt = m.group(1).replace(",", ".").strip()
                    try:
                        val = float(txt)
                    except Exception:
                        continue

                    if np.isfinite(val):
                        facteurs.append(val)

            if facteurs:
                return facteurs[-1]

        except Exception:
            continue

    return None


# =========================================================
# 5) Détection début données
# =========================================================
RE_DATA_START = re.compile(
    r"^\s*[-+]?(?:\d+\.?\d*|\.\d+|nan)\s+",
    re.IGNORECASE
)


def trouver_debut_donnees(path: str) -> Optional[int]:
    encodings = ["utf-8", "latin-1", "utf-16"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for i, line in enumerate(f):
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if RE_DATA_START.match(s):
                        return i
        except Exception:
            continue

    return None


# =========================================================
# 6) Lecture des colonnes Sub SAXS
# =========================================================
def lire_colonnes_sub(path: str):
    start = trouver_debut_donnees(path)
    if start is None:
        raise ValueError(f"Données non trouvées : {path}")

    df = pd.read_csv(
        path,
        skiprows=start,
        sep=r"\s+|\t+",
        engine="python",
        header=None,
        dtype=str,
        on_bad_lines="skip",
    )

    if df.shape[1] < 2:
        raise ValueError(f"Moins de 2 colonnes numériques dans : {path}")

    q = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    I = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)

    sig = None
    if df.shape[1] >= 3:
        sig = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(float)

    mask = np.isfinite(q)
    q = q[mask]
    I = I[mask]
    if sig is not None:
        sig = sig[mask]

    return q, I, sig


# =========================================================
# 7) Écriture du fichier corrigé
# =========================================================
def ecrire_corrige_depuis_sub_saxs(
    path_sub: str,
    path_out: str,
    factor: float,
    multiplier_sig: bool = False,
):
    q, I, sig = lire_colonnes_sub(path_sub)

    I_corr = np.where(np.isnan(I), np.nan, I * factor)

    sig_corr = None
    if sig is not None:
        if multiplier_sig:
            sig_corr = np.where(np.isnan(sig), np.nan, sig * factor)
        else:
            sig_corr = sig.copy()

    with open(path_out, "w", encoding="utf-8") as f:
        f.write(f"# Fichier Sub SAXS source : {os.path.basename(path_sub)}\n")
        f.write(f"# Factor applique (lu dans Sca WAXS) : {factor}\n")
        f.write(f"# Formule :  I_corr = I(Sub SAXS) * Factor(Sca WAXS)\n")

        if sig_corr is None:
            f.write("# q(A-1)\tI_corr\n")
            for a, b in zip(q, I_corr):
                b_str = "nan" if np.isnan(b) else f"{b:.10g}"
                f.write(f"{a:.10g}\t{b_str}\n")
        else:
            f.write("# q(A-1)\tI_corr\tSig\n")
            for a, b, c in zip(q, I_corr, sig_corr):
                b_str = "nan" if np.isnan(b) else f"{b:.10g}"
                c_str = "nan" if np.isnan(c) else f"{c:.10g}"
                f.write(f"{a:.10g}\t{b_str}\t{c_str}\n")


# =========================================================
# 8) Nom de sortie
# =========================================================
def construire_nom_sortie_depuis_sub_saxs(sub_name: str, factor: float) -> str:
    base, ext = os.path.splitext(sub_name)
    return f"{base}_corrFactor_{factor:.6f}{ext}"


# =========================================================
# 9) Dossier de sortie
# =========================================================
def construire_dossier_sortie(dossier_source: str) -> str:
    nom = os.path.basename(os.path.normpath(dossier_source))
    return os.path.join(dossier_source, f"{nom}_cor_saxs")


# =========================================================
# 10) Pipeline principal
# =========================================================
def traiter(dossier: str, multiplier_sig: bool = False, debug: bool = False):
    if not os.path.isdir(dossier):
        raise ValueError(f"Dossier introuvable : {dossier}")

    out_dir = ensure_dir(construire_dossier_sortie(dossier))

    fichiers_sca = sorted([f for f in os.listdir(dossier) if est_fichier_sca_waxs(f)])
    fichiers_sub = sorted([f for f in os.listdir(dossier) if est_fichier_sub_saxs(f)])

    log(f"{len(fichiers_sca)} fichiers Sca WAXS trouvés")
    log(f"{len(fichiers_sub)} fichiers Sub SAXS trouvés")
    log(f"Dossier de sortie : {out_dir}")

    sca_par_cle: Dict[Tuple[str, int, int], str] = {}
    n_sca_sans_cle = 0

    for fsca in fichiers_sca:
        cle = extraire_cle_sca_waxs(fsca)
        if cle is None:
            log(f"⚠ Clé Sca WAXS introuvable : {fsca}")
            n_sca_sans_cle += 1
            continue

        if debug:
            log(f"SCA KEY -> {cle} : {fsca}")

        sca_par_cle[cle] = fsca

    n_ok = 0
    n_skip = 0

    for fsub in fichiers_sub:
        cle = extraire_cle_sub_saxs(fsub)
        if cle is None:
            log(f"⚠ Clé Sub SAXS introuvable : {fsub}")
            n_skip += 1
            continue

        if debug:
            log(f"SUB KEY -> {cle} : {fsub}")

        fsca = sca_par_cle.get(cle)
        if fsca is None:
            log(f"⚠ Aucun Sca WAXS correspondant pour : {fsub}")
            n_skip += 1
            continue

        path_sca = os.path.join(dossier, fsca)
        path_sub = os.path.join(dossier, fsub)

        factor = lire_factor_depuis_sca(path_sca)
        if factor is None:
            log(f"⚠ Pas de factor dans : {fsca}")
            n_skip += 1
            continue

        out_name = construire_nom_sortie_depuis_sub_saxs(fsub, factor)
        out_path = os.path.join(out_dir, out_name)

        k = 1
        while os.path.exists(out_path):
            base, ext = os.path.splitext(out_name)
            out_path = os.path.join(out_dir, f"{base}_{k}{ext}")
            k += 1

        try:
            ecrire_corrige_depuis_sub_saxs(
                path_sub=path_sub,
                path_out=out_path,
                factor=factor,
                multiplier_sig=multiplier_sig,
            )
            log(f"OK -> {os.path.basename(out_path)}")
            n_ok += 1
        except Exception as e:
            log(f"Erreur {fsub} : {e}")
            n_skip += 1

    log("")
    log(f"Sca WAXS sans clé : {n_sca_sans_cle}")
    log(f"Terminé : {n_ok} fichier(s) corrigé(s), {n_skip} ignoré(s)")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    dossier = r"Z:\Public\traitement DATA SWING\test_export\Sca_km205_1_WAXS_"
    traiter(dossier, multiplier_sig=False, debug=False)