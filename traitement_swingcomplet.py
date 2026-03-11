# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:23:26 2026

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Ven  6 13:17:29 2026

@author: bara.fall
"""

"""
GUI Tkinter - Pipeline complet SWING SAXS/WAXS + Fichier Température
+ sélection manuelle de fichiers .dat pour tracer un waterfall
"""

import os
import re
import shutil
import threading
import queue
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =================================================
# 0) Utils dossiers / PNG
# =================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def copy_png_to_base(png_path: str, base_dir: str, log_fn=print):
    if png_path and os.path.isfile(png_path):
        ensure_dir(base_dir)
        dst = os.path.join(base_dir, os.path.basename(png_path))
        try:
            shutil.copy2(png_path, dst)
            log_fn(f"📌 Copie PNG -> {dst}")
        except Exception as e:
            log_fn(f"⚠️ Copie PNG impossible: {e}")


# =================================================
# ToolTip
# =================================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=5
        )
        label.pack()

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


# =================================================
# 1) Détecter début données numériques dans les .DAT
# =================================================
_numline = re.compile(r"^\s*[-+]?\d")


def trouver_debut_donnees(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if _numline.match(line):
                return idx
    return None


# =================================================
# 2) Lire Linkam : X (s) et Y (°C)
# =================================================
def lire_xy_linkam(path_txt: str):
    encodings = ["utf-8", "utf-16", "latin-1"]
    lines = None

    for enc in encodings:
        try:
            with open(path_txt, "r", encoding=enc, errors="ignore") as f:
                lines = f.readlines()
            break
        except Exception:
            continue

    if lines is None:
        return np.array([]), np.array([])

    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower().strip()
        if low.startswith("new index") and ("x" in low) and ("y" in low):
            header_idx = i
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            if re.match(r"^\s*\d+\t\d+\t", line):
                header_idx = i - 1
                break

    if header_idx is None:
        return np.array([]), np.array([])

    data_str = "".join(lines[header_idx + 1:]).replace("\xa0", " ")
    buf = StringIO(data_str)

    df = pd.read_csv(
        buf,
        sep="\t",
        header=None,
        engine="python",
        usecols=[2, 3],
        names=["X", "Y"],
        decimal=",",
        thousands=" ",
        on_bad_lines="skip",
    )

    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna()

    if df.empty:
        return np.array([]), np.array([])

    X = df["X"].to_numpy(float)
    Y = df["Y"].to_numpy(float)
    order = np.argsort(X)
    return X[order], Y[order]


# =================================================
# 3) Construire T(img)
# =================================================
def construire_T_of_img(X, Y, dt_image_s, t0_s, methode="interp"):
    if len(X) == 0:
        return lambda _: float("nan")

    def f(img: int) -> float:
        t = t0_s + img * dt_image_s

        if t <= X[0]:
            return float(Y[0])
        if t >= X[-1]:
            return float(Y[-1])

        if methode == "nearest":
            j = int(np.argmin(np.abs(X - t)))
            return float(Y[j])

        return float(np.interp(t, X, Y))

    return f


# =================================================
# 4) Helpers ordre / lecture courbes
# =================================================
def extraire_temperature_depuis_nom(filename: str):
    motif_T = re.compile(r"_([-\d]+(?:\.\d+)?)(?:_\d+)?\.dat$", re.IGNORECASE)
    m = motif_T.search(filename)
    if not m:
        return None
    return float(m.group(1))


def extraire_image_depuis_nom(filename: str):
    m = re.search(r"_(\d{5})_", filename)
    if not m:
        return None
    return int(m.group(1))


def lire_q_I_depuis_dat(path: str):
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", header=None)
    q = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    I = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    mask = np.isfinite(q) & np.isfinite(I)
    q = q[mask]
    I = I[mask]
    return q, I


# =================================================
# 5) Waterfall dossier
# =================================================
def waterfall_plot_like_user_script(
    dossier_renommes,
    titre,
    chemin_png,
    pas=10,
    y_log=True,
    decal_x=0.01,
    decal_log10=0.18,
    lw=1.2,
    alpha=0.95,
    dpi_save=200,
    label_dx=0.035,
    label_y_mult=1.09,
    reverse=False,
    log_fn=lambda s: None,
):
    if not os.path.isdir(dossier_renommes):
        log_fn(f"⚠️ Dossier introuvable : {dossier_renommes}")
        return

    fichiers = sorted(
        [f for f in os.listdir(dossier_renommes) if f.lower().endswith(".dat")]
    )
    log_fn(f"[Plot] Fichiers .dat trouvés : {len(fichiers)}")

    courbes = []
    for f in fichiers:
        T = extraire_temperature_depuis_nom(f)
        img = extraire_image_depuis_nom(f)
        if T is None:
            continue
        if img is None:
            img = 0

        path = os.path.join(dossier_renommes, f)

        try:
            q, I = lire_q_I_depuis_dat(path)
        except Exception:
            continue

        if len(q) == 0:
            continue

        courbes.append((img, T, q, I))

    if not courbes:
        log_fn(f"⚠️ Aucune courbe trouvée dans : {dossier_renommes}")
        return

    courbes.sort(key=lambda x: x[0], reverse=reverse)

    fig, ax = plt.subplots(figsize=(10, 6))

    k = 0
    for idx in range(0, len(courbes), pas):
        img, T, q, I = courbes[idx]

        if y_log:
            mask_pos = I > 0
            q_plot = q[mask_pos]
            I_plot = I[mask_pos]
            if len(q_plot) == 0:
                continue
            I_shift = I_plot * (10 ** (k * decal_log10))
        else:
            q_plot = q
            I_plot = I
            I_shift = I_plot + k * 50

        q_shift = q_plot + k * decal_x
        line, = ax.plot(q_shift, I_shift, linewidth=lw, alpha=alpha)

        j = max(0, int(0.95 * (len(q_shift) - 1)))
        ax.text(
            q_shift[j] + label_dx,
            I_shift[j] * (label_y_mult if y_log else 1.0),
            f"{T:.0f}°C",
            fontsize=7,
            va="center",
            color=line.get_color(),
        )

        k += 1

    ax.set_xlabel(r"q ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel("I(q)")
    ax.set_title(titre)
    if y_log:
        ax.set_yscale("log")
    ax.margins(x=0.02)

    fig.savefig(chemin_png, dpi=dpi_save, bbox_inches="tight")
    log_fn(f"✅ Figure enregistrée : {chemin_png}")

    plt.close(fig)
    plt.close("all")


# =================================================
# 6) Waterfall à partir d'une liste de fichiers
# =================================================
def waterfall_plot_from_files(
    file_paths,
    titre,
    chemin_png,
    pas=10,
    y_log=True,
    decal_x=0.01,
    decal_log10=0.18,
    lw=1.2,
    alpha=0.95,
    dpi_save=200,
    label_dx=0.035,
    label_y_mult=1.09,
    reverse=False,
    log_fn=lambda s: None,
):
    if not file_paths:
        log_fn("⚠️ Aucun fichier sélectionné.")
        return

    courbes = []
    for path in file_paths:
        f = os.path.basename(path)

        T = extraire_temperature_depuis_nom(f)
        img = extraire_image_depuis_nom(f)

        if T is None:
            log_fn(f"⚠️ Température introuvable dans le nom : {f}")
            continue
        if img is None:
            img = 0

        try:
            q, I = lire_q_I_depuis_dat(path)
        except Exception:
            log_fn(f"⚠️ Lecture impossible : {f}")
            continue

        if len(q) == 0:
            continue

        courbes.append((img, T, q, I))

    if not courbes:
        log_fn("⚠️ Aucune courbe valide à tracer.")
        return

    courbes.sort(key=lambda x: x[0], reverse=reverse)

    fig, ax = plt.subplots(figsize=(10, 6))

    k = 0
    for idx in range(0, len(courbes), pas):
        img, T, q, I = courbes[idx]

        if y_log:
            mask_pos = I > 0
            q_plot = q[mask_pos]
            I_plot = I[mask_pos]
            if len(q_plot) == 0:
                continue
            I_shift = I_plot * (10 ** (k * decal_log10))
        else:
            q_plot = q
            I_plot = I
            I_shift = I_plot + k * 50

        q_shift = q_plot + k * decal_x
        line, = ax.plot(q_shift, I_shift, linewidth=lw, alpha=alpha)

        j = max(0, int(0.95 * (len(q_shift) - 1)))
        ax.text(
            q_shift[j] + label_dx,
            I_shift[j] * (label_y_mult if y_log else 1.0),
            f"{T:.0f}°C",
            fontsize=7,
            va="center",
            color=line.get_color(),
        )

        k += 1

    ax.set_xlabel(r"q ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel("I(q)")
    ax.set_title(titre)
    if y_log:
        ax.set_yscale("log")
    ax.margins(x=0.02)

    fig.savefig(chemin_png, dpi=dpi_save, bbox_inches="tight")
    log_fn(f"✅ Figure enregistrée : {chemin_png}")

    plt.close(fig)
    plt.close("all")


# =================================================
# 7) Extraction 2 colonnes + renommage
# =================================================
_pattern_scan = re.compile(r"\{(\d+),(\d+)\}")


def extraire_2colonnes(in_path, out_path, log_fn):
    start = trouver_debut_donnees(in_path)
    if start is None:
        log_fn(f"⚠️ Données non trouvées : {os.path.basename(in_path)}")
        return False

    df = pd.read_csv(
        in_path,
        skiprows=start,
        sep=r"\s+",
        engine="python",
        header=None,
        usecols=[0, 1],
        on_bad_lines="skip",
    )
    df.columns = ["q", "I"]
    df.to_csv(out_path, sep="\t", index=False, header=False, na_rep="nan")
    return True


def extraire_et_renommer(
    dossier_exp,
    sample,
    kind,
    T_of_img,
    ignore_non_sample=True,
    log_fn=print,
    type_fichiers="standards",
):
    def est_waxs(fname: str) -> bool:
        return ("WAXS{" in fname) or ("_WAXS{" in fname)

    def est_sub(fname: str) -> bool:
        return fname.startswith("Sub_")

    suffix_mode = {
        "standards": "std",
        "sub": "sub",
        "tous": "all",
    }.get(type_fichiers, "std")

    dest_extract = os.path.join(dossier_exp, f"{sample}_{kind}_2colonnes_{suffix_mode}")
    dest_rename = os.path.join(dest_extract, "fichiers_renommes")
    os.makedirs(dest_extract, exist_ok=True)
    os.makedirs(dest_rename, exist_ok=True)

    sources = []
    for f in sorted(os.listdir(dossier_exp)):
        if not f.lower().endswith(".dat"):
            continue

        if kind == "WAXS" and not est_waxs(f):
            continue
        if kind == "SAXS" and est_waxs(f):
            continue

        # filtre standards / sub / tous
        if type_fichiers == "standards" and est_sub(f):
            continue
        elif type_fichiers == "sub" and not est_sub(f):
            continue

        # filtre échantillon
        if ignore_non_sample:
            if est_sub(f):
                if sample.lower() not in f.lower():
                    continue
            else:
                if not f.startswith(sample):
                    continue

        sources.append(f)

    if not sources:
        log_fn(f"⚠️ Aucun fichier {kind} dans {dossier_exp} pour le mode '{type_fichiers}'")
        return None

    log_fn(f"[{sample}][{kind}] à extraire : {len(sources)}")
    for f in sources:
        extraire_2colonnes(
            os.path.join(dossier_exp, f),
            os.path.join(dest_extract, f),
            log_fn,
        )
    log_fn(f"✅ Extraction {kind} -> {dest_extract}")

    extracted_files = sorted(
        [x for x in os.listdir(dest_extract) if x.lower().endswith(".dat")]
    )
    log_fn(f"[{sample}][{kind}] à renommer : {len(extracted_files)}")

    n_ok = 0
    for f in extracted_files:
        m = _pattern_scan.search(f)
        if not m:
            continue

        img = int(m.group(1))
        T = T_of_img(img)

        src = os.path.join(dest_extract, f)
        out = os.path.join(dest_rename, f"{sample}_{kind}_{img:05d}_{T:.2f}.dat")

        k = 1
        while os.path.exists(out):
            out = os.path.join(
                dest_rename, f"{sample}_{kind}_{img:05d}_{T:.2f}_{k}.dat"
            )
            k += 1

        shutil.copy2(src, out)
        n_ok += 1

    log_fn(f"✅ Renommage {kind} -> {dest_rename} (ok={n_ok})")
    return dest_rename


# =================================================
# 8) Découpage en phases
# =================================================
def _lire_img_T_depuis_nom(nom_fichier: str):
    m = re.search(r"_(\d{5})_([-\d]+(?:\.\d+)?)", nom_fichier)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def detecter_phases_depuis_T(dir_renommes, eps_plateau=0.15, smooth=5, min_len=10):
    files = [f for f in os.listdir(dir_renommes) if f.lower().endswith(".dat")]
    pts = []
    for f in files:
        img, T = _lire_img_T_depuis_nom(f)
        if img is None:
            continue
        pts.append((img, T))
    if len(pts) < 3:
        return []

    pts.sort(key=lambda x: x[0])
    imgs = np.array([p[0] for p in pts], dtype=int)
    Ts = np.array([p[1] for p in pts], dtype=float)

    dT = np.diff(Ts)

    if smooth and smooth > 1 and len(dT) >= smooth:
        kernel = np.ones(smooth) / smooth
        dT_s = np.convolve(dT, kernel, mode="same")
    else:
        dT_s = dT

    labels = []
    for v in dT_s:
        if abs(v) <= eps_plateau:
            labels.append("plateau")
        elif v > 0:
            labels.append("heat")
        else:
            labels.append("cool")

    segs = []
    cur = labels[0]
    seg_start = imgs[0]
    for k in range(1, len(imgs) - 1):
        lab = labels[k]
        if lab != cur:
            segs.append([seg_start, imgs[k], cur])
            seg_start = imgs[k]
            cur = lab
    segs.append([seg_start, imgs[-1], cur])

    def seg_len(s):
        return int(s[1] - s[0] + 1)

    merged = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue
        if seg_len(s) < min_len:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    merged2 = []
    for s in merged:
        if not merged2:
            merged2.append(s)
        else:
            if merged2[-1][2] == s[2]:
                merged2[-1][1] = s[1]
            else:
                merged2.append(s)

    return [(int(a), int(b), c) for a, b, c in merged2]


def ranger_par_phases(
    dir_renommes,
    move_files=False,
    eps_plateau=0.15,
    smooth=5,
    min_len=10,
    log_fn=print,
):
    segs = detecter_phases_depuis_T(
        dir_renommes, eps_plateau=eps_plateau, smooth=smooth, min_len=min_len
    )
    if not segs:
        log_fn("⚠️ Détection des phases impossible.")
        return []

    heat_n = 0
    cool_n = 0
    named = []
    for a, b, lab in segs:
        if lab == "heat":
            heat_n += 1
            named.append((a, b, f"heat{heat_n}"))
        elif lab == "cool":
            cool_n += 1
            named.append((a, b, f"cool{cool_n}"))
        else:
            named.append((a, b, "plateau"))

    for _, _, name in named:
        os.makedirs(os.path.join(dir_renommes, name), exist_ok=True)

    files = [f for f in os.listdir(dir_renommes) if f.lower().endswith(".dat")]
    for f in files:
        img, _T = _lire_img_T_depuis_nom(f)
        if img is None:
            continue

        dest_name = None
        for a, b, name in named:
            if a <= img <= b:
                dest_name = name
                break
        if dest_name is None:
            dest_name = "autres"

        dest_dir = os.path.join(dir_renommes, dest_name)
        os.makedirs(dest_dir, exist_ok=True)

        src = os.path.join(dir_renommes, f)
        dst = os.path.join(dest_dir, f)

        if move_files:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    log_fn("✅ Rangement par phases :")
    for a, b, name in named:
        log_fn(f"  - {name}: images {a} -> {b}")
    return [name for _, _, name in named]


# =================================================
# 9) Waterfall par phase + copie dans _PLOTS
# =================================================
def tracer_waterfall_par_phase(
    dir_renommes,
    sample,
    kind,
    pas=10,
    y_log=True,
    decal_x=0.01,
    decal_log10=0.18,
    dpi_save=200,
    plots_base_dir=None,
    log_fn=print,
):
    sous = [d for d in os.listdir(dir_renommes) if os.path.isdir(os.path.join(dir_renommes, d))]
    sous = [d for d in sous if d.lower().startswith(("heat", "cool", "plateau"))]
    sous.sort()

    for phase in sous:
        path_phase = os.path.join(dir_renommes, phase)
        fichiers = [f for f in os.listdir(path_phase) if f.lower().endswith(".dat")]
        if len(fichiers) < 5:
            continue

        png = os.path.join(path_phase, f"{sample}_{kind}_{phase}.png")
        log_fn(f"📈 Waterfall {kind} - {phase} ({len(fichiers)} fichiers) -> {png}")

        reverse = phase.lower().startswith("cool")

        waterfall_plot_like_user_script(
            path_phase,
            titre=f"{sample} - {kind} - {phase}",
            chemin_png=png,
            pas=pas,
            y_log=y_log,
            decal_x=decal_x,
            decal_log10=decal_log10,
            dpi_save=dpi_save,
            reverse=reverse,
            log_fn=log_fn,
        )

        if plots_base_dir:
            copy_png_to_base(png, plots_base_dir, log_fn=log_fn)


# =================================================
# 10) Utilitaires dossiers
# =================================================
def est_dossier_experience(name: str) -> bool:
    low = name.lower()
    return low.endswith("_saxs_waxs") or low.endswith("_saxs-waxs")


def sample_depuis_nom_dossier(dossier_name: str) -> str:
    m = re.split(r"_saxs[-_]?waxs$", dossier_name, flags=re.IGNORECASE)
    return m[0] if m else dossier_name


def trouver_txt_linkam(dossier_exp: str, sample: str):
    cand = os.path.join(dossier_exp, f"{sample}.txt")
    if os.path.isfile(cand):
        return cand

    txts = [f for f in os.listdir(dossier_exp) if f.lower().endswith(".txt")]
    if not txts:
        return None

    for f in txts:
        if sample.lower() in f.lower():
            return os.path.join(dossier_exp, f)

    for f in txts:
        low = f.lower()
        if ("linkam" in low) or ("temp" in low) or ("temperature" in low):
            return os.path.join(dossier_exp, f)

    return os.path.join(dossier_exp, sorted(txts)[0])


# =================================================
# 11) Traitement dossier expérience
# =================================================
def traiter_un_dossier_experience(
    dossier_exp,
    pas=10,
    y_log=True,
    decal_x=0.01,
    decal_log10=0.18,
    dpi_save=200,
    ignore_non_sample=True,
    methode_T="interp",
    step_per_image=5,
    split_phases=True,
    move_split=False,
    eps_plateau=0.15,
    type_fichiers="standards",
):
    def log(msg):
        print(msg)

    dname = os.path.basename(dossier_exp)
    sample = sample_depuis_nom_dossier(dname)

    plots_base = ensure_dir(os.path.join(dossier_exp, "_PLOTS"))

    log("\n" + "=" * 70)
    log(f"Traitement : {dname}")
    log(f"Sample     : {sample}")
    log(f"Type fichiers : {type_fichiers}")

    path_txt = trouver_txt_linkam(dossier_exp, sample)
    if not path_txt:
        log("⚠️ Pas de .txt Linkam -> stop")
        return
    log(f"Temp .txt  : {os.path.basename(path_txt)}")

    X, Y = lire_xy_linkam(path_txt)
    log(f"Linkam points: {len(X)}")
    if len(X) < 3:
        log("⚠️ X,Y insuffisants -> stop")
        return

    dX = float(np.median(np.diff(X)))
    dt_image_s = step_per_image * dX
    t0_s = float(X[0])

    log(
        f"DEBUG: dX={dX:.6f}s  step_per_image={step_per_image:.6f}  "
        f"dt_image_s={dt_image_s:.6f}s  t0_s={t0_s:.6f}s"
    )

    T_of_img = construire_T_of_img(
        X, Y, dt_image_s=dt_image_s, t0_s=t0_s, methode=methode_T
    )

    dossier_saxs = extraire_et_renommer(
        dossier_exp,
        sample,
        "SAXS",
        T_of_img,
        ignore_non_sample=ignore_non_sample,
        log_fn=log,
        type_fichiers=type_fichiers,
    )
    dossier_waxs = extraire_et_renommer(
        dossier_exp,
        sample,
        "WAXS",
        T_of_img,
        ignore_non_sample=ignore_non_sample,
        log_fn=log,
        type_fichiers=type_fichiers,
    )

    log("📈 Waterfall global ...")
    if dossier_saxs:
        png = os.path.join(dossier_exp, f"{sample}_SAXS_{type_fichiers}.png")
        waterfall_plot_like_user_script(
            dossier_saxs,
            f"{sample} - SAXS",
            png,
            pas=pas,
            y_log=y_log,
            decal_x=decal_x,
            decal_log10=decal_log10,
            dpi_save=dpi_save,
            reverse=False,
            log_fn=log,
        )
        copy_png_to_base(png, plots_base, log_fn=log)

    if dossier_waxs:
        png = os.path.join(dossier_exp, f"{sample}_WAXS_{type_fichiers}.png")
        waterfall_plot_like_user_script(
            dossier_waxs,
            f"{sample} - WAXS",
            png,
            pas=pas,
            y_log=y_log,
            decal_x=decal_x,
            decal_log10=decal_log10,
            dpi_save=dpi_save,
            reverse=False,
            log_fn=log,
        )
        copy_png_to_base(png, plots_base, log_fn=log)

    if split_phases:
        if dossier_saxs:
            log("📁 Découpage en phases SAXS ...")
            ranger_par_phases(
                dossier_saxs,
                move_files=move_split,
                eps_plateau=eps_plateau,
                smooth=5,
                min_len=10,
                log_fn=log,
            )
            log("📈 Waterfall SAXS par phase ...")
            tracer_waterfall_par_phase(
                dossier_saxs,
                sample,
                "SAXS",
                pas=pas,
                y_log=y_log,
                decal_x=decal_x,
                decal_log10=decal_log10,
                dpi_save=dpi_save,
                plots_base_dir=plots_base,
                log_fn=log,
            )

        if dossier_waxs:
            log("📁 Découpage en phases WAXS ...")
            ranger_par_phases(
                dossier_waxs,
                move_files=move_split,
                eps_plateau=eps_plateau,
                smooth=5,
                min_len=10,
                log_fn=log,
            )
            log("📈 Waterfall WAXS par phase ...")
            tracer_waterfall_par_phase(
                dossier_waxs,
                sample,
                "WAXS",
                pas=pas,
                y_log=y_log,
                decal_x=decal_x,
                decal_log10=decal_log10,
                dpi_save=dpi_save,
                plots_base_dir=plots_base,
                log_fn=log,
            )

    log("✅ Terminé.")


# =================================================
# 12) Auto : racine ou dossier exp
# =================================================
def traiter_chemin_auto(chemin, **kwargs):
    if not os.path.isdir(chemin):
        raise ValueError(f"Dossier introuvable: {chemin}")

    if est_dossier_experience(os.path.basename(chemin)):
        traiter_un_dossier_experience(chemin, **kwargs)
        return

    dossiers = sorted(
        [d for d in os.listdir(chemin) if os.path.isdir(os.path.join(chemin, d))]
    )
    dossiers_exp = [d for d in dossiers if est_dossier_experience(d)]
    print(f"=== Dossiers expérience trouvés : {len(dossiers_exp)} ===")

    for d in dossiers_exp:
        traiter_un_dossier_experience(os.path.join(chemin, d), **kwargs)


# =================================================
# INTERFACE TKINTER
# =================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface SWING SAXS/WAXS + Fichier Température")
        self.geometry("1280x880")
        self.minsize(1080, 720)

        self.q = queue.Queue()
        self.after(100, self._poll_queue)

        self.selected_files = []

        self._set_style()
        self._build_ui()

    def _set_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#f5f7fb"
        card = "#ffffff"
        text = "#1f2937"
        muted = "#6b7280"

        self.configure(bg=bg)

        style.configure("TFrame", background=bg)
        style.configure("Card.TLabelframe", background=card, borderwidth=1, relief="solid")
        style.configure(
            "Card.TLabelframe.Label",
            background=card,
            foreground=text,
            font=("Segoe UI", 11, "bold"),
        )

        style.configure("TLabel", background=bg, foreground=text, font=("Segoe UI", 10))
        style.configure("Info.TLabel", background=bg, foreground="#1f6feb", font=("Segoe UI", 10, "bold"))
        style.configure("Small.TLabel", background=bg, foreground=muted, font=("Segoe UI", 9))

        style.configure("TEntry", padding=6, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=bg, font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("TRadiobutton", background=bg, font=("Segoe UI", 10))
        style.configure("TProgressbar", thickness=8)

    def _add_label_with_info(self, parent, text, info, row, column, pad):
        container = ttk.Frame(parent)
        container.grid(row=row, column=column, sticky="w", **pad)

        ttk.Label(container, text=text).pack(side="left")
        info_lbl = ttk.Label(container, text=" ⓘ", style="Info.TLabel", cursor="hand2")
        info_lbl.pack(side="left")
        ToolTip(info_lbl, info)

    def _update_mode(self):
        mode = self.var_mode.get()

        if mode == "manual":
            self.pipeline_params_frame.pack_forget()
            self.var_pas.set("1")
        else:
            if not self.pipeline_params_frame.winfo_manager():
                self.pipeline_params_frame.pack(fill="x", pady=(0, 10), before=self.type_fichiers_frame)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=14, pady=14)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            header,
            text="Interface SWING SAXS/WAXS + Fichier Température",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        mode_card = ttk.LabelFrame(root, text="Mode", style="Card.TLabelframe")
        mode_card.pack(fill="x", pady=(0, 10))

        mode_inner = ttk.Frame(mode_card)
        mode_inner.pack(fill="x", padx=10, pady=10)

        self.var_mode = tk.StringVar(value="pipeline")
        ttk.Radiobutton(
            mode_inner,
            text="Dossier complet",
            variable=self.var_mode,
            value="pipeline",
            command=self._update_mode
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            mode_inner,
            text="Tracer des fichiers sélectionnés",
            variable=self.var_mode,
            value="manual",
            command=self._update_mode
        ).pack(side="left", padx=8)

        path_card = ttk.LabelFrame(root, text="Dossier / fichiers", style="Card.TLabelframe")
        path_card.pack(fill="x", pady=(0, 10))

        path_inner = ttk.Frame(path_card)
        path_inner.pack(fill="x", padx=10, pady=10)

        self.var_chemin = tk.StringVar(value="")
        ttk.Label(path_inner, text="Dossier racine ou dossier expérience :").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(path_inner, textvariable=self.var_chemin, width=90).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(path_inner, text="Parcourir dossier...", command=self.browse_folder, style="Secondary.TButton").grid(row=0, column=2, **pad)

        ttk.Label(path_inner, text="Fichiers .dat sélectionnés :").grid(row=1, column=0, sticky="w", **pad)
        self.var_files = tk.StringVar(value="Aucun fichier sélectionné")
        ttk.Label(path_inner, textvariable=self.var_files, style="Small.TLabel").grid(row=1, column=1, sticky="w", **pad)
        ttk.Button(path_inner, text="Choisir fichiers .dat...", command=self.browse_files, style="Secondary.TButton").grid(row=1, column=2, **pad)

        path_inner.columnconfigure(1, weight=1)

        self.var_temps_image = tk.StringVar(value="10")
        self.var_pas_linkam = tk.StringVar(value="2")
        self.var_eps = tk.StringVar(value="0.10")
        self.var_type_fichiers = tk.StringVar(value="standards")

        self.var_pas = tk.StringVar(value="10")
        self.var_decalx = tk.StringVar(value="0.01")
        self.var_decallog = tk.StringVar(value="0.18")

        self.var_ylog = tk.BooleanVar(value=True)
        self.var_move = tk.BooleanVar(value=False)
        self.var_split = tk.BooleanVar(value=True)
        self.var_reverse = tk.BooleanVar(value=False)
        self.var_titre = tk.StringVar(value="Graphe manuel")

        params_card = ttk.LabelFrame(root, text="Paramètres", style="Card.TLabelframe")
        params_card.pack(fill="x", pady=(0, 10))

        params = ttk.Frame(params_card)
        params.pack(fill="x", padx=10, pady=10)

        self.analyse_card = ttk.LabelFrame(params, text="Analyse fichier", style="Card.TLabelframe")
        self.analyse_card.pack(fill="x", pady=(0, 10))

        analyse_inner = ttk.Frame(self.analyse_card)
        analyse_inner.pack(fill="x", padx=10, pady=10)

        self.pipeline_params_frame = ttk.Frame(analyse_inner)
        self.pipeline_params_frame.pack(fill="x", pady=(0, 10))

        analyse_fields = [
            (
                "Temps par image (s)",
                self.var_temps_image,
                "Temps correspondant à une image SAXS/WAXS en secondes.\nExemple : 10"
            ),
            (
                "Pas de temps Linkam (s)",
                self.var_pas_linkam,
                "Pas de temps théorique entre deux points du fichier Linkam.\nExemple : 2"
            ),
            (
                "Epsilon palier",
                self.var_eps,
                "Seuil de variation de température pour détecter un palier."
            ),
        ]

        for i, (label, var, info) in enumerate(analyse_fields):
            col_base = i * 2
            self._add_label_with_info(self.pipeline_params_frame, f"{label} :", info, 0, col_base, pad)
            ttk.Entry(self.pipeline_params_frame, textvariable=var, width=14).grid(row=0, column=col_base + 1, sticky="w", **pad)

        self.type_fichiers_frame = ttk.Frame(analyse_inner)
        self.type_fichiers_frame.pack(fill="x")

        ttk.Label(self.type_fichiers_frame, text="Type de fichiers :").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        type_frame = ttk.Frame(self.type_fichiers_frame)
        type_frame.grid(row=0, column=1, columnspan=5, padx=10, pady=6, sticky="w")

        ttk.Radiobutton(
            type_frame,
            text="Fichiers standards",
            variable=self.var_type_fichiers,
            value="standards"
        ).pack(side="left", padx=6)

        ttk.Radiobutton(
            type_frame,
            text="Fichiers Sub",
            variable=self.var_type_fichiers,
            value="sub"
        ).pack(side="left", padx=6)

        ttk.Radiobutton(
            type_frame,
            text="Tous",
            variable=self.var_type_fichiers,
            value="tous"
        ).pack(side="left", padx=6)

        self.graph_card = ttk.LabelFrame(params, text="Graphique", style="Card.TLabelframe")
        self.graph_card.pack(fill="x", pady=(0, 10))

        graph_inner = ttk.Frame(self.graph_card)
        graph_inner.pack(fill="x", padx=10, pady=10)

        graph_fields = [
            (
                "Pas d'affichage",
                self.var_pas,
                "Trace une courbe tous les N fichiers.\nExemple : 10 = une courbe affichée tous les 10 fichiers."
            ),
            (
                "Décalage X",
                self.var_decalx,
                "Décalage horizontal entre les courbes."
            ),
            (
                "Décalage log10 Y",
                self.var_decallog,
                "Décalage vertical logarithmique entre les courbes.\nPlus grand = courbes plus espacées."
            ),
        ]

        for i, (label, var, info) in enumerate(graph_fields):
            col_base = i * 2
            self._add_label_with_info(graph_inner, f"{label} :", info, 0, col_base, pad)
            ttk.Entry(graph_inner, textvariable=var, width=14).grid(row=0, column=col_base + 1, sticky="w", **pad)

        manual_frame = ttk.Frame(graph_inner)
        manual_frame.grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(8, 0))
        ttk.Label(manual_frame, text="Titre manuel :").pack(side="left", padx=(0, 8))
        ttk.Entry(manual_frame, textvariable=self.var_titre, width=28).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(manual_frame, text="Ordre inverse (manuel)", variable=self.var_reverse).pack(side="left")

        opts = ttk.Frame(params)
        opts.pack(fill="x", padx=6, pady=(4, 0))

        cb1 = ttk.Checkbutton(opts, text="Axe Y en log", variable=self.var_ylog)
        cb1.pack(side="left", padx=8)
        info1 = ttk.Label(opts, text=" ⓘ", style="Info.TLabel", cursor="hand2")
        info1.pack(side="left")
        ToolTip(info1, "Active une échelle logarithmique sur l’axe Y.\nRecommandé pour les courbes SAXS/WAXS.")

        cb2 = ttk.Checkbutton(opts, text="Découper en phases", variable=self.var_split)
        cb2.pack(side="left", padx=8)
        info2 = ttk.Label(opts, text=" ⓘ", style="Info.TLabel", cursor="hand2")
        info2.pack(side="left")
        ToolTip(info2, "Sépare automatiquement les fichiers en phases : chauffe, refroidissement et plateau.")

        cb3 = ttk.Checkbutton(opts, text="Déplacer au lieu de copier", variable=self.var_move)
        cb3.pack(side="left", padx=8)
        info3 = ttk.Label(opts, text=" ⓘ", style="Info.TLabel", cursor="hand2")
        info3.pack(side="left")
        ToolTip(info3, "Si activé : les fichiers sont déplacés dans les sous-dossiers.\nSinon : ils sont copiés et restent aussi dans le dossier principal.")

        action_bar = ttk.Frame(root)
        action_bar.pack(fill="x", pady=(0, 10))

        self.btn_run = ttk.Button(action_bar, text="Lancer", command=self.run_clicked, style="Primary.TButton")
        self.btn_run.pack(side="left", padx=(0, 8))

        ttk.Button(action_bar, text="Fermer", command=self.destroy, style="Secondary.TButton").pack(side="left")

        self.progress = ttk.Progressbar(action_bar, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))

        log_card = ttk.LabelFrame(root, text="Journal d'exécution", style="Card.TLabelframe")
        log_card.pack(fill="both", expand=True)

        log_inner = ttk.Frame(log_card)
        log_inner.pack(fill="both", expand=True, padx=10, pady=10)

        self.txt = tk.Text(
            log_inner,
            height=24,
            wrap="word",
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.txt.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_inner, orient="vertical", command=self.txt.yview)
        scroll.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=scroll.set)

        self._update_mode()

    def browse_folder(self):
        d = filedialog.askdirectory(title="Choisir un dossier")
        if d:
            self.var_chemin.set(d)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Choisir des fichiers .dat",
            filetypes=[("Fichiers DAT", "*.dat"), ("Tous les fichiers", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.var_files.set(f"{len(self.selected_files)} fichier(s) sélectionné(s)")
        else:
            self.selected_files = []
            self.var_files.set("Aucun fichier sélectionné")

    def log(self, msg: str):
        self.q.put(("log", msg))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.txt.insert("end", payload + "\n")
                    self.txt.see("end")
                elif kind == "done":
                    self.progress.stop()
                    self.btn_run.config(state="normal")
                    messagebox.showinfo("Terminé", payload)
                elif kind == "error":
                    self.progress.stop()
                    self.btn_run.config(state="normal")
                    messagebox.showerror("Erreur", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _read_params(self):
        mode = self.var_mode.get()

        if mode == "pipeline":
            temps_image = float(self.var_temps_image.get())
            pas_linkam = float(self.var_pas_linkam.get())

            if pas_linkam == 0:
                raise ValueError("Le pas de temps Linkam ne peut pas être égal à 0.")

            step_per_image = temps_image / pas_linkam
        else:
            temps_image = None
            pas_linkam = None
            step_per_image = None

        params = {
            "mode": mode,
            "temps_image": temps_image,
            "pas_linkam": pas_linkam,
            "step_per_image": step_per_image,
            "pas": int(self.var_pas.get()),
            "decal_x": float(self.var_decalx.get()),
            "decal_log10": float(self.var_decallog.get()),
            "dpi_save": 200,
            "eps_plateau": float(self.var_eps.get()),
            "type_fichiers": self.var_type_fichiers.get(),
            "y_log": bool(self.var_ylog.get()),
            "split_phases": bool(self.var_split.get()),
            "move_split": bool(self.var_move.get()),
            "reverse": bool(self.var_reverse.get()),
            "titre": self.var_titre.get().strip() or "graphe manuel",
        }

        if mode == "pipeline":
            chemin = self.var_chemin.get().strip()
            if not chemin or not os.path.isdir(chemin):
                raise ValueError("Choisir un dossier valide.")
            params["chemin"] = chemin
        else:
            if not self.selected_files:
                raise ValueError("Choisir des fichiers .dat à tracer.")
            params["files"] = self.selected_files

        return params

    def run_clicked(self):
        self.btn_run.config(state="disabled")
        self.progress.start(10)
        self.txt.delete("1.0", "end")

        def worker():
            try:
                p = self._read_params()

                self.log(f"MODE = {p['mode']}")
                self.log(f"Type de fichiers = {p['type_fichiers']}")
                self.log(f"Pas d'affichage = {p['pas']}")
                self.log(f"Décalage X = {p['decal_x']}")
                self.log(f"Décalage log10 Y = {p['decal_log10']}")
                self.log("")

                if p["mode"] == "pipeline":
                    self.log(f"CHEMIN = {p['chemin']}")
                    self.log(f"Temps par image = {p['temps_image']} s")
                    self.log(f"Pas de temps Linkam = {p['pas_linkam']} s")
                    self.log(f"step_per_image = {p['step_per_image']:.4f}")
                    self.log(f"Epsilon palier = {p['eps_plateau']}")
                    self.log("")

                    _print_backup = print

                    def myprint(*args, **kwargs):
                        self.log(" ".join(str(a) for a in args))

                    globals()["print"] = myprint
                    try:
                        traiter_chemin_auto(
                            p["chemin"],
                            pas=p["pas"],
                            y_log=p["y_log"],
                            decal_x=p["decal_x"],
                            decal_log10=p["decal_log10"],
                            dpi_save=p["dpi_save"],
                            ignore_non_sample=True,
                            methode_T="interp",
                            step_per_image=p["step_per_image"],
                            split_phases=p["split_phases"],
                            move_split=p["move_split"],
                            eps_plateau=p["eps_plateau"],
                            type_fichiers=p["type_fichiers"],
                        )
                    finally:
                        globals()["print"] = _print_backup

                    self.q.put(("done", "Pipeline terminé.\nFigures enregistrées dans les dossiers expérience."))

                else:
                    self.log(f"FICHIERS sélectionnés = {len(p['files'])}")
                    out_dir = os.path.dirname(p["files"][0])
                    png = os.path.join(out_dir, f"{p['titre'].replace(' ', '_')}.png")

                    waterfall_plot_from_files(
                        p["files"],
                        titre=p["titre"],
                        chemin_png=png,
                        pas=p["pas"],
                        y_log=p["y_log"],
                        decal_x=p["decal_x"],
                        decal_log10=p["decal_log10"],
                        dpi_save=p["dpi_save"],
                        reverse=p["reverse"],
                        log_fn=self.log,
                    )

                    self.q.put(("done", f"Waterfall manuel terminé.\nFigure enregistrée :\n{png}"))

            except Exception as e:
                self.q.put(("error", str(e)))
                self.log(str(e))

        threading.Thread(target=worker, daemon=True).start()


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()