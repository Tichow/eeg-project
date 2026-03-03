"""
Outil interactif de mapping des électrodes — OpenBCI Ultracortex Mark IV / Cyton.

Vue 3D interactive : les 35 positions sont placées directement sur la sphère
(coordonnées 3D MNE, pas de projection qui déforme).

Contrôles :
  Clic gauche    → sélectionner une position  (glisser pour faire tourner la tête)
  Touches 1-8    → assigner CH{n} à la position sélectionnée
  Suppr/Backspace → effacer l'assignation
  Échap          → désélectionner
  S              → sauvegarder dans channel_map.json
  C              → effacer tout le mapping
  Q              → quitter

Usage :
    python headset_map.py
"""

import json
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (active la projection 3D)
from mpl_toolkits.mplot3d import proj3d
import mne

mne.set_log_level("WARNING")

MAP_FILE = "channel_map.json"

CH_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

MARK_IV_NODES = [
    "Fp1", "Fp2",
    "F7",  "F3",  "Fz",  "F4",  "F8",
    "FC5", "FC1", "FCz", "FC2", "FC6",
    "T7",  "C3",  "Cz",  "C4",  "T8",
    "TP9", "CP5", "CP1", "CPz", "CP2", "CP6", "TP10",
    "P7",  "P3",  "Pz",  "P4",  "P8",
    "PO9", "O1",  "Oz",  "O2",  "PO10",
    "Iz",
]

MARK_IV_DEFAULT = {"Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"}

DEFAULT_MAPPING = {
    "CH1": "Fp1", "CH2": "Fp2",
    "CH3": "C3",  "CH4": "C4",
    "CH5": "P7",  "CH6": "P8",
    "CH7": "O1",  "CH8": "O2",
}

CLICK_THRESH_PX = 20   # distance max en pixels pour valider un clic (vs drag)
PICK_RADIUS_PX  = 25   # distance max en pixels pour sélectionner une électrode


# ---------------------------------------------------------------------------
# Chargement / sauvegarde
# ---------------------------------------------------------------------------

def load_channel_map(path: str = MAP_FILE) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_channel_map(mapping: dict, path: str = MAP_FILE) -> None:
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  [OK] Mapping sauvegardé → {path}")
    for i in range(1, 9):
        ch = f"CH{i}"
        print(f"    {ch} → {mapping.get(ch, '—')}")


# ---------------------------------------------------------------------------
# Outil interactif 3D
# ---------------------------------------------------------------------------

def run_mapping_tool(map_path: str = MAP_FILE) -> None:
    montage    = mne.channels.make_standard_montage("standard_1020")
    pos_3d_all = montage.get_positions()["ch_pos"]

    positions = {}   # name → (x, y, z)
    for name in MARK_IV_NODES:
        if name in pos_3d_all:
            p = pos_3d_all[name]
            positions[name] = (float(p[0]), float(p[1]), float(p[2]))

    if os.path.exists(map_path):
        ch_to_pos = load_channel_map(map_path)
    else:
        ch_to_pos = dict(DEFAULT_MAPPING)

    pos_to_ch = {v: k for k, v in ch_to_pos.items()}
    state     = {"selected": None}

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    fig.canvas.manager.set_window_title("Mapping 3D — OpenBCI Ultracortex Mark IV")

    # Panneau 3D (gauche)
    ax = fig.add_axes([0.0, 0.08, 0.70, 0.88], projection="3d")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-65)

    # Panneau légende (droite)
    ax_leg = fig.add_axes([0.70, 0.08, 0.28, 0.88])
    ax_leg.set_axis_off()

    # ── Demi-sphère (tête) ──────────────────────────────────────────────────
    # Paramétrage cohérent avec ax.plot([px],[pz],[py]) :
    #   axX = r·sin(v)·cos(u) = MNE_x (droite)
    #   axY = r·cos(v)        = MNE_z (haut)
    #   axZ = r·sin(v)·sin(u) = MNE_y (avant)
    head_r = 0.097
    v_max  = 0.70 * np.pi          # inclut TP9/TP10 (MNE_z ≈ −0.046 m)
    u_pts  = np.linspace(0, 2 * np.pi, 48)
    v_pts  = np.linspace(0, v_max,    24)
    U, V   = np.meshgrid(u_pts, v_pts)
    Xs = head_r * np.sin(V) * np.cos(U)
    Ys = head_r * np.cos(V)
    Zs = head_r * np.sin(V) * np.sin(U)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.06, color="#aaaaaa", linewidth=0,
                    rcount=24, ccount=48, zorder=0)

    # Méridiens (8 lignes verticales)
    v_line = np.linspace(0, v_max, 60)
    for u_val in np.linspace(0, 2 * np.pi, 9)[:-1]:
        xl = head_r * np.sin(v_line) * np.cos(u_val)
        yl = head_r * np.cos(v_line)
        zl = head_r * np.sin(v_line) * np.sin(u_val)
        ax.plot(xl, yl, zl, color="#aaaaaa", lw=0.6, alpha=0.6, zorder=1)

    # Parallèles (3 lignes horizontales : sommet, équateur, bord de coupe)
    u_line = np.linspace(0, 2 * np.pi, 120)
    for v_val in [np.pi * 0.25, np.pi * 0.50, v_max]:
        xl  = head_r * np.sin(v_val) * np.cos(u_line)
        yl  = np.full_like(u_line, head_r * np.cos(v_val))
        zl  = head_r * np.sin(v_val) * np.sin(u_line)
        clr = "#888888" if v_val == v_max else "#aaaaaa"
        lw  = 1.0       if v_val == v_max else 0.6
        alp = 0.8       if v_val == v_max else 0.6
        ax.plot(xl, yl, zl, color=clr, lw=lw, alpha=alp, zorder=1)

    # Nez (direction +y MNE = avant = +Z dans ax)
    nose_r = head_r * 1.06
    ax.plot([0], [0], [nose_r], marker=(3, 0, 0), ms=10, color="#555555", zorder=5)

    # Repères d'orientation (convention neurologique : droite = droite)
    ax.text(head_r * 1.18, 0, 0, "G", fontsize=13, color="#3cb44b",
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(-head_r * 1.18, 0, 0, "D", fontsize=13, color="#e6194b",
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(0, 0, nose_r * 1.08, "Nez", fontsize=7, color="#555555",
            ha="center", va="bottom", zorder=6)

    ax.set_title("Vue neurologique (D = droite du sujet) · Glisser pour tourner",
                 fontsize=8, color="#555555", pad=4)

    # ── Points des électrodes ────────────────────────────────────────────────
    dots         = {}   # name → Line3D
    texts_3d     = {}   # name → Text3D
    artist_to_name = {}

    def _color_size(name: str):
        ch     = pos_to_ch.get(name)
        sel    = (name == state["selected"])
        is_def = name in MARK_IV_DEFAULT
        if ch:
            color = CH_COLORS[int(ch[2:]) - 1]
            ms    = 13
        elif sel:
            color = "#ffffff"
            ms    = 14
        elif is_def:
            color = "#555555"
            ms    = 10
        else:
            color = "#cccccc"
            ms    = 7
        ec = "#ffffff" if sel else "white"
        ew = 2.5      if sel else 0.7
        return color, ms, ec, ew

    for name, (px, py, pz) in positions.items():
        color, ms, ec, ew = _color_size(name)
        # MNE coords : x=right, y=anterior, z=superior
        # Dans ax 3D on mappe : X=-x (miroir, convention neurologique), Y=z, Z=y
        line, = ax.plot([-px], [pz], [py], "o", ms=ms, color=color,
                        markeredgecolor=ec, markeredgewidth=ew,
                        picker=False, zorder=3)
        ch    = pos_to_ch.get(name)
        label = f"{name}\n{ch}" if ch else name
        txt   = ax.text(-px, pz + 0.006, py, label,
                        fontsize=5.5, ha="center", color="#222222", zorder=4)
        dots[name]         = line
        texts_3d[name]     = txt
        artist_to_name[id(line)] = name

    # ── Panneau légende droite ───────────────────────────────────────────────
    ch_patches = [
        mpatches.Patch(color=CH_COLORS[i],
                       label=f"CH{i+1}  →  {ch_to_pos.get(f'CH{i+1}', '—')}")
        for i in range(8)
    ]
    node_def = mlines.Line2D([], [], marker="o", color="#555555", ms=8,
                             markeredgecolor="white", linestyle="none",
                             label="Nœud par défaut (8ch)")
    node_oth = mlines.Line2D([], [], marker="o", color="#cccccc", ms=6,
                             markeredgecolor="white", linestyle="none",
                             label="Nœud disponible")
    ax_leg.legend(handles=ch_patches + [node_def, node_oth],
                  loc="upper left", fontsize=8.5, framealpha=0.9,
                  title="Canal → Position", title_fontsize=9)

    # ── Barre de statut ──────────────────────────────────────────────────────
    status_txt = fig.text(
        0.5, 0.01,
        "Cliquer sur une position pour la sélectionner",
        ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.8),
    )

    # ── Refresh ──────────────────────────────────────────────────────────────

    def _refresh():
        for name, line in dots.items():
            color, ms, ec, ew = _color_size(name)
            line.set_color(color)
            line.set_markersize(ms)
            line.set_markeredgecolor(ec)
            line.set_markeredgewidth(ew)
            ch  = pos_to_ch.get(name)
            sel = (name == state["selected"])
            texts_3d[name].set_text(f"{name}\n{ch}" if ch else name)
            texts_3d[name].set_fontweight("bold" if (ch or sel) else "normal")
            texts_3d[name].set_fontsize(7 if (ch or sel) else 5.5)

        for i, patch in enumerate(ch_patches):
            patch.set_label(f"CH{i+1}  →  {ch_to_pos.get(f'CH{i+1}', '—')}")
        ax_leg.legend(handles=ch_patches + [node_def, node_oth],
                      loc="upper left", fontsize=8.5, framealpha=0.9,
                      title="Canal → Position", title_fontsize=9)
        fig.canvas.draw_idle()

    def _assign(pos_name: str, ch_name: Optional[str]):
        if pos_name in pos_to_ch:
            del ch_to_pos[pos_to_ch[pos_name]]
            del pos_to_ch[pos_name]
        if ch_name is not None:
            old_pos = ch_to_pos.get(ch_name)
            if old_pos and old_pos in pos_to_ch:
                del pos_to_ch[old_pos]
            ch_to_pos[ch_name] = pos_name
            pos_to_ch[pos_name] = ch_name

    def _set_status(msg: str):
        status_txt.set_text(msg)
        fig.canvas.draw_idle()

    # ── Détection clic (distingue clic simple vs drag de rotation) ───────────
    press_pos = {"x": None, "y": None}

    def on_press(event):
        press_pos["x"] = event.x
        press_pos["y"] = event.y

    def on_release(event):
        if press_pos["x"] is None or event.inaxes is not ax:
            press_pos["x"] = None
            return
        dx = abs(event.x - press_pos["x"])
        dy = abs(event.y - press_pos["y"])
        press_pos["x"] = None

        # Si drag significatif → rotation, pas une sélection
        if dx > CLICK_THRESH_PX or dy > CLICK_THRESH_PX:
            return

        # Trouver l'électrode la plus proche dans l'espace écran 2D
        proj = ax.get_proj()
        closest, min_d = None, float("inf")
        for name, (px, py, pz) in positions.items():
            # Conversion coords MNE → coords 3D ax (miroir X, même que le plot)
            x2, y2, _ = proj3d.proj_transform(-px, pz, py, proj)
            # En coordonnées écran
            xd, yd = ax.transData.transform((x2, y2))
            d = np.hypot(event.x - xd, event.y - yd)
            if d < min_d:
                min_d, closest = d, name

        if min_d > PICK_RADIUS_PX or closest is None:
            state["selected"] = None
            _set_status("Cliquer sur une position pour la sélectionner")
            _refresh()
            return

        state["selected"] = closest
        ch = pos_to_ch.get(closest, "—")
        _set_status(
            f"[ {closest} ]  actuel : {ch}  —  "
            "Appuyer sur 1-8 pour assigner, Suppr pour effacer, Échap pour annuler"
        )
        _refresh()

    def on_key(event):
        sel = state["selected"]

        if event.key in [str(n) for n in range(1, 9)]:
            if sel is None:
                _set_status("Sélectionner d'abord une position (clic)")
                return
            ch_name = f"CH{event.key}"
            _assign(sel, ch_name)
            state["selected"] = None
            _set_status(f"  {ch_name} → {sel}  ·  Cliquer pour sélectionner une autre position")
            _refresh()

        elif event.key in ("delete", "backspace"):
            if sel is None:
                return
            old_ch = pos_to_ch.get(sel, "—")
            _assign(sel, None)
            state["selected"] = None
            _set_status(f"  {sel} ({old_ch}) effacé  ·  Cliquer pour sélectionner une position")
            _refresh()

        elif event.key == "escape":
            state["selected"] = None
            _set_status("Cliquer sur une position pour la sélectionner")
            _refresh()

        elif event.key == "s":
            save_channel_map(ch_to_pos, map_path)
            _set_status(f"  Sauvegardé dans {map_path}  ({len(ch_to_pos)}/8 canaux assignés)")
            fig.canvas.draw_idle()

        elif event.key == "c":
            ch_to_pos.clear()
            pos_to_ch.clear()
            state["selected"] = None
            _set_status("Mapping effacé — Cliquer pour sélectionner une position")
            _refresh()

        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event",      on_key)

    plt.show()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_mapping_tool()
