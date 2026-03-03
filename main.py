"""
Point d'entrée principal du projet EEG.
Lance un menu interactif pour choisir le mode d'exécution.
"""

import sys
import os


def print_header():
    print()
    print("╔══════════════════════════════════════╗")
    print("║       Projet EEG — OpenBCI Cyton     ║")
    print("╚══════════════════════════════════════╝")
    print()


def ask(prompt: str, default: str = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {prompt}{suffix} : ").strip()
    return value if value else default


def menu_offline():
    print()
    print("── Analyse offline (PhysioNet) ──────────")
    subject = ask("Sujet", "S001")
    print()
    print("  Runs disponibles :")
    print("    R01 — baseline yeux ouverts")
    print("    R02 — baseline yeux fermés")
    print("    R04 — imagerie motrice")
    print()

    data_dir = "data"
    path_eo    = os.path.join(data_dir, subject, f"{subject}R01.edf")
    path_ec    = os.path.join(data_dir, subject, f"{subject}R02.edf")
    path_motor = os.path.join(data_dir, subject, f"{subject}R04.edf")

    missing = [p for p in (path_eo, path_ec, path_motor) if not os.path.exists(p)]
    if missing:
        print(f"  [ERREUR] Fichiers manquants :")
        for p in missing:
            print(f"    {p}")
        print()
        dl = ask("Télécharger maintenant ? (o/n)", "o")
        if dl.lower() == "o":
            import download_data
            download_data.download_subject(int(subject[1:]), download_data.RUNS)
        else:
            print("  Abandon.")
            return

    print(f"\n  Chargement de {subject}…")
    import eeg_analysis
    import matplotlib.pyplot as plt

    raw_eo    = eeg_analysis.load_raw(path_eo)
    raw_ec    = eeg_analysis.load_raw(path_ec)
    raw_motor = eeg_analysis.load_raw(path_motor)

    print(f"  {len(raw_eo.ch_names)} canaux, {raw_eo.info['sfreq']} Hz")

    OCC_CH    = ["O1", "OZ", "O2"]
    MOTOR_CH  = ["C3", "CZ", "C4"]
    DISPLAY_CH = ["FP1", "C3", "O1", "C4", "O2"]

    print("\n  Filtrage…")
    raw_eo_f    = eeg_analysis.apply_filters(raw_eo)
    raw_ec_f    = eeg_analysis.apply_filters(raw_ec)
    raw_motor_f = eeg_analysis.apply_filters(raw_motor)

    eeg_analysis.plot_raw_signal(raw_eo, DISPLAY_CH, duration=10,
                                 title=f"{subject} R01 — Signal brut")
    eeg_analysis.compute_psd(raw_eo_f, DISPLAY_CH,
                             title=f"{subject} R01 — PSD filtrée")
    eeg_analysis.compare_alpha(raw_eo_f, raw_ec_f, occ_channels=OCC_CH)
    eeg_analysis.plot_mu_rhythm(raw_motor_f, channels=MOTOR_CH)

    print("\n  Ferme les fenêtres pour quitter.")
    plt.show()


def menu_realtime():
    print()
    print("── Temps réel (OpenBCI Cyton) ───────────")
    port = ask("Port série", "/dev/cu.usbserial-DM03H2DU")
    ch_input = ask("Canaux à afficher (indices 0-7, vide = tous)", "")
    ch_indices = [int(c) for c in ch_input.split()] if ch_input else None

    import realtime_eeg
    realtime_eeg.run_realtime(port=port, ch_indices=ch_indices)


def menu_download():
    print()
    print("── Téléchargement PhysioNet ─────────────")
    import download_data

    sujets_input = ask("Sujets (ex: 1 2 3)", "1 2 3")
    sujets = [int(s) for s in sujets_input.split()]

    for s in sujets:
        download_data.download_subject(s, download_data.RUNS)

    print("\n  Téléchargement terminé.")


def menu_recording():
    print()
    print("── Analyser un enregistrement ───────────")

    rec_dir = "recordings"
    if not os.path.exists(rec_dir):
        print("  Aucun enregistrement trouvé (dossier recordings/ inexistant).")
        return

    files = sorted([f for f in os.listdir(rec_dir) if f.endswith(".npy")])
    if not files:
        print("  Aucun fichier .npy dans recordings/.")
        return

    print("  Enregistrements disponibles :")
    for i, f in enumerate(files):
        meta_path = os.path.join(rec_dir, f.replace(".npy", ".json"))
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = __import__("json").load(fh)
            print(f"    {i+1}. {f}  ({meta['duration_sec']} s, {meta['sfreq']} Hz)")
        else:
            print(f"    {i+1}. {f}")

    print()
    idx = ask(f"Numéro (1-{len(files)})", "1")
    npy_path = os.path.join(rec_dir, files[int(idx) - 1])

    import eeg_analysis
    import matplotlib.pyplot as plt

    print(f"\n  Chargement de {npy_path}…")
    raw = eeg_analysis.load_recording(npy_path)
    duration = raw.times[-1]
    print(f"  {len(raw.ch_names)} canaux, {raw.info['sfreq']} Hz, {duration:.1f} s")

    print("  Filtrage…")
    raw_filt = eeg_analysis.apply_filters(raw)

    # Signal brut + PSD globale
    eeg_analysis.plot_raw_signal(raw, raw.ch_names, duration=min(10, duration),
                                 title=f"Enregistrement — {os.path.basename(npy_path)}")
    eeg_analysis.compute_psd(raw_filt, raw.ch_names,
                             title=f"PSD complète — {os.path.basename(npy_path)}")

    # Analyse Berger (split yeux fermés / yeux ouverts)
    print()
    berger = ask("Analyse yeux fermés vs yeux ouverts ? (o/n)", "o")
    if berger.lower() == "o":
        split = ask(f"Temps de split en secondes (durée totale: {duration:.0f} s)", str(int(duration // 2)))
        split = float(split)

        sfreq = raw_filt.info["sfreq"]
        split_sample = int(split * sfreq)

        raw_ec = raw_filt.copy().crop(tmin=0,     tmax=split)
        raw_eo = raw_filt.copy().crop(tmin=split, tmax=duration)

        print(f"\n  Yeux fermés : 0 – {split:.0f} s")
        print(f"  Yeux ouverts : {split:.0f} – {duration:.0f} s")
        print("  Canaux disponibles :", raw.ch_names)
        print()

        ch_input = ask("Canaux occipitaux à comparer (ex: CH1 CH2)", " ".join(raw.ch_names[:3]))
        # Nettoie l'input : retire crochets, guillemets, virgules
        import re
        occ_ch = re.findall(r'CH\d+', ch_input.upper())

        eeg_analysis.compare_alpha(raw_eo, raw_ec, occ_channels=occ_ch)

    print("\n  Ferme les fenêtres pour quitter.")
    plt.show()


def main():
    print_header()
    print("  1. Analyse offline   (PhysioNet EDF+)")
    print("  2. Temps réel        (OpenBCI Cyton)")
    print("  3. Télécharger data  (PhysioNet)")
    print("  4. Analyser un enregistrement")
    print("  q. Quitter")
    print()

    choice = ask("Choix", "1")

    if choice == "1":
        menu_offline()
    elif choice == "2":
        menu_realtime()
    elif choice == "3":
        menu_download()
    elif choice == "4":
        menu_recording()
    elif choice in ("q", "Q"):
        sys.exit(0)
    else:
        print("  Choix invalide.")
        sys.exit(1)


if __name__ == "__main__":
    main()
