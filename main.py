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
    print("    1. R01 — baseline yeux ouverts")
    print("    2. R02 — baseline yeux fermés")
    print("    3. R04 — imagerie motrice")
    print()
    run_choice = ask("Run à visualiser (1/2/3)", "1")

    data_dir = "data"
    run_map = {
        "1": (f"{subject}R01.edf", "R01 — Yeux ouverts"),
        "2": (f"{subject}R02.edf", "R02 — Yeux fermés"),
        "3": (f"{subject}R04.edf", "R04 — Imagerie motrice"),
    }
    if run_choice not in run_map:
        print("  Choix invalide.")
        return

    filename, run_label = run_map[run_choice]
    edf_path = os.path.join(data_dir, subject, filename)

    if not os.path.exists(edf_path):
        print(f"  [ERREUR] Fichier manquant : {edf_path}")
        print()
        dl = ask("Télécharger maintenant ? (o/n)", "o")
        if dl.lower() == "o":
            import download_data
            download_data.download_subject(int(subject[1:]), download_data.RUNS)
        else:
            print("  Abandon.")
            return

    print(f"\n  Chargement de {edf_path}…")
    import eeg_analysis
    raw = eeg_analysis.load_raw(edf_path)
    sfreq = raw.info["sfreq"]
    all_ch = raw.ch_names
    print(f"  {len(all_ch)} canaux, {sfreq:.0f} Hz")

    # Sélection des canaux à afficher
    DEFAULT_CH = [c for c in ["Fp1", "C3", "O1", "C4", "O2", "Fz", "Cz", "Pz"] if c in all_ch]
    print(f"\n  Canaux disponibles (ex: {' '.join(all_ch[:6])} …)")
    ch_input = ask("Canaux à afficher", " ".join(DEFAULT_CH))
    ch_map = {c.upper(): c for c in all_ch}
    sel_ch = [ch_map[tok.upper()] for tok in ch_input.split() if tok.upper() in ch_map]
    if not sel_ch:
        print("  Aucun canal valide sélectionné, utilisation des canaux par défaut.")
        sel_ch = DEFAULT_CH or all_ch[:8]

    print(f"  Canaux sélectionnés : {sel_ch}")

    # Extraction numpy en µV (pas de pré-filtrage : les contrôles sont dans le dashboard)
    data_v = raw.get_data(picks=sel_ch)   # (n_ch, n_samples) en Volts
    data_uv = data_v * 1e6

    import offline_eeg
    offline_eeg.run_dashboard_from_array(
        data_uv, sfreq, sel_ch,
        title=f"PhysioNet {subject} — {run_label}",
    )


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
    import sys
    from PyQt5.QtWidgets import QApplication, QDialog
    from viz.recording_browser import RecordingBrowserDialog
    import offline_eeg

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = RecordingBrowserDialog()
    if dlg.exec_() == QDialog.Accepted:
        offline_eeg.run_recording_analysis(dlg.selected_npy_path, dlg.selected_meta)


def menu_mapping():
    print()
    print("── Mapping du casque ────────────────────")
    import headset_map
    headset_map.run_mapping_tool()


def menu_dashboard_offline():
    print()
    print("── Dashboard offline (enregistrement) ───")
    import offline_eeg
    offline_eeg.run_offline()


def main():
    print_header()
    print("  1. Analyse offline   (PhysioNet EDF+)")
    print("  2. Temps réel        (OpenBCI Cyton)")
    print("  3. Télécharger data  (PhysioNet)")
    print("  4. Analyser un enregistrement  (dashboard)")
    print("  5. Dashboard offline (enregistrement interactif)")
    print("  6. Mapping du casque (positions 10-20)")
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
    elif choice == "5":
        menu_dashboard_offline()
    elif choice == "6":
        menu_mapping()
    elif choice in ("q", "Q"):
        sys.exit(0)
    else:
        print("  Choix invalide.")
        sys.exit(1)


if __name__ == "__main__":
    main()
