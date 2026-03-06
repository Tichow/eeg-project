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


def menu_physionet_explorer():
    """Explorer un run PhysioNet dans le dashboard interactif (64 canaux)."""
    print()
    print("── PhysioNet — Explorer un run ──────────")
    print("  Runs principaux :")
    print("    1  — Baseline yeux ouverts")
    print("    2  — Baseline yeux fermés")
    print("    4  — Imagerie motrice main G/D")
    print("    8  — Imagerie motrice main G/D (répétition)")
    print("    12 — Imagerie motrice main G/D (répétition)")
    print()

    try:
        subject_id = int(ask("Numéro de sujet (1-109)", "1"))
        run_id     = int(ask("Numéro de run   (1-14)",  "2"))
    except (ValueError, TypeError):
        print("  Entrée invalide.")
        return

    import physionet_eeg
    physionet_eeg.run_physionet_dashboard(subject_id, run_id)


def menu_physionet_erp():
    """Vue ERP : formes d'onde moyennées, PSD et carte topo par condition."""
    print()
    print("── PhysioNet — ERP / Moyennes ───────────")
    print("  Recommandé : runs d'imagerie motrice (3-14) avec T0/T1/T2")
    print("  Exemple : run 4 — Imagerie motrice main G/D")
    print()

    try:
        subject_id = int(ask("Numéro de sujet (1-109)", "1"))
        run_id     = int(ask("Numéro de run   (1-14)",  "4"))
    except (ValueError, TypeError):
        print("  Entrée invalide.")
        return

    import physionet_eeg
    physionet_eeg.run_physionet_erp(subject_id, run_id)


def menu_physionet_comparison():
    """Comparer deux runs PhysioNet (ex: yeux ouverts vs fermés)."""
    print()
    print("── PhysioNet — Comparer deux runs ───────")
    print("  Exemple classique : R01 (yeux ouverts) vs R02 (yeux fermés)")
    print("  → met en évidence le pic alpha sur O1/Oz/O2 (effet Berger)")
    print()

    try:
        subject_id = int(ask("Numéro de sujet (1-109)", "1"))
        run_a      = int(ask("Run principal   (1-14)",  "1"))
        run_b      = int(ask("Run référence   (1-14)",  "2"))
    except (ValueError, TypeError):
        print("  Entrée invalide.")
        return

    import physionet_eeg
    physionet_eeg.run_physionet_comparison(subject_id, run_a, run_b)


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
    print("  1. PhysioNet — Explorer un run      (dashboard interactif, 64 canaux)")
    print("  2. PhysioNet — Comparer deux runs   (effet Berger, imagerie motrice)")
    print("  3. PhysioNet — ERP / Moyennes       (formes d'onde, PSD, carte topo)")
    print("  4. PhysioNet — Télécharger données")
    print("  5. OpenBCI   — Temps réel           (Cyton)")
    print("  6. OpenBCI   — Analyser enregistrement")
    print("  7. OpenBCI   — Dashboard offline    (enregistrement interactif)")
    print("  8. Mapping du casque                (positions 10-20)")
    print("  q. Quitter")
    print()

    choice = ask("Choix", "1")

    if choice == "1":
        menu_physionet_explorer()
    elif choice == "2":
        menu_physionet_comparison()
    elif choice == "3":
        menu_physionet_erp()
    elif choice == "4":
        menu_download()
    elif choice == "5":
        menu_realtime()
    elif choice == "6":
        menu_recording()
    elif choice == "7":
        menu_dashboard_offline()
    elif choice == "8":
        menu_mapping()
    elif choice in ("q", "Q"):
        sys.exit(0)
    else:
        print("  Choix invalide.")
        sys.exit(1)


if __name__ == "__main__":
    main()
