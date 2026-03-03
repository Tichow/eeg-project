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


def main():
    print_header()
    print("  1. Analyse offline   (PhysioNet EDF+)")
    print("  2. Temps réel        (OpenBCI Cyton)")
    print("  3. Télécharger data  (PhysioNet)")
    print("  q. Quitter")
    print()

    choice = ask("Choix", "1")

    if choice == "1":
        menu_offline()
    elif choice == "2":
        menu_realtime()
    elif choice == "3":
        menu_download()
    elif choice in ("q", "Q"):
        sys.exit(0)
    else:
        print("  Choix invalide.")
        sys.exit(1)


if __name__ == "__main__":
    main()
