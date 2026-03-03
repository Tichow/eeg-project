import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

def main():
    # 1. Configuration OpenBCI
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM03H2DU" 
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    try:
        board.prepare_session()
        
        # Configuration Gain 6x
        print("Configuring Gain to 6x...")
        for i in range(1, 9):
            board.config_board(f"x{i}030110X")
            time.sleep(0.02)
            
        board.start_stream()

        # 2. Setup de la figure Matplotlib
        fig, axes = plt.subplots(len(eeg_channels), 1, figsize=(10, 8), sharex=True)
        fig.canvas.manager.set_window_title('OpenBCI Real-Time 8 Channels (Gain 6x)')
        lines = []
        
        # On définit une fenêtre de visualisation de 10 secondes
        n_samples = sampling_rate * 10
        x_axis = np.linspace(0, 10, n_samples)

        colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                  '#911eb4', '#42d4f4', '#f032e6', '#bfef45']

        for i, ax in enumerate(axes):
            line, = ax.plot(x_axis, np.zeros(n_samples), lw=1, color=colors[i % len(colors)])
            lines.append(line)
            ax.set_ylabel(f"Ch {i+1}")
            ax.set_ylim(-300, 300) # Échelle en microvolts (standard EEG)
            ax.grid(True, alpha=0.3)

        plt.xlabel("Temps (secondes)")

        # 3. Fonction de mise à jour (Boucle temps réel)
        def update(frame):
            # Récupère les données récentes (sans vider le buffer principal)
            data = board.get_current_board_data(n_samples)
            
            if data.shape[1] < n_samples:
                return lines

            for i, channel in enumerate(eeg_channels):
                channel_data = data[channel].copy()

                # --- FILTRAGE INDISPENSABLE ---
                # Retrait de l'offset (Detrend)
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                # Filtre Passe-bande 1-50Hz
                DataFilter.perform_bandpass(channel_data, sampling_rate, 1.0, 50.0, 4, 
                                          FilterTypes.BUTTERWORTH.value, 0)
                # Filtre Notch 50Hz (Mac/Europe)
                DataFilter.perform_bandstop(channel_data, sampling_rate, 48.0, 52.0, 4, 
                                          FilterTypes.BUTTERWORTH.value, 0)

                # Mise à jour du graphique
                lines[i].set_ydata(channel_data)

            return lines

        # Lancement de l'animation
        ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        print("Session fermée.")

if __name__ == "__main__":
    main()