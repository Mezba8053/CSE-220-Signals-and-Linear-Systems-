import numpy as np
import matplotlib.pyplot as plt

n = 50
samples = np.arange(n) 
sampling_rate = 100
wave_velocity = 8000

# Use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):
    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz
    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
    # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    # noise_for_sigal_A += noise_for_sigal_A * np.random.normal(0, 1, n)
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B,shift_samples)
    
    return signal_A, signal_B

def dft(signal):
  
    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            dft_result[k] += signal[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    
    return dft_result

def idft(signal):
   
    N = len(signal)
    idft_result = np.zeros(N, dtype=complex)
    
    for n in range(N):
        for k in range(N):
            idft_result[n] += signal[k] * np.exp(1j * 2 * np.pi * k * n / N)
        idft_result[n] /= N
    
    return idft_result


def apply_filter(signal, threshold=0.5):
   
    filtered_signal = np.where(np.abs(signal) >= threshold, signal, 0)
    return filtered_signal

def filtering_experiment(signal_A,signal_B):
    filter_signal_A=apply_filter(signal_A)
    filter_signal_B=apply_filter(signal_B)
    cross_corr_filtered = cross_correlation(filter_signal_A,filter_signal_B)
    lag_filtered = sample_lag(cross_corr_filtered)
    distance_filtered = distance_estimation(lag_filtered)
    
    print("\nNoise Filtering Experiment Results:")
   
    print(f"Filtered Lag: {lag_filtered}, Distance: {distance_filtered} meters")
def cross_correlation(signal1, signal2):
   
    dft1 = dft(signal1)
    dft2 = dft(signal2)
    
    dft1_conj = np.conj(dft1)
    
    cross_spectrum =  dft2*dft1_conj
    
    cross_corr = idft(cross_spectrum)
    
    return np.real(cross_corr)

def sample_lag(cross_corr):
    
    n = len(cross_corr)
    max_index = np.argmax(cross_corr)  
    # print(max_index)
    if max_index >n // 2:
        lag = max_index - n  
    else:
        lag = max_index
    return lag


def distance_estimation(lag):
    return float(np.abs(lag) * wave_velocity / sampling_rate)

def plot_results(signal_a, signal_b, dft_a, dft_b, cross_corr, plot_type):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)

    if plot_type == 1:   
        # 1st Plot: Signals
        plt.subplot(2, 1, 1)
        plt.stem(signal_a, linefmt="b-", markerfmt="bo", basefmt=" ", label="Signal A")
        plt.title("Signal A (Station A)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.stem(signal_b, linefmt="g-", markerfmt="go", basefmt=" ", label="Signal B")
        plt.title("Signal B (Station B)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    
    elif plot_type == 2:
        # 2nd Plot: DFT (Magnitude Spectrum)
        plt.subplot(2, 1, 1)
        plt.stem(np.abs(dft_a), linefmt="r-", markerfmt="ro", basefmt=" ", label="Signal A Spectrum")
        plt.title("Frequency Spectrum of Signal (A)")
        plt.xlabel("Frequency Index")
        plt.ylabel("Magnitude")
        plt.xticks(np.arange(0, len(dft_a), step=max(1, len(dft_a)//10)))  
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.stem(np.abs(dft_b), linefmt="m-", markerfmt="mo", basefmt=" ", label="Signal B Spectrum")
        plt.title("Frequency Spectrum of Signal (B)")
        plt.xlabel("Frequency Index")
        plt.ylabel("Magnitude")
        plt.xticks(np.arange(0, len(dft_b), step=max(1, len(dft_b)//10)))  
        plt.legend()
        plt.grid(True)
    
    else:
        # 3rd Plot: Cross-Correlation
        n = len(cross_corr)
        lags = np.arange(-n//2, n//2)
        
        centered_cross_corr = np.roll(cross_corr, n//2)
        
        plt.stem(lags, centered_cross_corr, linefmt="c-", markerfmt="co", basefmt=" ", label="Cross-Correlation")
        plt.title("DFT-based Cross-Correlation")
        plt.xlabel("Lag (samples)")
        plt.ylabel("Correlation")
        plt.grid(True)

        max_corr_index = sample_lag(cross_corr)
        
        plt.axvline(max_corr_index, color="red", linestyle="--", label=f"Simple Lag: {max_corr_index}")
        plt.legend()

        plt.xticks(np.arange(-n//2, n//2, step=5))
    plt.tight_layout()
    plt.show()



signal_A, signal_B = generate_signals()
dft_a = dft(signal_A)
# print(signal_A)
# print(signal_B)
dft_b = dft(signal_B)

cross_corr = cross_correlation(signal_A, signal_B)
lag = sample_lag(cross_corr)
distance = distance_estimation(lag)

print("Sample Lag:",lag)
print(f"Estimated Distance: {distance} meters")
plot_results(signal_A, signal_B, dft_a, dft_b, cross_corr, 1)
plot_results(signal_A, signal_B, dft_a, dft_b, cross_corr, 2)
plot_results(signal_A, signal_B, dft_a, dft_b, cross_corr, 3)
filtering_experiment(signal_A,signal_B)


