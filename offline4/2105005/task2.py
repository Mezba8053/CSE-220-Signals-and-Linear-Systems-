import numpy as np
import time
import matplotlib.pyplot as plt

def generate_random_signal(n):
   
    return np.random.randn(n)

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
def fft(signal):
   
    N = len(signal)
    
    if N <= 1:
        return signal 

    even_part = fft(signal[0::2])
    odd_part = fft(signal[1::2])

    Twiddle_factor = [np.exp(-2j * np.pi * k / N) * odd_part[k] for k in range(N // 2)]
    
    result = [even_part[k] + Twiddle_factor[k] for k in range(N // 2)] + \
             [even_part[k] - Twiddle_factor[k] for k in range(N // 2)]
    
    return result

def ifft(signal):
   
    N = len(signal)
    
    conjugate_signal = np.conj(signal)
    fft_result = fft(conjugate_signal)
    
    return np.conj(fft_result) / N

def measure_time(transform_func_name, signal):
   
    start_time = time.perf_counter()
    transform_func_name(signal)
    return time.perf_counter() - start_time

def compare_performance():
    
    
    sizes = [2**k for k in range(2, 11)] 
    
    dft_times = []
    fft_times = []
    idft_times=[]
    ifft_times=[]
    for n in sizes:
        signal = generate_random_signal(n)
        
        dft_samples = []
        for _ in range(10):
            dft_samples.append(measure_time(dft, signal))
        dft_times.append(np.mean(dft_samples))
        
        fft_samples = []
        for _ in range(10):
            fft_samples.append(measure_time(fft, signal))
        fft_times.append(np.mean(fft_samples))
        idft_samples = []
        for _ in range(10):
            idft_samples.append(measure_time(idft, signal))
        idft_times.append(np.mean(idft_samples))
        ifft_samples = []
        for _ in range(10):
            ifft_samples.append(measure_time(ifft, signal))
        ifft_times.append(np.mean(ifft_samples))
        
    

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, dft_times, marker='o', label='DFT')
    plt.plot(sizes, fft_times, marker='s', label='FFT')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('DFT vs FFT Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, idft_times, marker='o', label='IDFT')
    plt.plot(sizes, ifft_times, marker='s', label='IFFT')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('IDFT vs IFFT Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    print("Input Size | DFT Time (s) | FFT Time (s)")
    print("-----------------------------------------")
    for n, dft_t, fft_t in zip(sizes, dft_times, fft_times):
        print(f"{n:10d} | {dft_t:12.6f} | {fft_t:12.6f}")
    print("-----------------------------------------")
    print("Input Size | IDFT Time (s) | IFFT Time (s)")
    print("-----------------------------------------")
    for n, idft_t, ifft_t in zip(sizes, idft_times, ifft_times):
        print(f"{n:10d} | {idft_t:12.6f} | {ifft_t:12.6f}")

compare_performance()