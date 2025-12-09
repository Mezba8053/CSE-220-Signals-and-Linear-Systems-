import numpy as np
import matplotlib.pyplot as plt

def parabola(x):
    return np.where((-2 <= x) & (x <= 2), x**2, 0)  # Parabolic

def triangular(x):
    period = 2 * 2
    y = (x % period) / period  
    return 4 * np.abs(y - 0.5) - 1 # Triangular wave

def sawtooth(x):
    period =  2*2
    return 2 * ((x + 2) % period) / period - 1  # Sawtooth wave

def rectengular(x):
    return np.where((-2 <= x) & (x <= 2), 1, 0)  # Rectangular pulse

# Fourier Transform
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_real = np.zeros(num_freqs)
    ft_imag = np.zeros(num_freqs)

    for i, freq in enumerate(frequencies):
        cosine_term = np.cos(2 * np.pi * freq * sampled_times)
        sine_term = np.sin(2 * np.pi * freq * sampled_times)
        ft_real[i] = np.trapezoid(signal * cosine_term, sampled_times)
        ft_imag[i] = np.trapezoid(signal * sine_term, sampled_times)

    return ft_real, ft_imag


def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    real_part, imag_part = ft_signal

    for i, t in enumerate(sampled_times):
        cos_term = np.cos(2 * np.pi * frequencies * t)
        sin_term = np.sin(2 * np.pi * frequencies * t)
        reconstructed_signal[i] = np.trapezoid(real_part * cos_term - imag_part * sin_term, frequencies)

    return reconstructed_signal

def plot_signals(x_values, original, reconstructed, title):
    plt.figure(figsize=(12, 4))
    plt.plot(x_values, original, label="Original", color="blue")
    plt.plot(x_values, reconstructed, label="Reconstructed", color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_frequency_spectrum(frequencies, ft_signal, title):
    ft_magnitude = np.sqrt(ft_signal[0]**2 + ft_signal[1]**2)
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, ft_magnitude, label="Frequency Spectrum", color="green")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    x_values = np.linspace(-10, 10, 1000)
    frequency_ranges = [
        np.linspace(-1, 1, 500),
        np.linspace(-2, 2, 500),
        np.linspace(-5, 5, 500)
    ]

    functions = [parabola, triangular, sawtooth, rectengular]
    function_names = ["Parabolic", "Triangular", "Sawtooth", "Rectangular"]

    for i in range(len(functions)):
        func = functions[i]
        name = function_names[i]
        y_values = func(x_values)  

        plt.figure(figsize=(12, 4))
        plt.plot(x_values, y_values, label=f"Original {name} Function", color="blue")
        plt.title(f"Original {name} Function (Time Domain)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

        for freq_range in frequency_ranges:
            ft_data = fourier_transform(y_values, freq_range, x_values)

            reconstructed_signal = inverse_fourier_transform(ft_data, freq_range, x_values)

            plot_frequency_spectrum(freq_range, ft_data,
                                     f"Frequency Spectrum of {name} (Freq Range: {freq_range[0]} to {freq_range[-1]})")

            plot_signals(x_values, y_values, reconstructed_signal,
                         f"{name} Function (Freq Range: {freq_range[0]} to {freq_range[-1]})")
























# import numpy as np
# import matplotlib.pyplot as plt

# # Define the interval and function
# x_values = np.linspace(-5, 5, 1000) 
# y_values = np.where((-2 <= x_values) & (x_values <= 2), x_values**2, 0)

# # Plot the original function
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original y = x^2")
# plt.title("Original Function (y = x^2)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()
# # Define sampled times and frequencies
# sampled_times = x_values 
# # frequencies = np.linspace(-1, 1, 500)  
# # frequencies=np.linspace(-2,2,500)
# frequencies=np.linspace(-5,5,500)

# # Fourier Transform
# def fourier_transform(signal, frequencies, sampled_times):
#     num_freqs = len(frequencies)
#     ft_result_real = np.zeros(num_freqs)
#     ft_result_imag = np.zeros(num_freqs)

#     for i, freq in enumerate(frequencies):
#         cosine_term = np.cos(2 * np.pi * freq * sampled_times)
#         sine_term = np.sin(2 * np.pi * freq * sampled_times)
#         ft_result_real[i] = np.trapezoid(signal * cosine_term, sampled_times)
#         ft_result_imag[i] = np.trapezoid(signal * sine_term, sampled_times)

#     return ft_result_real, ft_result_imag

# # Apply FT to the sampled data
# ft_data = fourier_transform(y_values, frequencies, sampled_times)

# # Plot the frequency spectrum
# plt.figure(figsize=(12, 6))
# plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
# plt.title("Frequency Spectrum of y = x^2")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.show()

# def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
#     n = len(sampled_times)
#     reconstructed_signal = np.zeros(n)
#     real_part, imag_part = ft_signal 

#     for i, t in enumerate(sampled_times):
#         cos_term = np.cos(2 * np.pi * frequencies * t)
#         sin_term = np.sin(2 * np.pi * frequencies * t)
#         # Real part of inverse FT
#         reconstructed_signal[i] = np.trapezoid(real_part * cos_term - imag_part * sin_term, frequencies)

#     return reconstructed_signal
 
        

    

# # Reconstruct the signal from the FT data
# reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)

# # Plot the original and reconstructed functions for comparison
# plt.figure(figsize=(12, 4))
# plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
# plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
# plt.title("Original vs Reconstructed Function (y = x^2)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()
# # if __name__=="__main__":
# #     range=[1,2,5]
# #     frequencies=np.linspace(-range[0],range[0],500)
# #     frequencies=np.linspace(-range[1],range[1],500)
    
    


