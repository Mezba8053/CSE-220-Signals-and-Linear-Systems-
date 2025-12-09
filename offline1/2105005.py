import os
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt

class DiscreteSignal:
    values = None
    
    def __init__(self, INF):
        self.INF = INF
        if self.values is None:
          self.values = np.zeros(2 * INF + 1) 

    def set_value_at_time(self, time, value):
        if -self.INF <= time <= self.INF:
            print('time',time+self.INF,value)
            self.values[time+self.INF ] = value
        else:
            print("Time index out of range")

    def shift_signal(self, shift):
    
        newvalues = np.zeros_like(self.values)
        if shift > 0:
            for i in range(-self.INF, self.INF + 1):
             shifted_index = i + shift + self.INF
            

        # Check that shifted_index remains within bounds of newvalues
             if 0 <= shifted_index <=self.INF:
                newvalues[-shifted_index+self.INF] = self.values[i + 2*self.INF]
                
        else:
            for i in range(-self.INF, self.INF + 1):
                shifted_index = i - shift + self.INF
               

                # Check that shifted_index remains within bounds of newvalues
                if 0 <= shifted_index <= 2*self.INF:
                  
                    newvalues[shifted_index] = self.values[i + self.INF]

    # Create a new DiscreteSignal instance with shifted values
        Discrete = DiscreteSignal(self.INF)
        Discrete.values = newvalues
        # print('Shifted values:', Discrete.values)
        return Discrete

    def add(self, other):
        if self.INF != other.INF:
            raise ValueError("Signals must have the same INF")
            print("Signals must have the same INF")
        Discrete=DiscreteSignal(self.INF)   
        Discrete.values=self.values + other.values
        return Discrete

    def multiply(self, other):
        if self.INF != other.INF:
            raise ValueError("Signals must have the same INF")
            print("Signals must have the same INF")
        Discrete=DiscreteSignal(self.INF)
        Discrete.values=self.values * other.values
        return Discrete

    def multiply_const_factor(self, scaler):
        Discrete=DiscreteSignal(self.INF)
        Discrete.values=self.values * scaler
        return Discrete

    def plot(self):
        plt.stem(range(-self.INF, self.INF + 1), self.values)
        plt.xlabel('Time index')
        plt.ylabel('Signal value')
        plt.title('Discrete Signal')
        plt.show()
class ContinuousSignal:
    INF = 3
    func = None
    values = None
    time_range = np.linspace(-INF,INF, 1000) 
    
    def __init__(self, func):
        self.func = np.vectorize(func)
     
            
    def func_value(self, func, time):
        return func(time)  
    def value(self):
        array=np.zeros(2*self.INF+1)
        for i in range(-self.INF, self.INF + 1):
            array[i]=self.values[i+self.INF]
    def shift(self, shift):
     signal = lambda x: self.func(x-shift)
     

     return ContinuousSignal(signal)

    def add(self, other):
        Continuous=ContinuousSignal(self.func)
        Continuous.values = np.zeros_like((self.time_range)+other.time_range)
        Continuous.values = self.values + other.values
        return Continuous
    def sub(self, other):
        Continuous=ContinuousSignal(self.func)
        Continuous.values = np.zeros_like(self.time_range)
        for index, i in enumerate(self.time_range):
            if self.values[index] < other.values[index]:
                Continuous.values[index] = 0
            else:
                Continuous.values[index] = self.values[index] - other.values[index]
        return Continuous
    def multiply(self, other):
        Continuous=ContinuousSignal(self.func)
        Continuous.values = np.zeros_like((self.time_range)+other.time_range)
        Continuous.values = self.values * other.values
        return Continuous
    def multiply_const_factor(self, scaler):
        Continuous=ContinuousSignal(self.func)
        Continuous.values = np.zeros_like(self.time_range)
        Continuous.values = self.values * scaler
        return Continuous
 
    def plot(self):
        time_range = np.linspace(-self.INF, self.INF, 1000)     
        plt.plot(time_range, self.values)
        plt.xlabel('Time index')
        plt.ylabel('Signal value')
        plt.title('Continuous Signal')
        plt.show()
        
class LTI_Discrete:
    
    impulse_response=None
    INF=5
    
    def __init__(self, impulse_response):
        self.impulse_response = DiscreteSignal(self.INF)
        self.impulse_response = impulse_response
        
    def linear_combination_of_impulses(self,input_signal):
        array = np.zeros(2 * self.INF + 1)
        output_signal = DiscreteSignal(self.INF)
        temp_output_signal = DiscreteSignal(self.INF)
        # plt.figure()
        # plt.xlabel('n(Time index)')
        # plt.ylabel('x[n]')
        
        # input_signal.plot()
        unit_impulse=DiscreteSignal(self.INF)
        # unit_impulse.set_value_at_time(0,1)
        for i in range(0, self.INF+1):
            # print('i',i)
            unit_impulse.set_value_at_time(i,1)
        # unit_impulse.plot()
        output_signal = DiscreteSignal(self.INF)
        individual_signals = []
        for i in range(-self.INF, self.INF +1):
            # plt.title(print('fi[n-]',i,'*x[',i,']'))
            temp_signal= DiscreteSignal(self.INF)
            temp_output_signal = DiscreteSignal(self.INF)
            # temp_input_signal = self.impulse_response
            temp_signal=unit_impulse.shift_signal(-i)
            val=input_signal.values[i+self.INF]*temp_signal.values[i+self.INF]
            temp_output_signal.set_value_at_time(i,val)
            individual_signals.append((i, temp_output_signal))
            output_signal = output_signal.add(temp_output_signal)
        return  individual_signals, output_signal
    def output(self, input_signal): 
        array = np.zeros(2 * self.INF + 1)
        output_signal = DiscreteSignal(self.INF)
        temp_output_signal = DiscreteSignal(self.INF)
        individual_signals = []
        for i in range(-self.INF, self.INF + 1):
            # plt.title(print('fi[n-]',i,'*x[',i,']'))
            temp_signal= DiscreteSignal(self.INF)
            temp_input_signal=temp_output_signal = DiscreteSignal(self.INF)
            temp_signal=self.impulse_response.shift_signal(-i)
            temp_output_signal=temp_signal.multiply_const_factor(input_signal.values[i+self.INF])
            for j in range(-self.INF, self.INF + 1):
                # output_signal.values[j] += temp_output_signal.values[j]
                array[j+self.INF] += temp_output_signal.values[j+self.INF]
            #  array[i+self.INF]+=temp_output_signal.values[i+self.INF]
            individual_signals.append((i, temp_output_signal))
            # self.plot_output_response(temp_output_signal,i)
            
        for j in range(-self.INF, self.INF + 1):
                output_signal.values[j+self.INF]=array[j+self.INF] 
        
            
            
        # self.plot_output_sum(output_signal)
        
        return individual_signals,output_signal
def plot_discrete_outputs(individual_signals, output_signal, Title_prefix=None):
    INF = 5  # Set this according to your DiscreteSignal implementation
    x_axis = np.arange(-INF, INF + 1)

    # Create a figure to plot individual signals
    num_signals = len(individual_signals)+1  # Number of individual signals + 1 for the output signal
    col_num = 3  # Number of columns for subplots
    rows_num = (num_signals + col_num - 1) // col_num  # Calculate rows needed

    fig, axes = plt.subplots(rows_num, col_num, figsize=(15, 10))
    if(Title_prefix=='δ'):
     fig.suptitle(f'impulses multiplied by coefficient', fontsize=16)
    else:
        fig.suptitle(f'Response of Input Signal', fontsize=16)
    axes = axes.flatten()
    # output_signal.plot()
    # Plot each individual signal
    for i, (delta, signal) in enumerate(individual_signals):
        axes[i].stem(x_axis, signal.values)
        axes[i].set_title(f"{Title_prefix}[n- ({i-INF})]*[{i-INF}]")
        axes[i].set_xlabel('n [Time Index]')
        axes[i].set_ylabel('x[n]')
        axes[i].grid(True)
        axes[i].set_ylim(-1, 4)
        
    if(Title_prefix=='δ'):
     axes[num_signals-1].set_title('Sum')
    else:
     axes[num_signals-1].set_title('Output=sum')
    axes[num_signals-1].stem(x_axis, output_signal.values)
    axes[num_signals-1].set_xlabel('n [Time Index]')
    axes[num_signals-1].set_ylabel('x[n]')
    # axes[num_signals-1].legend()
    axes[num_signals-1].grid(True)
    axes[num_signals-1].set_ylim(-1, 4)


    # Hide any extra subplots if there are fewer signals than axes
    for j in range(num_signals, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if(Title_prefix=='δ'):
     plt.savefig('discrete/lti_impulses_multiplied.png')
    else:
     plt.savefig('discrete/lti_output_multiplied.png')
    plt.show()

    # Plot the overall output signal in a separate figure
    # plt.figure(figsize=(10, 6))
    # if(Title_prefix=='δ'):
    #  plt.stem(x_axis, output_signal.values, label='Sum')
    # else:
    #  plt.stem(x_axis, output_signal.values, label='Output=sum')
    # plt.xlabel('n[t]')
    # plt.ylabel('x[n]')
    # plt.title('Sum')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # def plot_linear_combination_of_impulses(self, input_signal,value):
        
    #         plt.stem(range(-self.INF, self.INF + 1), input_signal.values)
    #         plt.xlabel('n(Time index)')
    #         plt.ylabel('x[n]')
    #         plt.title(r'$\delta[n - ({0})] \cdot x[{0}]$'.format(value))
    #         plt.savefig(f"discrete/linear_combination_{value}.png")
    #         # plt.show()
    #         plt.close()
    # def plot_linear_sum_of_impulses(self, input_signal):
    #     plt.stem(range(-self.INF, self.INF + 1), input_signal.values)
    #     plt.xlabel('n(Time index)')
    #     plt.ylabel('x[n]')
    #     plt.title('sum')
    #     plt.savefig("discrete/linear_sum_of_impulses.png") 
    #     plt.show()  
    #     plt.close()  

    # def plot_output_response(self, input_signal,value):
    #             plt.stem(range(-self.INF, self.INF + 1), input_signal.values)
    #             plt.xlabel('n(Time index)')
    #             plt.ylabel('x[n]')
    #             plt.title('h[n-({})]*x[{}]'.format(value, value))
    #             plt.savefig(f"discrete/output_response_{value}.png")
        
    #             # plt.show()
    #             plt.close()
    # def plot_output_sum(self, input_signal):
    #             plt.stem(range(-self.INF, self.INF + 1), input_signal.values)
    #             plt.xlabel('n(Time index)')
    #             plt.ylabel('x[n]')
    #             plt.title('Output=sum')
    #             plt.savefig("discrete/output.png") 
    #             plt.show()  
    #             plt.close() 
        
        
class LTI_Continuous:
    INF = 3 
    impulse_response = None
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response
    def linear_combination_of_impulses(self,input_signal,delta):
        delta = float(delta)
        
        unit_impulse = ContinuousSignal(lambda x: np.where((x >= 0) & (x < delta), 1.0 / delta, 0))
        n_val = floor(float(self.INF)/delta)
        indices = np.arange(-n_val*delta,n_val*delta,delta,dtype=float)
        coefficients_array = np.zeros(2*n_val)
        for i in range(2*n_val):
            coefficients_array[i] = input_signal.func(indices[i])
        output_signals = [unit_impulse.shift(indices[i]) for i in range(len(indices))]
        # if(delta==0.5):
        #   self.plot_continuous(output_signals,coefficients_array,delta,title_prefix='δ')
        return output_signals,coefficients_array
                
    
    
    def output_approx(self, input_signal, delta):
     delta = float(delta)

    # Get coefficients_array as scalar values
     signal, coefficients_array = self.linear_combination_of_impulses(input_signal, delta)

    # Calculate indices for delayed impulses
     n_val = floor(self.INF / delta)
     indices = np.arange(-n_val, n_val + 1) * delta

    # Ensure coefficients_array and indices have the same length by slicing to the minimum length
     min_length = min(len(indices), len(coefficients_array))
     indices = indices[:min_length]
     coefficients_array = coefficients_array[:min_length]

     return_impulses = [self.impulse_response.shift(indices[i]) for i in range(len(indices))]

     output = ContinuousSignal(
        lambda x: sum([return_impulses[i].func(x) * coefficients_array[i] * delta for i in range(len(indices))])
     )

     time_range = np.linspace(-self.INF, self.INF, 1000)
     return return_impulses, coefficients_array, output


        
     
def plot_continuous_signal(signals, coefficients_array=None, delta=None, title_prefix=None):
    INF = 3
    y_range = (-0.05, 1.05)
    figsize = (15, 12)
    num_plots = len(signals) + 1
    col_num = 3
    rows_num = (num_plots + col_num - 1) // col_num

    fig, axes = plt.subplots(rows_num, col_num, figsize=figsize)
    axes = axes.flatten()
    x_top = INF + 0.2
    x = np.linspace(-INF, INF, 1000) 
    output_final = np.zeros_like(x)  
    for i in range(min(len(signals), len(coefficients_array))):
        signal = signals[i]
        axis = axes[i]
        
        # Calculate y-values for the current signal
        signal_values = signal.func(x)
        if coefficients_array is not None:
            y = signal_values * coefficients_array[i] * delta  
        else:
            y=signal_values
        output_final += y  

        # Plot the current signal
        axis.plot(x, y)
        axis.set_title(f'{title_prefix} (t-({i-2*INF}∇)) x({i-2*INF}∇) ∇', fontsize=10)
        axis.set_xlabel('t (Time)', fontsize=9)
        axis.set_ylabel('x(t)', fontsize=9)
        axis.set_xlim(-x_top, x_top)
        axis.set_ylim(*y_range)
        axis.set_xticks(np.arange(-INF, INF + 1, 1))
        axis.set_yticks(np.arange(0, y_range[1], 0.5))
        axis.grid(True)

    # Plot the final accumulated output
    axis = axes[num_plots - 1]
    print('h',title_prefix) 
    label_text = 'output=sum' if title_prefix =='h' else 'Reconstructed Signal'
    axis.set_title(label_text, fontsize=10)
    axis.set_xlabel('t (Time)', fontsize=9)
    axis.set_ylabel('x(t)', fontsize=9)
    axis.plot(x, output_final, label=label_text, color='purple')
    axis.set_xlim(-x_top, x_top)
    
    axis.grid(True)

    # Remove any extra axes
    for j in range(len(signals) + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Impulses multiplied by coefficients_array', fontsize=16)
    plt.tight_layout()
    if title_prefix=='h':
        print('h+',title_prefix)
        plt.savefig('continuous/Returned impulses multiplied by their coefficients_array.png')
    else:
     plt.savefig('continuous/linear_combination_of_impulses.png')
    plt.show()



 

def plot_input(input_signal,deltas):  
    y_range=(-0.05, 1.05)
    figsize=(15, 10)
    INF = 3
    num_plots = len(deltas)
    col_num = 2
    rows_num = (num_plots + col_num - 1) // col_num
    fig, axes = plt.subplots(rows_num, col_num, figsize=figsize)
    axes = axes.flatten()
    x_top = INF + 0.2
    x = np.linspace(-INF, INF, 1000) 
    lti = LTI_Continuous(ContinuousSignal(lambda x: np.where(x >= 0, 1, 0)))
    
    for i, delta in enumerate(deltas):
        axis = axes[i]
        x = np.linspace(-INF, INF,1000)
        # Obtain delayed signals and coefficients_array for the current delta
        output_signals, coefficients_array = lti.linear_combination_of_impulses(input_signal, delta)
        
        # Initialize `y_approximate` to accumulate the sum of impulses
        y_approximate = sum([output_signals[i].func(x)*coefficients_array[i]*delta for i in range(len(output_signals))])
        y_actual = input_signal.func(x)
        
        # Obtain actual signal values for comparison
        y_actual = input_signal.func(x)
        axis.set_title(f'∇={delta}')
        axis.plot(x, y_approximate, label='Reconstructed', color='orange')
        axis.plot(x, y_actual, label='Original', color='blue')
        axis.set_xlim(-x_top, x_top)
        axis.legend()
        axis.set_xticks(np.arange(-INF, INF + 1, 1))
        
        axis.grid(True)
    
    # Remove any extra subplots
    for j in range(len(deltas), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    plt.savefig('continuous/lti_input.png')
    
    plt.show()


        
def plot_output(input_signal,impulse_response,deltas):  
    figsize=(15, 10)
    INF=3
    num_plots = len(deltas)
    col_num = 2
    rows_num = (num_plots + col_num - 1) // col_num 
    fig,axes =plt.subplots(rows_num,col_num,figsize=figsize)
    axes = axes.flatten()
    x_top = INF+0.2
    x_values = np.linspace(-INF,INF,1000)
    y_values = input_signal.func(x_values)
    output_original= np.cumsum(y_values)*(x_values[1]-x_values[0])
    lti = LTI_Continuous(impulse_response)
    for i,delta in enumerate(deltas):
        axis = axes[i]
        return_impulses,coefficients_array,output = lti.output_approx(input_signal,delta)
        # print('output',output.values,return_impulses.values)
        y_approximate = output.func(x_values)
        axis.set_title(f'∇={delta}')
        axis.plot(x_values,y_approximate,label='y_approximate(t)')
        axis.plot(x_values,output_original,label='y(t)=1-e^(-t)u(t)')
        axis.set_xlim(-x_top, x_top)
        axis.legend()
        axis.set_xticks(np.arange(-INF, INF + 1, 1))
        axis.grid(True)

    for j in range(len(deltas),len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Approximate Output as ∇ tends to 0', fontsize=16)

    plt.tight_layout()
    
    plt.savefig('continuous/lti_output.png')
    
    plt.show()

def main():
    INF = 5
    os.makedirs("discrete", exist_ok=True)
    os.makedirs("continuous", exist_ok=True) 
    signal = DiscreteSignal(INF)
    
    # Setting some values
    signal.set_value_at_time(0, 0.5)
    signal.set_value_at_time(1, 2)
    # print(signal.values)
    # Plotting the original signal
    # signal.plot()
    
#     # Shifting the signal
#     shifted_signal = signal.shift_signal(-3)
#     print(shifted_signal.values)
#     shifted_signal.plot()
    
# #     # Adding two signals
#     signal2 = DiscreteSignal(INF)
#     signal2.set_value_at_time(0, 0.5)
#     signal2.set_value_at_time(1, 1)
    
#     added_signal = signal.add(signal2)
#     added_signal.plot()
    
# #     # Multiplying two signals
#     multiplied_signal = signal.multiply(signal2)
#     multiplied_signal.plot()
    
# #     # Multiplying signal by a constant factor
#     scaled_signal = signal.multiply_const_factor(2)
#     scaled_signal.plot()
    # continuous_Signal=ContinuousSignal(lambda x: np.exp(-x))
    # continuous_Signal.plot()
    # continue_shift=continuous_Signal.shift(-2)
    # continue_shift.plot()
    # continue_add=continuous_Signal.add(continue_shift)
    # continue_add.plot()
    # continue_multiply=continuous_Signal.multiply(continue_shift)
    # continue_multiply.plot()
    # continue_multiply_const=continuous_Signal.multiply_const_factor(2)
    # continue_multiply_const.plot()
    impulse = DiscreteSignal(INF)
    impulse.set_value_at_time(0, 1)
    impulse.set_value_at_time(1, 1)
    impulse.set_value_at_time(2, 1)
    L=LTI_Discrete(impulse)
    individual_signals, output_signal = L.linear_combination_of_impulses(signal)
    plot_discrete_outputs(individual_signals,output_signal,'δ')
    individual_signals_output, output_signal_output = L.output(signal)
    plot_discrete_outputs(individual_signals_output,output_signal_output,'h')
    delta=[0.5,0.05,0.1,0.001]
    impulse_response=ContinuousSignal(lambda x: np.where(x>=0,1,0))
    lti_continuous=LTI_Continuous(impulse_response)
    continuous_Signal=ContinuousSignal(lambda x: np.exp(-x)*(x>=0))
    n,o=lti_continuous.linear_combination_of_impulses(continuous_Signal,delta[0])
    plot_continuous_signal(n,o,delta[0],title_prefix='δ')
    impulse,coeff,out=lti_continuous.output_approx(continuous_Signal,delta[0])
    plot_continuous_signal(impulse,coeff,delta[0],title_prefix='h')
    plot_input(continuous_Signal,delta)
    plot_output(continuous_Signal,impulse_response,delta)

    
if __name__ == "__main__":
    main()


