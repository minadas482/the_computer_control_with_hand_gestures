
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_scaled_boxplot():
    print("Loading telemetry data...")
    try:
        df_python = pd.read_csv('python_latency.csv')
        df_cpp = pd.read_csv('cpp_latency.csv')
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return

    # Add environment labels
    df_python['Environment'] = 'Python (Interpreter Baseline)'
    df_cpp['Environment'] = 'C++ (AYNIYH Compiled)'

    # Combine into a single DataFrame
    df = pd.concat([df_python, df_cpp], ignore_index=True)

    # Generate Publication-Ready Boxplot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # showfliers=False removes the extreme outliers (the 7ms spikes) from drawing,
    # allowing the main body of the data to take up the visual space.
    ax = sns.boxplot(x='Environment', y='Latency_ms', data=df, 
                     palette="Set2", width=0.5, showfliers=False)
    
    plt.title('Inference Latency Comparison: Python vs. C++ Execution', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Inference Latency (Milliseconds)', fontsize=12)
    plt.xlabel('Execution Environment', fontsize=12)
    
    # Strictly reduce the scale of the Y-axis to frame the sub-millisecond data perfectly
    plt.ylim(0.0, 0.4) 

    # Save the figure
    plt.savefig('scaled_latency_boxplot.png', dpi=300, bbox_inches='tight')
    print("\nSuccess! The scaled boxplot has been saved as 'scaled_latency_boxplot.png'.")
    plt.show()

if __name__ == "__main__":
    generate_scaled_boxplot()
