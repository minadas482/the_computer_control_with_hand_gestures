
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_real_data():
    print("Loading actual CSV files...")
    df_python = pd.read_csv('python_latency.csv')
    df_cpp = pd.read_csv('cpp_latency.csv')

    df_python['Environment'] = 'Python (Interpreter)'
    df_cpp['Environment'] = 'C++ (Compiled)'

    df = pd.concat([df_python, df_cpp], ignore_index=True)

    # 1. Print Real Stats
    print("\n--- ACTUAL STATISTICAL SUMMARY (10,000 Frames) ---")
    stats = df.groupby('Environment')['Latency_ms'].agg(['mean', 'median', 'std', 'max']).round(4)
    print(stats)

    # 2. Generate a High-Resolution Density Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # We use a KDE (Kernel Density Estimate) plot because the times are so small
    sns.kdeplot(data=df, x='Latency_ms', hue='Environment', fill=True, common_norm=False, palette="Set1", alpha=0.5)
    
    plt.title('Inference Latency Density: Python vs. C++ (Microsecond Scale)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Inference Latency (Milliseconds)', fontsize=12)
    plt.ylabel('Density (Frequency of Occurrence)', fontsize=12)
    
    # Restrict the X-axis to focus on the core data, ignoring the rare 7ms outliers
    plt.xlim(0, 1.0) 

    plt.savefig('real_latency_density.png', dpi=300, bbox_inches='tight')
    print("\nGraph successfully saved as 'real_latency_density.png'.")
    plt.show()

if __name__ == "__main__":
    analyze_real_data()
