import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_analysis():
    # Load dataset
    df = pd.read_csv("dataset/flood.csv")
    
    print("=== Dataset Head ===")
    print(df.head())
    
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Descriptive Statistics ===")
    stats = df.describe()
    print(stats)
    
    print("\n=== Check Missing Values ===")
    missing = df.isnull().sum()
    print(missing)
    
    # Ensure static directory exists
    os.makedirs("static/images", exist_ok=True)
    
    # Generate Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Meteorological & Environmental Features Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig("static/images/correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nCorrelation heatmap saved to 'static/images/correlation_heatmap.png'")

if __name__ == "__main__":
    run_analysis()
