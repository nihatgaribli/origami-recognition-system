import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from visualization._db_config import get_creators_data, get_models_data
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_top_creators():
    """Bar chart of top 20 creators by model count."""
    df = get_creators_data()
    top_creators = df.head(30)
    
    plt.figure(figsize=(14, 8))
    
    colors = sns.color_palette('coolwarm', len(top_creators))
    bars = plt.barh(range(len(top_creators)), top_creators['model_count'], color=colors)
    
    plt.yticks(range(len(top_creators)), top_creators['name_original'])
    plt.xlabel('Number of Models', fontsize=12, fontweight='bold')
    plt.ylabel('Creator Name', fontsize=12, fontweight='bold')
    plt.title('Top 30 Most Productive Origami Creators', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_creators['model_count'])):
        plt.text(value + 0.5, i, str(value), va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_creators.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: top_creators.png")
    plt.close()


def plot_creator_productivity_distribution():
    """Histogram showing creator productivity."""
    df = get_creators_data()
    df_active = df[df['model_count'] > 0]
    
    if df_active.empty:
        print(" No productivity data available yet")
        return

    max_count = df_active['model_count'].max()
    if max_count <= 100:
        bin_size = 10
    elif max_count <= 300:
        bin_size = 25
    elif max_count <= 800:
        bin_size = 50
    else:
        bin_size = 100

    max_edge = int(np.ceil(max_count / bin_size) * bin_size)
    bins = np.arange(0, max_edge + bin_size, bin_size)
    plt.figure(figsize=(14, 7))
    n, _, patches = plt.hist(df_active['model_count'], bins=bins, color='teal', edgecolor='black', alpha=0.7)

    # Highlight extremely sparse bins so singles are visible
    for patch, height in zip(patches, n):
        if height <= 1:
            patch.set_facecolor('#ff8c42')
            patch.set_edgecolor('#b34700')
            patch.set_alpha(0.95)
            patch.set_linewidth(1.1)

    # annotate counts where visible
    for patch, height in zip(patches, n):
        if height >= 1:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, height + max(1, int(height * 0.02)), str(int(height)), ha='center', va='bottom', fontsize=7, fontweight='bold')

    median_val = df_active['model_count'].median()
    mean_val = df_active['model_count'].mean()
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Median: {median_val:.0f}')
    plt.axvline(mean_val, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.1f}')

    plt.xlim(0, max_edge)
    plt.xticks(np.arange(0, max_edge + bin_size, bin_size), rotation=90, ha='center')
    plt.xlabel('Models per Creator', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Creators', fontsize=12, fontweight='bold')
    plt.title('Creator Productivity Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'creator_productivity.png'), dpi=300, bbox_inches='tight')
    print(" Saved: creator_productivity.png")
    plt.close()


def plot_country_distribution():
    """Bar chart of models by country (top 15)."""
    df = get_models_data()
    df_clean = df[df['creator_country'].notna()]
    
    if len(df_clean) == 0:
        print(" No country data available yet")
        return
    
    country_counts = df_clean['creator_country'].value_counts().head(15)
    
    plt.figure(figsize=(12, 7))
    
    colors = sns.color_palette('tab20', len(country_counts))
    bars = plt.bar(range(len(country_counts)), country_counts.values, color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    plt.xticks(range(len(country_counts)), country_counts.index, rotation=45, ha='right')
    plt.xlabel('Country', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Models', fontsize=12, fontweight='bold')
    plt.title('Top 15 Countries by Origami Model Count', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, country_counts.values)):
        plt.text(i, value + 5, str(value), ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'country_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: country_distribution.png")
    plt.close()



if __name__ == "__main__":
    print("Generating Creator Analysis Visualizations...\n")
    
    try:
        plot_top_creators()
        plot_creator_productivity_distribution()
        plot_country_distribution()
        
        print("\n All creator visualizations generated successfully!")
        
    except Exception as e:
        print(f" Error: {e}")
