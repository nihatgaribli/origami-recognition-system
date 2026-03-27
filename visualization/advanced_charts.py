import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from visualization._db_config import get_models_data
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_difficulty_vs_sheets_heatmap():
    """2D heatmap showing difficulty vs piece count."""
    df = get_models_data()
    # Normalize difficulty to integer 1-5
    df = df.copy()
    if 'difficulty' in df.columns:
        df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').round().astype('Int64')
        df['difficulty'] = df['difficulty'].clip(lower=1, upper=5)
    max_pieces = int(df[df['pieces'].notna()]['pieces'].max())
    max_pieces = min(max_pieces, 20)  # Cap at 20 for readability
    df_clean = df[df['difficulty'].notna() & df['pieces'].notna() & (df['pieces'] <= max_pieces)]
    
    if len(df_clean) < 10:
        print(" Not enough data for heatmap")
        return
    
    # Create pivot table
    pivot = df_clean.groupby(['difficulty', 'pieces']).size().unstack(fill_value=0)

    # Ensure difficulty rows 1..5 exist (even if counts are zero)
    diff_rows = [1, 2, 3, 4, 5]
    pivot = pivot.reindex(diff_rows, fill_value=0)

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5,
                     cbar_kws={'label': 'Model Count'}, linecolor='gray')

    plt.xlabel('Count Pieces', fontsize=12, fontweight='bold')
    # Map numeric difficulty to descriptive labels
    diff_labels = {1: 'Simple', 2: 'Medium', 3: 'Intermediate', 4: 'Complex', 5: 'Super complex'}
    ax.set_yticklabels([diff_labels.get(int(v), str(v)) for v in pivot.index], rotation=0)
    ax.set_ylabel('Difficulty Level', fontsize=12, fontweight='bold')
    plt.title('Difficulty vs Piece Count Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'difficulty_sheets_heatmap.png'), dpi=300, bbox_inches='tight')
    print(" Saved: difficulty_sheets_heatmap.png")
    plt.close()


def plot_correlation_matrix():
    """Correlation matrix for numerical features."""
    df = get_models_data()
    
    numerical_cols = ['year_created', 'pieces', 'difficulty']
    df_num = df[numerical_cols].dropna()
    
    if len(df_num) < 10:
        print(" Not enough data for correlation matrix")
        return
    
    corr = df_num.corr()
    
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    print(" Saved: correlation_matrix.png")
    plt.close()


def plot_paper_shape_difficulty_violin():
    """Violin plot showing difficulty distribution across paper shapes."""
    df = get_models_data()
    df = df.copy()
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').round().astype('Int64')
    df['difficulty'] = df['difficulty'].clip(lower=1, upper=5)
    df_clean = df[df['difficulty'].notna() & df['paper_shape'].notna()]
    
    # Get top 8 most common paper shapes
    top_shapes = df_clean['paper_shape'].value_counts().head(8).index
    df_filtered = df_clean[df_clean['paper_shape'].isin(top_shapes)]
    
    if len(df_filtered) < 20:
        print(" Not enough data for violin plot")
        return
    
    plt.figure(figsize=(14, 8))
    
    sns.violinplot(data=df_filtered, x='paper_shape', y='difficulty', palette='muted', 
                   inner='box', linewidth=1.5)
    
    plt.xlabel('Paper Shape', fontsize=12, fontweight='bold')
    # Display difficulty as named categories instead of numbers
    diff_labels = {1: 'Simple', 2: 'Medium', 3: 'Intermediate', 4: 'Complex', 5: 'Super complex'}
    ax = plt.gca()
    ax.set_ylabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([diff_labels[i] for i in [1, 2, 3, 4, 5]])
    ax.set_ylim(0.5, 5.5)
    plt.title('Difficulty Distribution by Paper Shape (Top 8 Shapes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'paper_difficulty_violin.png'), dpi=300, bbox_inches='tight')
    print(" Saved: paper_difficulty_violin.png")
    plt.close()


def plot_complexity_scatter():
    """Scatter plot: Difficulty vs Pieces with cuts/glue indicators."""
    df = get_models_data()
    df = df.copy()
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').round().astype('Int64')
    df['difficulty'] = df['difficulty'].clip(lower=1, upper=5)
    df['pieces'] = pd.to_numeric(df['pieces'], errors='coerce').astype('Int64')
    df_clean = df[df['difficulty'].notna() & df['pieces'].notna() & (df['pieces'] <= 15)]
    
    if len(df_clean) < 10:
        print(" Not enough data for scatter plot")
        return
    
    plt.figure(figsize=(12, 8)) 
    
    # Create categories
    df_clean['category'] = 'Normal'
    df_clean.loc[df_clean['uses_cutting'] & df_clean['uses_glue'], 'category'] = 'Cuts + Glue'
    df_clean.loc[df_clean['uses_cutting'] & ~df_clean['uses_glue'], 'category'] = 'Cuts Only'
    df_clean.loc[~df_clean['uses_cutting'] & df_clean['uses_glue'], 'category'] = 'Glue Only'
    
    colors = {'Normal': '#2ecc71', 'Cuts Only': '#e74c3c', 
              'Glue Only': '#f39c12', 'Cuts + Glue': '#8e44ad'}
    
    for category, color in colors.items():
        subset = df_clean[df_clean['category'] == category]
        plt.scatter(subset['pieces'], subset['difficulty'], 
                   label=category, alpha=0.6, s=100, c=color, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Count Pieces', fontsize=12, fontweight='bold')
    plt.ylabel('Difficulty Level', fontsize=12, fontweight='bold')
    plt.title('Model Complexity Analysis: Difficulty vs Pieces', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Difficulty labels for Y-axis
    difficulty_labels = {1: 'Simple', 2: 'Medium', 3: 'Intermediate', 4: 'Complex', 5: 'Super complex'}
    plt.yticks([1, 2, 3, 4, 5], [difficulty_labels[i] for i in [1, 2, 3, 4, 5]])
    
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'complexity_scatter.png'), dpi=300, bbox_inches='tight')
    print(" Saved: complexity_scatter.png")
    plt.close()


def plot_advanced_summary():
    """Multi-panel summary dashboard."""
    df = get_models_data()
    df = df.copy()
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').round().astype('Int64')
    df['difficulty'] = df['difficulty'].clip(lower=1, upper=5)
    # Create human-readable labels for difficulty
    diff_labels = {1: 'Simple', 2: 'Medium', 3: 'Intermediate', 4: 'Complex', 5: 'Super complex'}
    df['difficulty_label'] = df['difficulty'].map(diff_labels)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Difficulty distribution (left)
    ax1 = fig.add_subplot(gs[0, 0])
    df_diff = df[df['difficulty_label'].notna()]
    # Ensure order 1..5
    ordered = ['Simple', 'Medium', 'Intermediate', 'Complex', 'Super complex']
    difficulty_counts = df_diff['difficulty_label'].value_counts().reindex(ordered, fill_value=0)
    bars1 = ax1.bar(range(len(difficulty_counts)), difficulty_counts.values, color='skyblue', edgecolor='black', width=0.8)
    ax1.set_title('Difficulty Distribution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Difficulty')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(len(difficulty_counts)))
    ax1.set_xticklabels(difficulty_counts.index, rotation=30)
    for i, v in enumerate(difficulty_counts.values):
        ax1.text(i, v + max(1, int(v*0.02)), str(int(v)), ha='center', va='bottom', fontweight='bold')
    
    # 2. Cuts vs Glue pie (right)
    ax3 = fig.add_subplot(gs[0, 1])
    # Count categories: cuts only, glue only, both, neither
    df['uses_cutting'] = df['uses_cutting'].fillna(False).astype(bool)
    df['uses_glue'] = df['uses_glue'].fillna(False).astype(bool)
    both = int(((df['uses_cutting']) & (df['uses_glue'])).sum())
    cuts_only = int(((df['uses_cutting']) & (~df['uses_glue'])).sum())
    glue_only = int(((~df['uses_cutting']) & (df['uses_glue'])).sum())
    neither = int(((~df['uses_cutting']) & (~df['uses_glue'])).sum())
    techniques = [('Cuts Only', cuts_only), ('Glue Only', glue_only), ('Both', both), ('Neither', neither)]
    labels, sizes = zip(*[(l, s) for l, s in techniques if s > 0])
    colors = ['#e74c3c', '#f39c12', '#8e44ad', '#95a5a6']
    wedges, texts = ax3.pie(sizes, labels=None, startangle=90, colors=colors[:len(sizes)])
    total = sum(sizes)
    legend_labels = [f"{lab}: {cnt} ({cnt/total*100:.1f}%)" for lab, cnt in zip(labels, sizes)]
    ax3.legend(wedges, legend_labels, title='Techniques', loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_title('Techniques Used', fontweight='bold', fontsize=12)
    
    # 4. Top paper shapes (full width bottom)
    ax4 = fig.add_subplot(gs[1, :])
    paper_top = df['paper_shape'].value_counts().head(10)
    bars4 = ax4.barh(range(len(paper_top)), paper_top.values, color=sns.color_palette('viridis', len(paper_top)))
    ax4.set_yticks(range(len(paper_top)))
    ax4.set_yticklabels(paper_top.index)
    # annotate counts at end of bars
    for i, b in enumerate(bars4):
        ax4.text(b.get_width() + max(1, int(b.get_width()*0.02)), b.get_y() + b.get_height()/2, str(int(b.get_width())), va='center', fontweight='bold')
    ax4.set_title('Top 10 Paper Shapes', fontweight='bold')
    ax4.set_xlabel('Count')
    
    fig.suptitle('Origami Database Advanced Summary', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'advanced_summary.png'), dpi=300, bbox_inches='tight')
    print(" Saved: advanced_summary.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Advanced Visualizations...\n")
    
    try:
        plot_difficulty_vs_sheets_heatmap()
        plot_correlation_matrix()
        plot_paper_shape_difficulty_violin()
        plot_complexity_scatter()
        plot_advanced_summary()
        
        print("\n All advanced visualizations generated successfully!")
        
    except Exception as e:
        print(f" Error: {e}")
