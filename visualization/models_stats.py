import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from visualization._db_config import get_models_data, get_connection
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_difficulty_distribution():
    """Bar chart of difficulty distribution."""
    df = get_models_data()
    # Normalize difficulty to integer 1-5 and map to named categories
    df = df.copy()
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').round().astype('Int64')
    df['difficulty'] = df['difficulty'].clip(lower=1, upper=5)
    diff_map = {1: 'Simple', 2: 'Medium', 3: 'Intermediate', 4: 'Complex', 5: 'Super Complex'}
    df['difficulty_label'] = df['difficulty'].map(diff_map)

    plt.figure(figsize=(10, 6))
    ordered = ['Simple', 'Medium', 'Intermediate', 'Complex', 'Super Complex']
    difficulty_counts = df['difficulty_label'].value_counts().reindex(ordered, fill_value=0)

    sns.barplot(x=difficulty_counts.index, y=difficulty_counts.values, palette='viridis')
    plt.title('Origami Model Difficulty Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Number of Models', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    for i, v in enumerate(difficulty_counts.values):
        plt.text(i, v + max(1, int(v*0.02)), str(int(v)), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'difficulty_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: difficulty_distribution.png")
    plt.close()


def plot_paper_shape_distribution():
    """Pie chart of paper shapes."""
    df = get_models_data()
    df_clean = df[df['paper_shape'].notna()]
    
    plt.figure(figsize=(16, 10))
    paper_counts = df_clean['paper_shape'].value_counts().head(10)
    
    colors = sns.color_palette('Set3')[0:len(paper_counts)]
    
    # No labels on pie chart
    wedges, texts = plt.pie(
        paper_counts.values,
        labels=None,  # No names on pie chart
        autopct=None,  # No percentages on pie chart
        startangle=140, 
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Legend showing names and percentages
    total = paper_counts.sum()
    legend_labels = [f'{name}: {count} ({count/total*100:.1f}%)' 
                     for name, count in paper_counts.items()]
    
    plt.legend(wedges, legend_labels, 
              title="Paper Shapes", 
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=11,
              title_fontsize=13,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    plt.title('Top 10 Paper Shapes Used in Origami Models', fontsize=18, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'paper_shapes.png'), dpi=300, bbox_inches='tight')
    print(" Saved: paper_shapes.png")
    plt.close()


def plot_cuts_glue_usage():
    """Stacked bar chart for cutting and glue usage."""
    df = get_models_data()
    df = df.copy()
    df['uses_cutting'] = df['uses_cutting'].fillna(False).astype(bool)
    df['uses_glue'] = df['uses_glue'].fillna(False).astype(bool)

    cuts_count = df['uses_cutting'].value_counts()
    glue_count = df['uses_glue'].value_counts()
    both_count = int(((df['uses_cutting']) & (df['uses_glue'])).sum())
    neither_count = int(((~df['uses_cutting']) & (~df['uses_glue'])).sum())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Cuts
    colors_cuts = ['#2ecc71', '#e74c3c']
    ax1.bar(['No Cuts', 'Uses Cuts'], [cuts_count.get(False, 0), cuts_count.get(True, 0)],
            color=colors_cuts, edgecolor='black', linewidth=1.5)
    ax1.set_title('Models Using Cuts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Models', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    for i, v in enumerate([cuts_count.get(False, 0), cuts_count.get(True, 0)]):
        ax1.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Glue
    colors_glue = ['#3498db', '#e67e22']
    ax2.bar(['No Glue', 'Uses Glue'], [glue_count.get(False, 0), glue_count.get(True, 0)],
            color=colors_glue, edgecolor='black', linewidth=1.5)
    ax2.set_title('Models Using Glue', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Models', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    for i, v in enumerate([glue_count.get(False, 0), glue_count.get(True, 0)]):
        ax2.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Both cuts & glue vs neither
    bars3 = ax3.bar(['Both Cuts & Glue', 'No Cuts & No Glue'], [both_count, neither_count],
                    color=['#8e44ad', '#7f8c8d'], edgecolor='black', linewidth=0.8, width=0.55, alpha=0.85)
    ax3.set_title('Cuts & Glue Combo', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Models', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 5, str(int(h)), ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cuts_glue_usage.png'), dpi=300, bbox_inches='tight')
    print(" Saved: cuts_glue_usage.png")
    plt.close()


def plot_sheet_count_distribution():
    """Bar chart of piece count distribution."""
    df = get_models_data()
    df = df.copy()
    df['pieces'] = pd.to_numeric(df['pieces'], errors='coerce').astype('Int64')
    df_clean = df[df['pieces'].notna() & (df['pieces'] < 50)]  # Filter outliers

    plt.figure(figsize=(14, 7))

    # Count frequency of each integer piece count and ensure contiguous x-axis
    pieces_counts = df_clean['pieces'].value_counts().sort_index()
    if pieces_counts.shape[0] == 0:
        print(" No piece count data to plot")
        return

    min_p = int(pieces_counts.index.min())
    max_p = int(pieces_counts.index.max())
    idx = list(range(min_p, max_p + 1))
    pieces_counts = pieces_counts.reindex(idx, fill_value=0)

    # Create bar chart
    bars = plt.bar(pieces_counts.index, pieces_counts.values, 
                   color='skyblue', edgecolor='navy', linewidth=1.2, alpha=0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(int(bar.get_x() + bar.get_width()/2.), height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.title('Distribution of Piece Count per Model', fontsize=16, fontweight='bold')
    plt.xlabel('Count Pieces', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Models', fontsize=12, fontweight='bold')
    plt.xticks(idx)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sheet_count_distribution.png'), dpi=300, bbox_inches='tight')
    print(" Saved: sheet_count_distribution.png")
    plt.close()


def _get_image_coverage_df() -> pd.DataFrame:
    """Per-model image coverage + source bucket info."""
    query = """
        WITH per_model AS (
            SELECT
                m.model_id,
                m.model_name_original,
                m.source_url,
                COUNT(i.image_id) AS image_count
            FROM models m
            LEFT JOIN images i ON i.model_id = m.model_id
            GROUP BY m.model_id, m.model_name_original, m.source_url
        )
        SELECT
            model_id,
            model_name_original,
            source_url,
            image_count,
            CASE WHEN image_count > 0 THEN TRUE ELSE FALSE END AS has_image,
            CASE
                WHEN source_url ILIKE '%origami-resource-center.com%' THEN 'ORC'
                WHEN source_url ILIKE '%origami-database.com%' OR source_url ILIKE '%cfc%' THEN 'CFC'
                WHEN source_url IS NULL OR source_url = '' THEN 'Unknown'
                ELSE 'Other'
            END AS source_bucket
        FROM per_model
    """
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def plot_image_coverage_overall():
    """Donut chart for models with vs without image."""
    df = _get_image_coverage_df()
    if df.empty:
        print(" No data for image coverage")
        return

    with_img = int(df['has_image'].sum())
    without_img = int((~df['has_image']).sum())
    total = with_img + without_img

    plt.figure(figsize=(9, 7))
    labels = ['Has Image', 'No Image']
    values = [with_img, without_img]
    colors = ['#2ecc71', '#e74c3c']

    wedges, _ = plt.pie(
        values,
        labels=None,
        colors=colors,
        startangle=120,
        wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 2},
    )

    legend_labels = [
        f'{labels[0]}: {with_img} ({with_img / total * 100:.1f}%)',
        f'{labels[1]}: {without_img} ({without_img / total * 100:.1f}%)',
    ]
    plt.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), title='Coverage')
    plt.title('Model Image Coverage (Overall)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_coverage_overall.png'), dpi=300, bbox_inches='tight')
    print(" Saved: image_coverage_overall.png")
    plt.close()


def plot_image_coverage_by_source():
    """Stacked bar chart: image coverage split by source bucket."""
    df = _get_image_coverage_df()
    if df.empty:
        print(" No data for source coverage")
        return

    summary = (
        df.groupby(['source_bucket', 'has_image'])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: 'No Image', True: 'Has Image'})
    )

    order = ['ORC', 'CFC', 'Other', 'Unknown']
    for key in order:
        if key not in summary.index:
            summary.loc[key] = [0, 0]
    summary = summary.loc[order]

    plt.figure(figsize=(12, 7))
    plt.bar(summary.index, summary['No Image'], color='#e74c3c', label='No Image')
    plt.bar(summary.index, summary['Has Image'], bottom=summary['No Image'], color='#2ecc71', label='Has Image')

    totals = (summary['No Image'] + summary['Has Image']).values
    for i, total in enumerate(totals):
        plt.text(i, total + max(1, int(total * 0.01)), str(int(total)), ha='center', va='bottom', fontweight='bold')

    plt.title('Image Coverage by Source', fontsize=16, fontweight='bold')
    plt.xlabel('Source Bucket', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Models', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_coverage_by_source.png'), dpi=300, bbox_inches='tight')
    print(" Saved: image_coverage_by_source.png")
    plt.close()


def plot_top_models_by_image_count(limit: int = 20):
    """Horizontal bar chart of models with highest image counts."""
    df = _get_image_coverage_df()
    df = df[df['image_count'] > 0].copy()
    if df.empty:
        print(" No models with images to plot")
        return

    top_df = df.sort_values('image_count', ascending=False).head(limit)
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(top_df)), top_df['image_count'], color=sns.color_palette('magma', len(top_df)))
    plt.yticks(range(len(top_df)), top_df['model_name_original'])
    plt.gca().invert_yaxis()
    plt.xlabel('Image Count', fontsize=12, fontweight='bold')
    plt.ylabel('Model Name', fontsize=12, fontweight='bold')
    plt.title(f'Top {limit} Models by Image Count', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    for i, bar in enumerate(bars):
        value = int(top_df.iloc[i]['image_count'])
        plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(value), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_models_by_image_count.png'), dpi=300, bbox_inches='tight')
    print(" Saved: top_models_by_image_count.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Model Statistics Visualizations...\n")
    
    try:
        plot_difficulty_distribution()
        plot_paper_shape_distribution()
        plot_cuts_glue_usage()
        plot_sheet_count_distribution()
        plot_image_coverage_overall()
        plot_image_coverage_by_source()
        plot_top_models_by_image_count()
        
        print("\n All visualizations generated successfully!")
        print("  Check the 'visualization/output/' folder for output files.")
        
    except Exception as e:
        print(f" Error: {e}")