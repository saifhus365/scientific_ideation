import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Data
configs = ['No_Synthesis', 'No_RAG', 'Full_System', 'No_Viewpoint', 'No_Critique']
precision_data = {
    'No_Synthesis': [0.630, 0.607, 0.570, 0.509],
    'No_RAG': [0.583, 0.529, 0.482, 0.491],
    'Full_System': [0.542, 0.538, 0.478, 0.477],
    'No_Viewpoint': [0.520, 0.536, 0.452, 0.458],
    'No_Critique': [0.519, 0.489, 0.428, 0.419]
}

novelty_data = {
    'No_Synthesis': [1.157, 1.085, 3.418, 3.660],
    'No_RAG': [1.021, 1.065, 5.944, 5.480],
    'Full_System': [1.103, 1.044, 2.958, 2.819],
    'No_Viewpoint': [1.125, 1.107, 3.639, 3.450],
    'No_Critique': [1.020, 1.038, 5.448, 5.650]
}

# ========== FIGURE 1: Precision@N Bar Chart ==========
def figure1_precision_bars():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(4)
    width = 0.15
    multiplier = 0
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, (config, values) in enumerate(precision_data.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=config, color=colors[idx])
        multiplier += 1
    
    ax.set_xlabel('Precision Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision@N Performance Across Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(['P@3', 'P@5', 'P@10', 'P@20'])
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    plt.savefig('figure1_precision_bars.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_precision_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 2: Novelty Radar Chart ==========
def figure2_novelty_radar():
    categories = ['Historical\nDissimilarity', 'Contemporary\nDissimilarity', 
                  'Contemporary\nImpact', 'Overall\nNovelty']
    N = len(categories)
    
    # Normalize data for better visualization
    max_vals = np.max([novelty_data[c] for c in configs], axis=0)
    normalized_data = {config: [v/m for v, m in zip(novelty_data[config], max_vals)] 
                       for config in configs}
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, config in enumerate(configs):
        values = normalized_data[config]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=config, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
    ax.set_title('Novelty Metrics Comparison (Normalized)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure2_novelty_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_novelty_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 3: Precision-Novelty Trade-off Scatter ==========
def figure3_precision_novelty_scatter():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precision_10 = [precision_data[c][2] for c in configs]
    overall_novelty = [novelty_data[c][3] for c in configs]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, config in enumerate(configs):
        ax.scatter(precision_10[idx], overall_novelty[idx], 
                  s=300, alpha=0.7, c=colors[idx], 
                  edgecolors='black', linewidth=2, label=config)
        ax.annotate(config, (precision_10[idx], overall_novelty[idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[idx], alpha=0.3))
    
    ax.set_xlabel('Precision@10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Novelty Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Novelty Trade-off Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # Add quadrant lines
    ax.axhline(y=np.mean(overall_novelty), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.mean(precision_10), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figure3_precision_novelty_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_precision_novelty_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 4: Distribution Box Plots ==========
def figure4_distribution_boxes():
    # Simulated distributions based on your data
    # In practice, you'd use your actual per-run data
    np.random.seed(42)
    
    data_dict = {}
    for config in configs:
        mean = precision_data[config][2]  # P@10
        # Generate sample data with realistic variance
        data_dict[config] = np.random.normal(mean, 0.08, 10)
        data_dict[config] = np.clip(data_dict[config], 0, 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = np.arange(len(configs))
    bp = ax.boxplot([data_dict[c] for c in configs], 
                     positions=positions,
                     labels=configs,
                     patch_artist=True,
                     notch=True,
                     showmeans=True)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Precision@10 Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision@10 Distribution Across Configurations', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('figure4_distribution_boxes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_distribution_boxes.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 5: Ablation Impact Analysis ==========
def figure5_ablation_impact():
    baseline = precision_data['No_Synthesis'][2]  # P@10
    
    comparisons = {
        'Full System\n(+Synthesis)': precision_data['Full_System'][2] - baseline,
        'No RAG': precision_data['No_RAG'][2] - baseline,
        'No Critique': precision_data['No_Critique'][2] - baseline,
        'No Viewpoint': precision_data['No_Viewpoint'][2] - baseline
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(comparisons.keys())
    values = list(comparisons.values())
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Change in Precision@10 vs. No_Synthesis', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Impact Analysis\n(Negative = Performance Drop)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + 0.003 if value > 0 else value - 0.003, i, 
                f'{value:.3f}', va='center', 
                ha='left' if value > 0 else 'right',
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure5_ablation_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_ablation_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 6: Query-Level Heatmap ==========
def figure6_query_heatmap():
    # Sample query data (you'd use your actual per-query results)
    queries = [f'Q{i+1}' for i in range(15)]
    
    # Create sample data matrix
    np.random.seed(42)
    data_matrix = np.zeros((len(queries), len(configs)))
    for j, config in enumerate(configs):
        mean = precision_data[config][2]
        data_matrix[:, j] = np.random.normal(mean, 0.1, len(queries))
        data_matrix[:, j] = np.clip(data_matrix[:, j], 0, 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(queries)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yticklabels(queries)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query', fontsize=12, fontweight='bold')
    ax.set_title('Query-Level Precision@10 Performance Heatmap', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Precision@10', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(queries)):
        for j in range(len(configs)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    plt.savefig('figure6_query_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure6_query_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== FIGURE 9: VirSci Comparison ==========
def figure9_virsci_comparison():
    metrics = ['P@3', 'P@5', 'P@10', 'P@20']
    no_synthesis = [0.630, 0.607, 0.570, 0.509]
    virsci = [0.556, 0.717, 0.450, 0.575]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, no_synthesis, width, label='No_Synthesis (Ours)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, virsci, width, label='VirSci (Baseline)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Precision Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision Score', fontsize=12, fontweight='bold')
    ax.set_title('Direct Comparison: No_Synthesis vs. VirSci Baseline\n(N=8 matched queries)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, shadow=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)
    
    # Add statistical significance note
    ax.text(0.5, 0.02, 'Difference not statistically significant (p=0.49, Wilcoxon p=0.91)',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('figure9_virsci_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure9_virsci_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all figures
if __name__ == '__main__':
    print("Generating Figure 1: Precision@N Bar Chart...")
    figure1_precision_bars()
    
    print("Generating Figure 2: Novelty Radar Chart...")
    figure2_novelty_radar()
    
    print("Generating Figure 3: Precision-Novelty Scatter...")
    figure3_precision_novelty_scatter()
    
    print("Generating Figure 4: Distribution Box Plots...")
    figure4_distribution_boxes()
    
    print("Generating Figure 5: Ablation Impact...")
    figure5_ablation_impact()
    
    print("Generating Figure 6: Query-Level Heatmap...")
    figure6_query_heatmap()
    
    print("Generating Figure 9: VirSci Comparison...")
    figure9_virsci_comparison()
    
    print("\nAll figures generated successfully!")