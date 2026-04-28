#!/usr/bin/env python3
"""
Create architecture diagram for XPipeline paper.
Addresses ITU reviewer comment: "Missing Architecture Diagram"
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def create_architecture_diagram():
    """Create Figure 0: XPipeline Architecture Overview"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ========================================================================
    # LEFT PANEL: Pipeline Architecture
    # ========================================================================
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('(a) Multi-Stage RAG Pipeline', fontweight='bold', fontsize=12)

    # Stage boxes
    stages = [
        {'name': 'Query', 'x': 1, 'y': 8, 'color': '#E8F4F8'},
        {'name': 'Retrieval\n($S_r$)', 'x': 1, 'y': 6, 'color': '#A8D8EA'},
        {'name': 'Synthesis\n($S_s$)', 'x': 1, 'y': 4, 'color': '#79C7E3'},
        {'name': 'Judge\n($S_j$)', 'x': 1, 'y': 2, 'color': '#4A9BAA'},
        {'name': 'Answer', 'x': 1, 'y': 0.3, 'color': '#E8F4F8'}
    ]

    for i, stage in enumerate(stages):
        # Draw box
        box = FancyBboxPatch((stage['x'], stage['y']), 2, 0.8,
                              boxstyle='round,pad=0.05',
                              facecolor=stage['color'],
                              edgecolor='#2C5F6F', linewidth=2)
        ax1.add_patch(box)

        # Add text
        ax1.text(stage['x'] + 1, stage['y'] + 0.4, stage['name'],
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Add arrows between stages
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((stage['x'] + 1, stage['y']),
                                   (stage['x'] + 1, stages[i+1]['y'] + 0.8),
                                   arrowstyle='->,head_width=0.3,head_length=0.3',
                                   color='#2C5F6F', linewidth=2)
            ax1.add_patch(arrow)

    # Corpus box (right side)
    corpus_box = FancyBboxPatch((5, 6), 3.5, 1.5,
                                boxstyle='round,pad=0.1',
                                facecolor='#FFE6CC',
                                edgecolor='#CC8800', linewidth=2)
    ax1.add_patch(corpus_box)
    ax1.text(6.75, 7, 'Document\nCorpus', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax1.text(6.75, 6.4, '(Network logs, standards,\n'
                        'support tickets, legal docs)',
            ha='center', va='center', fontsize=8, style='italic')

    # Arrow from corpus to retrieval
    corpus_arrow = FancyArrowPatch((5, 6.75), (3, 6.4),
                                  arrowstyle='->,head_width=0.3,head_length=0.3',
                                  color='#CC8800', linewidth=2, linestyle='dashed')
    ax1.add_patch(corpus_arrow)

    # Ablation annotation
    ax1.text(5.5, 3, 'Ablation:\nDisable stages\nto measure\ncontribution',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD',
                     edgecolor='#FF9800', linewidth=1.5))

    # ========================================================================
    # RIGHT PANEL: Shapley Value Computation
    # ========================================================================
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) Shapley Value Computation', fontweight='bold', fontsize=12)

    # Permutation examples
    y_start = 8.5
    permutations = [
        ['$S_r$', '$S_s$', '$S_j$'],
        ['$S_r$', '$S_j$', '$S_s$'],
        ['$S_s$', '$S_r$', '$S_j$'],
        ['$S_s$', '$S_j$', '$S_r$'],
        ['$S_j$', '$S_r$', '$S_s$'],
        ['$S_j$', '$S_s$', '$S_r$']
    ]

    ax2.text(5, 9.3, 'Random Stage Orderings (N=20 samples):',
            ha='center', fontsize=10, fontweight='bold')

    colors = ['#A8D8EA', '#79C7E3', '#4A9BAA']

    for i, perm in enumerate(permutations[:4]):  # Show first 4
        y = y_start - i * 0.8
        ax2.text(1, y, f'π{i+1}:', ha='right', fontsize=9)

        for j, stage in enumerate(perm):
            x = 2 + j * 1.5
            circle = plt.Circle((x, y), 0.3, color=colors[j], ec='black', linewidth=1)
            ax2.add_patch(circle)
            ax2.text(x, y, stage, ha='center', va='center', fontsize=9)

            if j < len(perm) - 1:
                ax2.plot([x + 0.3, x + 1.2], [y, y], 'k-', linewidth=1)
                ax2.plot([x + 1.15, x + 1.2], [y, y + 0.05], 'k-', linewidth=1)
                ax2.plot([x + 1.15, x + 1.2], [y, y - 0.05], 'k-', linewidth=1)

    ax2.text(5, y_start - 4 * 0.8 - 0.3, '⋮ (16 more)',
            ha='center', fontsize=10, style='italic')

    # Marginal contribution formula
    formula_y = 3.5
    formula_box = FancyBboxPatch((0.5, formula_y - 0.5), 9, 1.5,
                                 boxstyle='round,pad=0.1',
                                 facecolor='#F0F0F0',
                                 edgecolor='#333333', linewidth=2)
    ax2.add_patch(formula_box)

    ax2.text(5, formula_y + 0.5, 'Marginal Contribution:',
            ha='center', fontsize=10, fontweight='bold')
    ax2.text(5, formula_y, r'$\phi_i = \frac{1}{N} \sum_{k=1}^{N} [Q(S_{\leq i}^{(\pi_k)}) - Q(S_{< i}^{(\pi_k)})]$',
            ha='center', va='center', fontsize=11)

    # Final Shapley values output
    output_y = 1.5
    ax2.text(5, output_y, 'Shapley Values (Stage Importance):',
            ha='center', fontsize=10, fontweight='bold')

    shapley_results = [
        ('$\\phi_r$ = 0.412', colors[0]),
        ('$\\phi_s$ = 0.455', colors[1]),
        ('$\\phi_j$ = 0.105', colors[2])
    ]

    for i, (result, color) in enumerate(shapley_results):
        x = 2.5 + i * 2
        result_box = FancyBboxPatch((x - 0.6, output_y - 0.7), 1.2, 0.4,
                                   boxstyle='round,pad=0.05',
                                   facecolor=color,
                                   edgecolor='black', linewidth=1.5)
        ax2.add_patch(result_box)
        ax2.text(x, output_y - 0.5, result, ha='center', va='center',
                fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = 'paper/2026_ITU_conf/figures'
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/fig0_architecture.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}/fig0_architecture.png', bbox_inches='tight', dpi=300)

    print(f' Generated: {output_dir}/fig0_architecture.pdf')
    print(f' Generated: {output_dir}/fig0_architecture.png')

    return fig


if __name__ == '__main__':
    print('Creating architecture diagram...')
    create_architecture_diagram()
    print(' Architecture diagram complete!')
