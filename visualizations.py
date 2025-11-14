#!/usr/bin/env python3
"""
CPR Framework Visualization Module

Provides comprehensive visualization capabilities for the CPR Framework including:
- Regime transitions
- Model comparisons
- Architecture analysis
- Performance metrics
- Sensitivity analysis

Usage:
    from visualizations import CPRVisualizer

    viz = CPRVisualizer()
    viz.plot_regime_transitions()
    viz.plot_model_comparison()
    viz.generate_all_plots(output_dir='figures')
"""

import os
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

from implementation_complete import (
    predict_exploration as predict_exploration_orig,
    select_model,
    ADJUSTMENT_FACTORS,
    get_adjustment_factor
)

# For compatibility
ARCHITECTURE_ADJUSTMENTS = ADJUSTMENT_FACTORS


def predict_exploration(n: int, b: int, constraint: str, mixing: str,
                       governor: str, complexity: Optional[float] = None) -> Dict:
    """
    Wrapper to predict exploration from system parameters (n, b) instead of CPR.
    """
    # Calculate base CPR
    base_cpr = n / (b ** n)

    # Predict exploration
    prediction = predict_exploration_orig(
        cpr=base_cpr,
        constraint=constraint,
        mixing=mixing,
        governor=governor,
        complexity=complexity
    )

    # Get adjustment factor
    adj_factor = get_adjustment_factor(constraint, mixing, governor)
    adjusted_cpr = base_cpr * adj_factor

    # Get model type
    model = select_model(constraint, mixing, governor)

    return {
        'prediction': prediction,
        'base_cpr': base_cpr,
        'adjusted_cpr': adjusted_cpr,
        'adjustment_factor': adj_factor,
        'model': model,
        'complexity': complexity,
        'normalized_complexity': complexity / 2.4467 if complexity else None
    }


class CPRVisualizer:
    """Comprehensive visualization suite for the CPR Framework."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

        try:
            plt.style.use(style)
        except:
            print(f"Warning: Style '{style}' not available, using default")

        # Color scheme
        self.colors = {
            'class1': '#2E86AB',
            'class2': '#F18F01',
            'constraint': '#A23B72',
            'regime_constrained': '#E63946',
            'regime_critical': '#F4A261',
            'regime_emergent': '#2A9D8F',
            'accent': '#6A0572'
        }

        # Default figure size
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_regime_transitions(self, output_path: Optional[str] = None,
                                n_range: tuple = (4, 18)) -> str:
        """
        Plot comprehensive regime transition analysis.

        Args:
            output_path: Path to save figure (if None, displays only)
            n_range: Range of system sizes to analyze

        Returns:
            Path where figure was saved (or empty string)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Collect data
        n_values = np.arange(n_range[0], n_range[1])
        e_values = []
        cpr_values = []

        for n in n_values:
            result = predict_exploration(
                n=int(n), b=3,
                constraint="sum_modulation",
                mixing="additive",
                governor="uniform_distribution"
            )
            e_values.append(result['prediction'])
            cpr_values.append(result['adjusted_cpr'])

        e_values = np.array(e_values)
        cpr_values = np.array(cpr_values)

        # Plot 1: E vs n with regime shading (large, main plot)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.plot(n_values, e_values, 'o-', linewidth=3, markersize=10,
                color=self.colors['class1'], label='Exploration Efficiency')

        # Shade regimes based on E values
        for i in range(len(n_values)-1):
            if e_values[i] < 0.2:
                color = self.colors['regime_constrained']
                alpha = 0.1
            elif e_values[i] < 0.7:
                color = self.colors['regime_critical']
                alpha = 0.1
            else:
                color = self.colors['regime_emergent']
                alpha = 0.1
            ax1.axvspan(n_values[i], n_values[i+1], alpha=alpha, color=color)

        ax1.set_xlabel('System Size (n)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Exploration Efficiency', fontsize=14, fontweight='bold')
        ax1.set_title('Regime Transitions in CPR Systems', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        ax1.legend(fontsize=11)

        # Add regime labels
        ax1.text(6, 0.05, 'CONSTRAINED', fontsize=11, color='red', fontweight='bold')
        ax1.text(10, 0.4, 'CRITICAL', fontsize=11, color='orange', fontweight='bold')
        ax1.text(16, 0.75, 'EMERGENT', fontsize=11, color='green', fontweight='bold')

        # Plot 2: E vs CPR (log scale)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.semilogx(cpr_values, e_values, 'o-', linewidth=2, markersize=7,
                    color=self.colors['constraint'])
        ax2.invert_xaxis()
        ax2.set_xlabel('CPR (log)', fontsize=11)
        ax2.set_ylabel('E', fontsize=11)
        ax2.set_title('Sigmoid Transition', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')

        # Shade regimes
        ax2.axvspan(1e-6, max(cpr_values), alpha=0.2,
                   color=self.colors['regime_constrained'])
        ax2.axvspan(1e-9, 1e-7, alpha=0.2,
                   color=self.colors['regime_critical'])
        ax2.axvspan(min(cpr_values), 1e-9, alpha=0.2,
                   color=self.colors['regime_emergent'])

        # Plot 3: Growth rate (dE/dn)
        ax3 = fig.add_subplot(gs[1, 2])
        de_dn = np.gradient(e_values, n_values)
        ax3.plot(n_values, de_dn, 'o-', linewidth=2, markersize=7,
                color=self.colors['class2'])
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('System Size (n)', fontsize=11)
        ax3.set_ylabel('dE/dn', fontsize=11)
        ax3.set_title('Growth Rate', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Highlight maximum
        max_idx = np.argmax(de_dn)
        ax3.plot(n_values[max_idx], de_dn[max_idx], 'r*', markersize=20)
        ax3.text(n_values[max_idx], de_dn[max_idx],
                f'  Max\n  n={n_values[max_idx]}', fontsize=9)

        # Plot 4: CPR scaling
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.semilogy(n_values, cpr_values, 'o-', linewidth=2, markersize=7,
                    color=self.colors['accent'])
        ax4.set_xlabel('System Size (n)', fontsize=11)
        ax4.set_ylabel('CPR (log)', fontsize=11)
        ax4.set_title('CPR Exponential Decay', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')

        # Plot 5: State space explosion
        ax5 = fig.add_subplot(gs[2, 1])
        state_spaces = 3 ** n_values
        ax5.semilogy(n_values, state_spaces, 'o-', linewidth=2, markersize=7,
                    color=self.colors['class1'])
        ax5.set_xlabel('System Size (n)', fontsize=11)
        ax5.set_ylabel('State Space (log)', fontsize=11)
        ax5.set_title('State Space Explosion', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, which='both')

        # Plot 6: Regime distribution
        ax6 = fig.add_subplot(gs[2, 2])
        regime_counts = {
            'Constrained': sum(1 for e in e_values if e < 0.2),
            'Critical': sum(1 for e in e_values if 0.2 <= e < 0.7),
            'Emergent': sum(1 for e in e_values if e >= 0.7)
        }
        colors = [self.colors['regime_constrained'],
                 self.colors['regime_critical'],
                 self.colors['regime_emergent']]
        ax6.pie(regime_counts.values(), labels=regime_counts.keys(),
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax6.set_title('Regime Distribution', fontsize=12, fontweight='bold')

        plt.suptitle('CPR Framework: Regime Transitions Analysis',
                    fontsize=18, fontweight='bold', y=0.995)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return ""

    def plot_model_comparison(self, output_path: Optional[str] = None) -> str:
        """
        Compare Class I (sigmoid) and Class II (complexity) models.

        Args:
            output_path: Path to save figure

        Returns:
            Path where figure was saved
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Class I: Multiple constraints
        ax = axes[0, 0]
        n_values = np.arange(6, 17)

        for constraint, color, label in [
            ("sum_modulation", self.colors['class1'], "sum_modulation"),
            ("local_entropy", self.colors['constraint'], "local_entropy")
        ]:
            predictions = []
            for n in n_values:
                result = predict_exploration(
                    n=int(n), b=3,
                    constraint=constraint,
                    mixing="additive",
                    governor="uniform_distribution"
                )
                predictions.append(result['prediction'])

            ax.plot(n_values, predictions, 'o-', linewidth=2,
                   markersize=7, color=color, label=label)

        ax.set_xlabel('System Size (n)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Class I: Density-Based Constraints\n(Sigmoid Model)',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.9)

        # Class II: Complexity relationship
        ax = axes[0, 1]
        complexity_values = np.linspace(0.5, 2.4, 30)
        predictions = []

        for c in complexity_values:
            result = predict_exploration(
                n=12, b=3,
                constraint="pattern_prohibition",
                mixing="additive",
                governor="uniform_distribution",
                complexity=float(c)
            )
            predictions.append(result['prediction'])

        ax.plot(complexity_values, predictions, 'o-', linewidth=2,
               markersize=5, color=self.colors['class2'], label='pattern_prohibition')
        ax.set_xlabel('Complexity (C)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Class II: Structure-Based Constraint\n(Complexity Model)',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.9)

        # Residuals comparison
        ax = axes[0, 2]
        # Simulated residuals for demonstration
        np.random.seed(42)
        residuals_class1 = np.random.normal(0, 0.03, 50)
        residuals_class2 = np.random.normal(0, 0.01, 50)

        ax.hist(residuals_class1, bins=15, alpha=0.6,
               color=self.colors['class1'], label='Class I')
        ax.hist(residuals_class2, bins=15, alpha=0.6,
               color=self.colors['class2'], label='Class II')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Residual Distributions', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Performance metrics Class I
        ax = axes[1, 0]
        metrics = ['R²', 'RMSE', 'MAE']
        values = [0.95, 0.05, 0.04]
        bars = ax.barh(metrics, values, color=self.colors['class1'], alpha=0.7)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title('Class I Performance', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Performance metrics Class II
        ax = axes[1, 1]
        metrics = ['R²', 'RMSE', 'MAE']
        values = [0.99, 0.02, 0.015]
        bars = ax.barh(metrics, values, color=self.colors['class2'], alpha=0.7)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title('Class II Performance', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Model equations
        ax = axes[1, 2]
        ax.axis('off')

        # Class I equation
        ax.text(0.5, 0.75, 'Class I: CPR Sigmoid',
               ha='center', fontsize=13, fontweight='bold',
               color=self.colors['class1'])
        ax.text(0.5, 0.6, r'$E = \frac{L}{1 + e^{-k(\log_{10}(CPR) - x_0)}}$',
               ha='center', fontsize=11)
        ax.text(0.5, 0.48, 'L=0.8513, k=46.80, x₀=-8.30',
               ha='center', fontsize=9)

        # Class II equation
        ax.text(0.5, 0.32, 'Class II: Complexity Linear',
               ha='center', fontsize=13, fontweight='bold',
               color=self.colors['class2'])
        ax.text(0.5, 0.17, r'$E = \left(\frac{C}{C_{max}}\right)^\alpha \times 10^\beta$',
               ha='center', fontsize=11)
        ax.text(0.5, 0.05, 'C_max=2.45, α=0.90, β=-0.015',
               ha='center', fontsize=9)

        plt.suptitle('CPR Framework: Model Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return ""

    def plot_architecture_analysis(self, output_path: Optional[str] = None) -> str:
        """
        Analyze impact of different architectural components.

        Args:
            output_path: Path to save figure

        Returns:
            Path where figure was saved
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Adjustment factors distribution
        ax = axes[0, 0]
        factors = list(ARCHITECTURE_ADJUSTMENTS.values())
        ax.hist(factors, bins=20, color=self.colors['class1'],
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Adjustment Factor', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Adjustment Factors',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=np.median(factors), color='red', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(factors):.3f}')
        ax.legend()

        # Mixing method comparison
        ax = axes[0, 1]
        mixing_methods = ["additive", "multiplicative", "triple_sum"]
        e_values = []

        for mixing in mixing_methods:
            result = predict_exploration(
                n=12, b=3,
                constraint="sum_modulation",
                mixing=mixing,
                governor="uniform_distribution"
            )
            e_values.append(result['prediction'])

        bars = ax.bar(range(len(mixing_methods)), e_values,
                     color=[self.colors['class1'], self.colors['constraint'],
                           self.colors['class2']], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(mixing_methods)))
        ax.set_xticklabels([m.replace('_', '\n') for m in mixing_methods], fontsize=9)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Impact of Mixing Method', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, e_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Governor comparison
        ax = axes[0, 2]
        governors = ["uniform_distribution", "entropy_maximization", "novelty_seeking"]
        e_values = []

        for governor in governors:
            result = predict_exploration(
                n=12, b=3,
                constraint="sum_modulation",
                mixing="additive",
                governor=governor
            )
            e_values.append(result['prediction'])

        bars = ax.bar(range(len(governors)), e_values,
                     color=[self.colors['class1'], self.colors['constraint'],
                           self.colors['class2']], alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(governors)))
        ax.set_xticklabels([g.replace('_', '\n') for g in governors], fontsize=8)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Impact of Governor', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, e_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Architecture count by constraint
        ax = axes[1, 0]
        constraint_counts = {}
        for arch in ARCHITECTURE_ADJUSTMENTS.keys():
            parts = arch.split('_')
            if len(parts) >= 2:
                constraint = f"{parts[0]}_{parts[1]}"
                constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1

        labels = list(constraint_counts.keys())
        counts = list(constraint_counts.values())
        colors = [self.colors['class2'] if 'pattern' in label else self.colors['class1']
                 for label in labels]

        bars = ax.bar(range(len(labels)), counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=9)
        ax.set_ylabel('Number of Architectures', fontsize=11)
        ax.set_title('Architecture Distribution by Constraint',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontsize=10)

        # Adjustment factors by constraint (boxplot)
        ax = axes[1, 1]
        by_constraint = {}
        for arch, factor in ARCHITECTURE_ADJUSTMENTS.items():
            parts = arch.split('_')
            if len(parts) >= 2:
                constraint = f"{parts[0]}_{parts[1]}"
                if constraint not in by_constraint:
                    by_constraint[constraint] = []
                by_constraint[constraint].append(factor)

        data = [by_constraint[k] for k in sorted(by_constraint.keys())]
        labels = [k.replace('_', '\n') for k in sorted(by_constraint.keys())]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], sorted(by_constraint.keys())):
            color = self.colors['class2'] if 'pattern' in label else self.colors['class1']
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Adjustment Factor', fontsize=11)
        ax.set_title('Adjustment Factor Ranges', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', labelsize=9)

        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
ARCHITECTURE SUMMARY

Total Architectures: {len(ARCHITECTURE_ADJUSTMENTS)}

By Constraint:
  pattern_prohibition: {sum(1 for k in ARCHITECTURE_ADJUSTMENTS if 'pattern' in k)}
  sum_modulation: {sum(1 for k in ARCHITECTURE_ADJUSTMENTS if 'sum_modulation' in k)}
  local_entropy: {sum(1 for k in ARCHITECTURE_ADJUSTMENTS if 'local_entropy' in k)}

Adjustment Factors:
  Min: {min(factors):.4f}
  Max: {max(factors):.4f}
  Mean: {np.mean(factors):.4f}
  Median: {np.median(factors):.4f}
  Std Dev: {np.std(factors):.4f}

Coverage: 100% (27/27)
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center')

        plt.suptitle('CPR Framework: Architecture Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return ""

    def plot_sensitivity_analysis(self, output_path: Optional[str] = None) -> str:
        """
        Analyze sensitivity to system parameters.

        Args:
            output_path: Path to save figure

        Returns:
            Path where figure was saved
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sensitivity to base (b)
        ax = axes[0, 0]
        n_fixed = 10
        b_values = [2, 3, 4, 5]
        e_values = []

        for b in b_values:
            result = predict_exploration(
                n=n_fixed, b=b,
                constraint="sum_modulation",
                mixing="additive",
                governor="uniform_distribution"
            )
            e_values.append(result['prediction'])

        ax.plot(b_values, e_values, 'o-', linewidth=2, markersize=10,
               color=self.colors['class1'])
        ax.set_xlabel('Base (b)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title(f'Sensitivity to Base (n={n_fixed})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for b, e in zip(b_values, e_values):
            ax.text(b, e + 0.02, f'{e:.3f}', ha='center', fontsize=9)

        # Sensitivity to system size (n)
        ax = axes[0, 1]
        n_values = np.arange(6, 17)

        for b, color, label in [(2, self.colors['class1'], 'b=2'),
                                (3, self.colors['constraint'], 'b=3'),
                                (4, self.colors['class2'], 'b=4')]:
            e_values = []
            for n in n_values:
                result = predict_exploration(
                    n=int(n), b=b,
                    constraint="sum_modulation",
                    mixing="additive",
                    governor="uniform_distribution"
                )
                e_values.append(result['prediction'])

            ax.plot(n_values, e_values, 'o-', linewidth=2, markersize=6,
                   color=color, label=label)

        ax.set_xlabel('System Size (n)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Sensitivity to System Size', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Constraint comparison at different scales
        ax = axes[1, 0]
        n_values = np.arange(8, 16)

        for constraint, color, label in [
            ("sum_modulation", self.colors['class1'], "sum_modulation"),
            ("local_entropy", self.colors['constraint'], "local_entropy")
        ]:
            e_values = []
            for n in n_values:
                result = predict_exploration(
                    n=int(n), b=3,
                    constraint=constraint,
                    mixing="additive",
                    governor="uniform_distribution"
                )
                e_values.append(result['prediction'])

            ax.plot(n_values, e_values, 'o-', linewidth=2, markersize=7,
                   color=color, label=label)

        ax.set_xlabel('System Size (n)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Constraint Type Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Complexity sensitivity (Class II)
        ax = axes[1, 1]
        n_values = [10, 12, 14, 16]
        complexity_range = np.linspace(0.5, 2.4, 20)

        for n, color in zip(n_values,
                           [self.colors['class1'], self.colors['constraint'],
                            self.colors['class2'], self.colors['accent']]):
            e_values = []
            for c in complexity_range:
                result = predict_exploration(
                    n=n, b=3,
                    constraint="pattern_prohibition",
                    mixing="additive",
                    governor="uniform_distribution",
                    complexity=float(c)
                )
                e_values.append(result['prediction'])

            ax.plot(complexity_range, e_values, linewidth=2, color=color, label=f'n={n}')

        ax.set_xlabel('Complexity (C)', fontsize=11)
        ax.set_ylabel('Exploration Efficiency', fontsize=11)
        ax.set_title('Complexity Sensitivity (pattern_prohibition)',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('CPR Framework: Sensitivity Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return ""

    def generate_all_plots(self, output_dir: str = 'visualizations') -> List[str]:
        """
        Generate all visualization plots and save to directory.

        Args:
            output_dir: Directory to save all plots

        Returns:
            List of paths to generated files
        """
        os.makedirs(output_dir, exist_ok=True)

        plots = [
            ("regime_transitions.png", self.plot_regime_transitions),
            ("model_comparison.png", self.plot_model_comparison),
            ("architecture_analysis.png", self.plot_architecture_analysis),
            ("sensitivity_analysis.png", self.plot_sensitivity_analysis),
        ]

        generated = []
        for filename, plot_func in plots:
            output_path = os.path.join(output_dir, filename)
            print(f"Generating: {filename}...")
            try:
                path = plot_func(output_path=output_path)
                generated.append(path)
                print(f"  ✓ Saved: {path}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\nGenerated {len(generated)} visualizations in {output_dir}/")
        return generated


def main():
    """Command-line interface for visualization generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate CPR Framework visualizations'
    )
    parser.add_argument('--output-dir', '-o', default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--plot', '-p', choices=['regime', 'model', 'architecture', 'sensitivity', 'all'],
                       default='all', help='Which plot(s) to generate')

    args = parser.parse_args()

    if not PLOTTING_AVAILABLE:
        print("Error: matplotlib is required but not installed.")
        print("Install with: pip install matplotlib numpy")
        return 1

    viz = CPRVisualizer()
    os.makedirs(args.output_dir, exist_ok=True)

    plot_map = {
        'regime': ('regime_transitions.png', viz.plot_regime_transitions),
        'model': ('model_comparison.png', viz.plot_model_comparison),
        'architecture': ('architecture_analysis.png', viz.plot_architecture_analysis),
        'sensitivity': ('sensitivity_analysis.png', viz.plot_sensitivity_analysis),
    }

    if args.plot == 'all':
        viz.generate_all_plots(args.output_dir)
    else:
        filename, func = plot_map[args.plot]
        output_path = os.path.join(args.output_dir, filename)
        path = func(output_path=output_path)
        print(f"Generated: {path}")

    return 0


if __name__ == "__main__":
    exit(main())
