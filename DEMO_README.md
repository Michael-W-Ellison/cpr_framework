# CPR Framework Demo Guide

This guide provides comprehensive information on exploring all aspects of the CPR (Constraint Pressure Ratio) Framework through interactive demos, tutorials, and visualizations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Demo Components](#demo-components)
3. [Interactive CLI Demo](#interactive-cli-demo)
4. [Jupyter Notebook Tutorial](#jupyter-notebook-tutorial)
5. [Visualization Suite](#visualization-suite)
6. [Example Use Cases](#example-use-cases)
7. [What You Can Explore](#what-you-can-explore)
8. [Installation](#installation)
9. [Troubleshooting](#troubleshooting)

## Quick Start

Get started with the CPR Framework demo in 3 simple steps:

```bash
# 1. Ensure you have Python 3.7+
python --version

# 2. Install optional dependencies (for visualizations)
pip install matplotlib numpy

# 3. Run the interactive demo
python demo.py
```

For Jupyter notebook:
```bash
# Install Jupyter
pip install jupyter matplotlib numpy

# Launch notebook
jupyter notebook CPR_Framework_Tutorial.ipynb
```

## Demo Components

The CPR Framework provides **four comprehensive demo components**:

| Component | File | Purpose | Best For |
|-----------|------|---------|----------|
| **Interactive CLI Demo** | `demo.py` | Menu-driven exploration | Quick exploration, guided scenarios |
| **Jupyter Tutorial** | `CPR_Framework_Tutorial.ipynb` | Complete walkthrough | Learning, experimentation, documentation |
| **Visualization Suite** | `visualizations.py` | Generate all plots | Publication figures, presentations |
| **Example Use Cases** | `examples.py` | Real-world applications | Practical implementation guidance |

## Interactive CLI Demo

### Overview

The `demo.py` script provides an interactive, menu-driven interface to explore all aspects of the CPR Framework.

### Features

- **6 Guided Scenarios**: Step-by-step demonstrations of key concepts
- **Quick Predictions**: Try pre-configured examples
- **Custom Predictions**: Enter your own parameters
- **Visualization Generation**: Create comprehensive plots
- **Framework Documentation**: Built-in reference materials

### Running the Demo

```bash
# Interactive menu (recommended for first-time users)
python demo.py

# Run specific scenario
python demo.py --scenario 1

# Interactive prediction mode
python demo.py --predict

# Generate all visualizations
python demo.py --visualize

# Show framework overview
python demo.py --overview

# Show mathematical models
python demo.py --models
```

### Available Scenarios

1. **Universality Classes Discovery**
   - Demonstrates the two fundamental constraint classes
   - Compares sigmoid vs. linear complexity behaviors
   - Shows automatic model selection

2. **Model Selection Logic**
   - Explains how the framework chooses the right model
   - Shows decision tree for Class I vs. Class II
   - Demonstrates fallback behavior

3. **Regime Transitions**
   - Explores constrained, critical, and emergent regimes
   - Shows how exploration changes across system sizes
   - Identifies critical transition points

4. **Architecture Impact**
   - Compares 27 different architectures
   - Shows effect of mixing methods and governors
   - Demonstrates adjustment factor system

5. **Validation Results**
   - Presents performance metrics (R², RMSE, MAE)
   - Shows 100% architecture coverage
   - Compares accuracy across constraint types

6. **Before/After Comparison**
   - Shows improvement from original to hybrid model
   - Demonstrates the discovery process
   - Quantifies performance gains (43× improvement)

### Example Session

```
CPR FRAMEWORK - INTERACTIVE DEMONSTRATION

What would you like to explore?

GUIDED SCENARIOS:
  1. Universality Classes Discovery
     → Demonstrates the two fundamental constraint classes
  2. Model Selection Logic
     → Shows how the framework automatically selects the right model
  ...

INTERACTIVE TOOLS:
  7. Quick Predictions - Try example configurations
  8. Custom Prediction - Enter your own parameters
  9. Generate Visualizations - Create all plots

Enter your choice (0-11): 1
```

## Jupyter Notebook Tutorial

### Overview

The `CPR_Framework_Tutorial.ipynb` provides a comprehensive, hands-on tutorial with executable code, visualizations, and detailed explanations.

### Contents

1. **Introduction** - Framework overview and key discoveries
2. **Setup and Installation** - Import framework and configure environment
3. **Core Concepts** - CPR metric, exploration efficiency, system components
4. **Two Universality Classes** - Detailed analysis of Class I and II
5. **Making Predictions** - Basic, batch, and custom predictions
6. **Regime Analysis** - Constrained, critical, and emergent regimes
7. **Architecture Comparison** - All 27 architectures analyzed
8. **Visualizations** - Publication-quality plots
9. **Advanced Topics** - Sensitivity analysis, critical points
10. **Real-World Applications** - Practical use cases

### Key Features

- **Executable Code**: All examples can be run and modified
- **Interactive Plots**: Matplotlib visualizations embedded
- **Comprehensive Coverage**: All framework features demonstrated
- **Educational Focus**: Clear explanations with mathematical foundations
- **Practical Examples**: Real-world applications and use cases

### Usage Tips

1. **Run cells sequentially** on first pass to build understanding
2. **Modify parameters** to experiment with different configurations
3. **Use as reference** for implementing your own analyses
4. **Export plots** for presentations and publications

### Example Notebook Cells

```python
# Make a prediction
result = predict_exploration(
    n=12, b=3,
    constraint="sum_modulation",
    mixing="additive",
    governor="uniform_distribution"
)

print(f"Predicted Exploration: {result['prediction']:.4f}")
print(f"Model Used: {result['model']}")
```

## Visualization Suite

### Overview

The `visualizations.py` module provides publication-quality visualizations for all aspects of the framework.

### Available Plots

1. **Regime Transitions** (`regime_transitions.png`)
   - E vs n with regime shading
   - E vs CPR sigmoid curve
   - Growth rate analysis
   - CPR exponential decay
   - State space explosion
   - Regime distribution pie chart

2. **Model Comparison** (`model_comparison.png`)
   - Class I constraints (sigmoid)
   - Class II constraint (complexity)
   - Residual distributions
   - Performance metrics
   - Model equations

3. **Architecture Analysis** (`architecture_analysis.png`)
   - Adjustment factor distribution
   - Mixing method comparison
   - Governor comparison
   - Architecture count by constraint
   - Adjustment factor ranges
   - Summary statistics

4. **Sensitivity Analysis** (`sensitivity_analysis.png`)
   - Sensitivity to base (b)
   - Sensitivity to system size (n)
   - Constraint type comparison
   - Complexity sensitivity

### Usage

```bash
# Generate all visualizations
python visualizations.py --output-dir figures/

# Generate specific plot
python visualizations.py --plot regime --output-dir figures/
python visualizations.py --plot model --output-dir figures/
python visualizations.py --plot architecture --output-dir figures/
python visualizations.py --plot sensitivity --output-dir figures/
```

### Programmatic Usage

```python
from visualizations import CPRVisualizer

viz = CPRVisualizer()

# Generate specific plot
viz.plot_regime_transitions(output_path='regime.png')
viz.plot_model_comparison(output_path='models.png')

# Generate all plots
viz.generate_all_plots(output_dir='figures')
```

### Customization

All visualizations use a consistent color scheme and can be customized:

```python
viz = CPRVisualizer(style='seaborn-v0_8-darkgrid')
viz.colors['class1'] = '#FF6B6B'  # Change color scheme
```

## Example Use Cases

### Overview

The `examples.py` module demonstrates practical applications of the CPR Framework with real-world scenarios.

### Available Examples

1. **System Design for Target Exploration**
   - Find system size (n) that achieves specific exploration level
   - Design for constrained, balanced, or emergent systems
   - Minimize resources while meeting requirements

2. **Architecture Comparison and Ranking**
   - Evaluate all 27 architectures
   - Rank by exploration efficiency
   - Select optimal architecture for goals

3. **Rapid Prototyping Without Simulation**
   - Evaluate configurations instantly
   - Compare alternatives without expensive computation
   - Filter candidates for detailed simulation

4. **Optimization: Finding Optimal Configuration**
   - Search entire parameter space
   - Multi-objective optimization
   - Minimize size while achieving target

5. **Regime Detection and Analysis**
   - Track regime transitions
   - Identify critical zones
   - Quantify growth rates

6. **Scaling Analysis**
   - Understand parameter sensitivity
   - Analyze scaling relationships
   - Predict effects of parameter changes

7. **Before/After Comparison**
   - Show improvement from original model
   - Demonstrate scientific discovery
   - Quantify performance gains

8. **Batch Prediction for Multiple Configurations**
   - Process multiple systems efficiently
   - Automatic model selection
   - Comprehensive regime classification

### Usage

```bash
# Run all examples interactively
python examples.py

# Run specific example
python examples.py --example 1

# List all examples
python examples.py --list
```

### Example Output

```
EXAMPLE 1: SYSTEM DESIGN FOR TARGET EXPLORATION

SCENARIO:
You need to design a system that achieves a specific exploration level.
The CPR Framework can predict the required system size (n) for your target.

Target E | Description              | Design n | Actual E | Error
========================================================================
      0.10 | Highly constrained       |       7  |   0.0984 | 0.0016
      0.30 | Moderately constrained   |       9  |   0.2956 | 0.0044
      0.50 | Balanced system          |      10  |   0.5012 | 0.0012
...
```

## What You Can Explore

### Core Concepts

- ✅ **CPR Metric**: Understand n / b^n and constraint pressure
- ✅ **Exploration Efficiency**: Learn how E ∈ [0, 1] measures state space coverage
- ✅ **System Components**: Constraints, mixing methods, governors

### Mathematical Models

- ✅ **Class I Sigmoid Model**: L / (1 + exp(-k(log(CPR) - x₀)))
- ✅ **Class II Complexity Model**: (C/C_max)^α × 10^β
- ✅ **Model Selection**: Automatic routing based on constraint type
- ✅ **Adjustment Factors**: Architecture-specific calibrations

### Universality Classes

- ✅ **Class I (Density-Based)**: sum_modulation, local_entropy
- ✅ **Class II (Structure-Based)**: pattern_prohibition
- ✅ **Behavioral Differences**: Sigmoid vs. linear scaling
- ✅ **Performance Comparison**: R² > 0.95 vs. R² > 0.99

### Regime Analysis

- ✅ **Constrained Regime**: High CPR, low exploration
- ✅ **Critical Regime**: Phase transition, rapid E growth
- ✅ **Emergent Regime**: Low CPR, high exploration
- ✅ **Transition Detection**: Identify regime boundaries

### Architecture System

- ✅ **27 Architectures**: 3 constraints × 3 mixings × 3 governors
- ✅ **Mixing Methods**: additive, multiplicative, triple_sum
- ✅ **Governors**: uniform_distribution, entropy_maximization, novelty_seeking
- ✅ **Impact Analysis**: How components affect exploration

### Performance & Validation

- ✅ **Accuracy Metrics**: R², RMSE, MAE for all models
- ✅ **Coverage**: 100% of 27 architectures
- ✅ **Improvement**: 67% → 100% coverage, 43× RMSE reduction
- ✅ **Validation**: 312 experiments across all architectures

### Practical Applications

- ✅ **System Design**: Target specific exploration levels
- ✅ **Optimization**: Find optimal configurations
- ✅ **Comparison**: Rank architectures
- ✅ **Prediction**: Avoid expensive simulations
- ✅ **Analysis**: Understand scaling and sensitivity

## Installation

### Minimum Requirements

```bash
# Python 3.7 or higher
python --version

# Core framework (no additional dependencies)
python -c "from implementation_complete import predict_exploration; print('✓ Core framework ready')"
```

### Full Demo Suite

```bash
# For visualizations and Jupyter notebook
pip install matplotlib numpy jupyter

# Verify installation
python -c "import matplotlib; import numpy; print('✓ All dependencies installed')"
```

### Optional Enhancement

```bash
# For enhanced plotting styles
pip install seaborn

# For interactive notebook features
pip install ipywidgets
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**:
```bash
pip install matplotlib numpy
```

---

**Issue**: `ImportError: cannot import name 'predict_exploration'`

**Solution**: Ensure you're in the correct directory:
```bash
ls implementation_complete.py  # Should exist
python demo.py
```

---

**Issue**: Jupyter notebook won't start

**Solution**:
```bash
pip install --upgrade jupyter
jupyter notebook --version
jupyter notebook CPR_Framework_Tutorial.ipynb
```

---

**Issue**: Plots not displaying in notebook

**Solution**: Add this cell at the top of the notebook:
```python
%matplotlib inline
```

---

**Issue**: Demo runs but no visualizations

**Solution**: This is expected if matplotlib isn't installed. Install it:
```bash
pip install matplotlib
```

---

### Getting Help

1. **Check Documentation**: Read `README.md` and `TECHNICAL_REPORT_SOLUTION.md`
2. **Review Examples**: Look at `examples.py` for usage patterns
3. **Check Implementation**: See `implementation_complete.py` for API details
4. **Validation Results**: Review `SCIENTIFIC_VALIDATION_REPORT.md`

## Additional Resources

### Framework Documentation

- `README.md` - Main framework documentation
- `TECHNICAL_REPORT_SOLUTION.md` - Detailed technical analysis
- `SCIENTIFIC_VALIDATION_REPORT.md` - Validation methodology and results
- `EXECUTIVE_SUMMARY.md` - High-level overview
- `DATA_PROVENANCE_REPORT.md` - Data sources and methodology

### Source Code

- `implementation_complete.py` - Production implementation
- `validation_results.py` - Validation test suite
- `failure_analysis.py` - Diagnostic analysis
- `alternative_models.py` - Model development history

### Demo Files

- `demo.py` - Interactive CLI demo
- `CPR_Framework_Tutorial.ipynb` - Jupyter tutorial
- `visualizations.py` - Visualization suite
- `examples.py` - Use case demonstrations
- `cpr_framework_demo.html` - Web-based demo

## Quick Reference

### Common Commands

```bash
# Interactive exploration
python demo.py

# Specific scenario
python demo.py --scenario 3

# Custom prediction
python demo.py --predict

# Generate visualizations
python visualizations.py

# Run examples
python examples.py --example 1

# Jupyter tutorial
jupyter notebook CPR_Framework_Tutorial.ipynb
```

### API Quick Start

```python
from implementation_complete import predict_exploration

# Make a prediction
result = predict_exploration(
    n=12,              # System size
    b=3,               # Base
    constraint="sum_modulation",
    mixing="additive",
    governor="uniform_distribution"
)

print(f"Exploration: {result['prediction']:.4f}")
print(f"Model: {result['model']}")
```

### Performance Summary

| Metric | Class I | Class II |
|--------|---------|----------|
| R² | > 0.95 | > 0.99 |
| RMSE | < 0.05 | < 0.02 |
| MAE | < 0.04 | < 0.015 |
| Coverage | 100% | 100% |

## Conclusion

The CPR Framework demo suite provides comprehensive tools for exploring all aspects of the framework, from basic concepts to advanced applications. Whether you're a researcher, developer, or student, these demos will help you understand and apply the framework effectively.

**Start exploring today**: `python demo.py`

For questions or issues, please refer to the documentation files or review the implementation source code.
