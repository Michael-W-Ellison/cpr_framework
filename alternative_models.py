"""
ALTERNATIVE MODELS FOR CPR PREDICTION FAILURES

Based on root cause analysis, this script develops and validates
alternative model formulations for the 30% failing architectures.
"""

import re
import math
from collections import defaultdict

# Load data
data_list = []
with open('/home/ubuntu/Desktop/CPR/312 experiment test.txt', 'r') as f:
    content = f.read()
    match = re.search(r'embedded_data_full_list = \[(.*?)\]', content, re.DOTALL)
    if match:
        data_text = '[' + match.group(1) + ']'
        data_list = eval(data_text)

# Organize data
data_by_arch = defaultdict(list)
for row in data_list:
    constraint = row['Constraint']
    if constraint == 'adjacent_duplicates':
        constraint = 'pattern_prohibition'
    elif constraint == 'sum_limits':
        constraint = 'sum_modulation'

    arch_key = f"{constraint}_{row['Mixing']}_{row['Governor']}"
    data_by_arch[arch_key].append({
        'CPR': row['CPR'],
        'log_CPR': math.log10(row['CPR']) if row['CPR'] > 0 else -100,
        'Exploration': row['Exploration'],
        'Complexity': row['Complexity'],
        'Config': row['Config']
    })

print("="*90)
print("ALTERNATIVE MODEL DEVELOPMENT FOR FAILING ARCHITECTURES")
print("="*90)

# ============================================================================
# MODEL 1: PIECEWISE LINEAR MODEL
# ============================================================================

print("\n" + "="*90)
print("MODEL 1: PIECEWISE LINEAR IN LOG-LOG SPACE")
print("="*90)
print()
print("Hypothesis: Pattern prohibition creates piecewise linear behavior")
print("Model: log(Exploration) = m * log(CPR) + b, with different (m,b) per regime")
print()

def fit_piecewise_linear(data_points):
    """Fit piecewise linear model: log(E) = m*log(CPR) + b"""
    # Define regimes
    constrained = [d for d in data_points if d['log_CPR'] > -7.8]
    critical = [d for d in data_points if -8.8 <= d['log_CPR'] <= -7.8]
    emergent = [d for d in data_points if d['log_CPR'] < -8.8]

    def fit_linear_regime(regime_data):
        if len(regime_data) < 2:
            return None, None, 0

        # Filter out zero exploration
        valid = [d for d in regime_data if d['Exploration'] > 0]
        if len(valid) < 2:
            return None, None, 0

        # Simple linear regression in log-log space
        x = [d['log_CPR'] for d in valid]
        y = [math.log10(d['Exploration']) if d['Exploration'] > 0 else -10 for d in valid]

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi**2 for xi in x)

        # Slope and intercept
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-10:
            return None, None, 0

        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n

        # Calculate R^2
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean)**2 for yi in y)
        ss_res = sum((y[i] - (m * x[i] + b))**2 for i in range(n))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return m, b, r2

    results = {}
    for regime_name, regime_data in [('constrained', constrained), ('critical', critical), ('emergent', emergent)]:
        m, b, r2 = fit_linear_regime(regime_data)
        results[regime_name] = {'m': m, 'b': b, 'r2': r2, 'n': len(regime_data)}

    return results

# Test on worst failure
arch_key = 'pattern_prohibition_multiplicative_entropy_maximization'
if arch_key in data_by_arch:
    print(f"Testing on: {arch_key}")
    results = fit_piecewise_linear(data_by_arch[arch_key])

    for regime, params in results.items():
        print(f"\n  {regime.upper()} Regime ({params['n']} points):")
        if params['m'] is not None:
            print(f"    log(E) = {params['m']:.4f} * log(CPR) + {params['b']:.4f}")
            print(f"    R^2 = {params['r2']:.4f}")
            print(f"    Prediction: E = 10^({params['b']:.4f}) * CPR^({params['m']:.4f})")
            print(f"                E = {10**params['b']:.4e} * CPR^{params['m']:.4f}")
        else:
            print(f"    Insufficient data for fitting")

# ============================================================================
# MODEL 2: COMPLEXITY-BASED MODEL
# ============================================================================

print("\n" + "="*90)
print("MODEL 2: DIRECT COMPLEXITY-TO-EXPLORATION MAPPING")
print("="*90)
print()
print("Hypothesis: For pattern_prohibition, Exploration != Sigmoid(CPR)")
print("Instead: Exploration = f(Complexity), where Complexity itself depends on CPR")
print()

def analyze_complexity_exploration_relationship(data_points):
    """Analyze Complexity -> Exploration relationship"""
    # Filter valid points
    valid = [d for d in data_points if d['Complexity'] > 0 and d['Exploration'] > 0]

    if len(valid) < 2:
        return None

    # Check if Exploration ≈ Complexity / 2.4467
    max_complexity = 2.4467
    errors = []
    for d in valid:
        predicted_exp = d['Complexity'] / max_complexity
        actual_exp = d['Exploration']
        error = abs(predicted_exp - actual_exp)
        errors.append(error)

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)

    # Check if there's a power law: E = (C / C_max)^alpha
    log_comp_norm = [math.log10(d['Complexity'] / max_complexity) for d in valid]
    log_exp = [math.log10(d['Exploration']) for d in valid]

    # Linear regression
    n = len(log_comp_norm)
    sum_x = sum(log_comp_norm)
    sum_y = sum(log_exp)
    sum_xy = sum(log_comp_norm[i] * log_exp[i] for i in range(n))
    sum_x2 = sum(x**2 for x in log_comp_norm)

    denom = n * sum_x2 - sum_x**2
    if abs(denom) > 1e-10:
        alpha = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - alpha * sum_x) / n

        # Calculate R^2
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean)**2 for yi in log_exp)
        ss_res = sum((log_exp[i] - (alpha * log_comp_norm[i] + intercept))**2 for i in range(n))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        alpha = None
        intercept = None
        r2 = 0

    return {
        'linear_mean_error': mean_error,
        'linear_max_error': max_error,
        'power_alpha': alpha,
        'power_intercept': intercept,
        'power_r2': r2,
        'n_points': len(valid)
    }

# Test on failing architecture
if arch_key in data_by_arch:
    print(f"Testing on: {arch_key}")
    result = analyze_complexity_exploration_relationship(data_by_arch[arch_key])

    if result:
        print(f"\n  Simple Linear Model: E = C / {2.4467:.4f}")
        print(f"    Mean absolute error: {result['linear_mean_error']:.4f}")
        print(f"    Max absolute error: {result['linear_max_error']:.4f}")

        if result['power_alpha']:
            print(f"\n  Power Law Model: E = (C / C_max)^{result['power_alpha']:.4f}")
            print(f"    Intercept (log scale): {result['power_intercept']:.4f}")
            print(f"    R^2 = {result['power_r2']:.4f}")
            print(f"    Interpretation: E = 10^{result['power_intercept']:.4f} * (C / C_max)^{result['power_alpha']:.4f}")

# ============================================================================
# MODEL 3: DUAL-PATHWAY MODEL
# ============================================================================

print("\n" + "="*90)
print("MODEL 3: DUAL-PATHWAY MODEL")
print("="*90)
print()
print("Hypothesis: Pattern prohibition creates two competing dynamics:")
print("  Pathway 1: CPR reduces valid states (constraining effect)")
print("  Pathway 2: Pattern structure enables strategic exploration (enabling effect)")
print()
print("Model: E = w1 * Sigmoid(log(CPR)) + w2 * f(Complexity)")
print()

def fit_dual_pathway_model(data_points):
    """Fit dual pathway model"""
    # Standard sigmoid parameters
    L_sigmoid = 0.8513
    k_sigmoid = 46.7978
    x0_sigmoid = -8.2999

    def sigmoid(log_cpr):
        try:
            exp_arg = -k_sigmoid * (log_cpr - x0_sigmoid)
            if exp_arg > 100:  # Avoid overflow
                return 0.0
            elif exp_arg < -100:
                return L_sigmoid
            return L_sigmoid / (1 + math.exp(exp_arg))
        except:
            return 0.0

    # Prepare data
    valid = [d for d in data_points if d['Complexity'] > 0]
    if len(valid) < 2:
        return None

    # Try different weight combinations
    best_error = float('inf')
    best_w1 = None
    best_w2 = None

    for w1_int in range(0, 11):  # 0.0 to 1.0 in steps of 0.1
        w1 = w1_int / 10.0
        w2 = 1.0 - w1

        total_error = 0
        for d in valid:
            sig_component = sigmoid(d['log_CPR'])
            comp_component = d['Complexity'] / 2.4467
            predicted = w1 * sig_component + w2 * comp_component
            error = (predicted - d['Exploration'])**2
            total_error += error

        avg_error = math.sqrt(total_error / len(valid))

        if avg_error < best_error:
            best_error = avg_error
            best_w1 = w1
            best_w2 = w2

    return {
        'w1': best_w1,
        'w2': best_w2,
        'rmse': best_error,
        'n_points': len(valid)
    }

# Test on failing architecture
if arch_key in data_by_arch:
    print(f"Testing on: {arch_key}")
    result = fit_dual_pathway_model(data_by_arch[arch_key])

    if result:
        print(f"\n  Optimal Weights:")
        print(f"    w1 (Sigmoid pathway) = {result['w1']:.2f}")
        print(f"    w2 (Complexity pathway) = {result['w2']:.2f}")
        print(f"    RMSE = {result['rmse']:.4f}")
        print(f"\n  Model Equation:")
        print(f"    E = {result['w1']:.2f} * Sigmoid(log(CPR)) + {result['w2']:.2f} * (C / C_max)")

        if result['w2'] > 0.7:
            print(f"\n  >>> FINDING: Complexity pathway dominates ({result['w2']*100:.0f}%)")
            print(f"      This architecture is better predicted by Complexity than CPR!")

# ============================================================================
# MODEL 4: MODIFIED SIGMOID WITH COMPLEXITY MODULATION
# ============================================================================

print("\n" + "="*90)
print("MODEL 4: COMPLEXITY-MODULATED SIGMOID")
print("="*90)
print()
print("Hypothesis: CPR determines potential, Complexity determines realization")
print("Model: E = Sigmoid(log(CPR)) * (C / C_max)^beta")
print("where beta captures how strongly complexity gates the emergence")
print()

def fit_complexity_modulated_sigmoid(data_points):
    """Fit complexity-modulated sigmoid"""
    L_sigmoid = 0.8513
    k_sigmoid = 46.7978
    x0_sigmoid = -8.2999
    C_max = 2.4467

    def sigmoid(log_cpr):
        try:
            exp_arg = -k_sigmoid * (log_cpr - x0_sigmoid)
            if exp_arg > 100:  # Avoid overflow
                return 0.0
            elif exp_arg < -100:
                return L_sigmoid
            return L_sigmoid / (1 + math.exp(exp_arg))
        except:
            return 0.0

    valid = [d for d in data_points if d['Complexity'] > 0 and d['Exploration'] > 0]
    if len(valid) < 2:
        return None

    # Try different beta values
    best_error = float('inf')
    best_beta = None

    for beta_int in range(-20, 21):  # -2.0 to 2.0 in steps of 0.1
        beta = beta_int / 10.0

        total_error = 0
        for d in valid:
            sig_component = sigmoid(d['log_CPR'])
            comp_factor = (d['Complexity'] / C_max) ** beta if d['Complexity'] > 0 else 0
            predicted = sig_component * comp_factor
            error = (predicted - d['Exploration'])**2
            total_error += error

        avg_error = math.sqrt(total_error / len(valid))

        if avg_error < best_error:
            best_error = avg_error
            best_beta = beta

    return {
        'beta': best_beta,
        'rmse': best_error,
        'n_points': len(valid)
    }

# Test on failing architecture
if arch_key in data_by_arch:
    print(f"Testing on: {arch_key}")
    result = fit_complexity_modulated_sigmoid(data_by_arch[arch_key])

    if result:
        print(f"\n  Optimal Beta: {result['beta']:.2f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"\n  Model Equation:")
        print(f"    E = Sigmoid(log(CPR)) * (C / C_max)^{result['beta']:.2f}")

        if result['beta'] < 0.5:
            print(f"\n  >>> Low beta suggests complexity is a weak enabler")
        elif result['beta'] > 1.5:
            print(f"\n  >>> High beta suggests complexity is a strong enabler")

# ============================================================================
# MODEL COMPARISON ACROSS ALL FAILING ARCHITECTURES
# ============================================================================

print("\n" + "="*90)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*90)

failing_architectures = [
    'pattern_prohibition_multiplicative_entropy_maximization',
    'pattern_prohibition_multiplicative_novelty_seeking',
    'pattern_prohibition_multiplicative_uniform_distribution',
    'pattern_prohibition_additive_entropy_maximization',
    'pattern_prohibition_triple_sum_entropy_maximization',
    'pattern_prohibition_additive_uniform_distribution',
]

print(f"\n{'Architecture':<55} {'Model':<20} {'RMSE':<10} {'Param':<15}")
print("-"*100)

for arch_key in failing_architectures:
    if arch_key in data_by_arch:
        data = data_by_arch[arch_key]

        # Dual pathway
        dp_result = fit_dual_pathway_model(data)
        if dp_result:
            print(f"{arch_key:<55} {'Dual Pathway':<20} {dp_result['rmse']:<10.4f} w2={dp_result['w2']:.2f}")

        # Complexity-modulated
        cm_result = fit_complexity_modulated_sigmoid(data)
        if cm_result:
            print(f"{'':55} {'Complexity-Modulated':<20} {cm_result['rmse']:<10.4f} beta={cm_result['beta']:.2f}")

print("\n" + "="*90)
print("END OF ALTERNATIVE MODEL ANALYSIS")
print("="*90)
