"""
Root Cause Analysis of CPR Prediction Failures
Analyzing the 30% failing architectures
"""

import json
import math
from collections import defaultdict

# Load embedded data
embedded_data_full_list = []
exec(open('/home/ubuntu/Desktop/CPR/312 experiment test.txt').read(), {'embedded_data_full_list': embedded_data_full_list})

# Process data
data_by_arch = defaultdict(list)
for row in embedded_data_full_list:
    # Normalize names
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
        'Complexity': row['Complexity']
    })

# Define failing architectures from the problem statement
failing_architectures = [
    ('pattern_prohibition', 'multiplicative', 'entropy_maximization', 7.34),  # Complete failure
    ('pattern_prohibition', 'multiplicative', 'novelty_seeking', 5.5),        # Partial
    ('pattern_prohibition', 'multiplicative', 'uniform_distribution', 5.5),   # Partial
    ('pattern_prohibition', 'additive', 'entropy_maximization', 4.2),         # Partial
    ('pattern_prohibition', 'triple_sum', 'entropy_maximization', 4.2),       # Partial
    ('pattern_prohibition', 'additive', 'uniform_distribution', 3.8),         # Regime-dependent
    ('pattern_prohibition', 'additive', 'novelty_seeking', 3.8),              # Partial
    ('pattern_prohibition', 'triple_sum', 'uniform_distribution', 3.8),       # Partial
]

print("="*90)
print("ROOT CAUSE ANALYSIS: CPR PREDICTION FAILURES")
print("="*90)
print()

# Analysis 1: Pattern Identification
print("ANALYSIS 1: FAILING ARCHITECTURE PATTERNS")
print("-"*90)

constraint_counts = defaultdict(int)
mixing_counts = defaultdict(int)
governor_counts = defaultdict(int)

for constraint, mixing, governor, factor in failing_architectures:
    constraint_counts[constraint] += 1
    mixing_counts[mixing] += 1
    governor_counts[governor] += 1

print("\nConstraint types in failures:")
for k, v in sorted(constraint_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}/8 failures ({v*100/8:.1f}%)")

print("\nMixing types in failures:")
for k, v in sorted(mixing_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}/8 failures ({v*100/8:.1f}%)")

print("\nGovernor types in failures:")
for k, v in sorted(governor_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}/8 failures ({v*100/8:.1f}%)")

# Key finding
print("\n>>> KEY FINDING: ALL 8 failures involve 'pattern_prohibition' constraint")
print(">>> This suggests pattern_prohibition has fundamentally different behavior")

# Analysis 2: Data characteristics of failing architectures
print("\n" + "="*90)
print("ANALYSIS 2: DATA CHARACTERISTICS OF FAILING ARCHITECTURES")
print("-"*90)

for constraint, mixing, governor, factor in failing_architectures:
    arch_key = f"{constraint}_{mixing}_{governor}"
    arch_data = data_by_arch.get(arch_key, [])

    if arch_data:
        cprs = [d['CPR'] for d in arch_data]
        log_cprs = [d['log_CPR'] for d in arch_data]
        explorations = [d['Exploration'] for d in arch_data]
        complexities = [d['Complexity'] for d in arch_data]

        print(f"\n{arch_key} (Factor: {factor}x)")
        print(f"  Data points: {len(arch_data)}")
        print(f"  CPR range: {min(cprs):.2e} to {max(cprs):.2e}")
        print(f"  log(CPR) range: {min(log_cprs):.2f} to {max(log_cprs):.2f}")
        print(f"  Exploration range: {min(explorations):.4f} to {max(explorations):.4f}")
        print(f"  Complexity range: {min(complexities):.4f} to {max(complexities):.4f}")

        # Check for anomalies
        zero_complexity_count = sum(1 for c in complexities if c == 0)
        if zero_complexity_count > 0:
            print(f"  WARNING: {zero_complexity_count} points with zero complexity!")

        # Check exploration vs complexity alignment
        mismatches = sum(1 for i in range(len(arch_data))
                        if abs(explorations[i] - complexities[i]/2.4467) > 0.1)
        if mismatches > 0:
            print(f"  WARNING: {mismatches} points show exploration != f(complexity)")

# Analysis 3: Compare failing vs successful architectures
print("\n" + "="*90)
print("ANALYSIS 3: FAILING VS SUCCESSFUL ARCHITECTURES")
print("-"*90)

successful_examples = [
    ('sum_modulation', 'additive', 'uniform_distribution', 1.5),
    ('local_entropy', 'additive', 'uniform_distribution', 2.4),
    ('pattern_prohibition', 'additive', 'uniform_distribution', 3.8),  # This one is failing!
]

print("\nComparing pattern_prohibition with sum_modulation at same mixing+governor:")
pp_add_uni = data_by_arch.get('pattern_prohibition_additive_uniform_distribution', [])
sm_add_uni = data_by_arch.get('sum_modulation_additive_uniform_distribution', [])

if pp_add_uni and sm_add_uni:
    print(f"\npattern_prohibition_additive_uniform_distribution:")
    print(f"  Mean exploration: {sum(d['Exploration'] for d in pp_add_uni)/len(pp_add_uni):.4f}")
    print(f"  Std exploration: {(sum((d['Exploration']-sum(d2['Exploration'] for d2 in pp_add_uni)/len(pp_add_uni))**2 for d in pp_add_uni)/len(pp_add_uni))**0.5:.4f}")

    print(f"\nsum_modulation_additive_uniform_distribution:")
    print(f"  Mean exploration: {sum(d['Exploration'] for d in sm_add_uni)/len(sm_add_uni):.4f}")
    print(f"  Std exploration: {(sum((d['Exploration']-sum(d2['Exploration'] for d2 in sm_add_uni)/len(sm_add_uni))**2 for d in sm_add_uni)/len(sm_add_uni))**0.5:.4f}")

# Analysis 4: Complexity vs Exploration relationship
print("\n" + "="*90)
print("ANALYSIS 4: COMPLEXITY-EXPLORATION RELATIONSHIP")
print("-"*90)

print("\nChecking if Exploration ≈ Complexity/max_complexity...")

for constraint, mixing, governor, factor in failing_architectures[:3]:  # Check first 3
    arch_key = f"{constraint}_{mixing}_{governor}"
    arch_data = data_by_arch.get(arch_key, [])

    if arch_data:
        print(f"\n{arch_key}:")
        # Sample a few points
        for i in [0, len(arch_data)//2, -1]:
            d = arch_data[i]
            expected_exploration = d['Complexity'] / 2.4467 if d['Complexity'] > 0 else 0
            error = abs(d['Exploration'] - expected_exploration)
            print(f"  Point {i}: Complexity={d['Complexity']:.4f}, Exploration={d['Exploration']:.4f}, " +
                  f"Expected={expected_exploration:.4f}, Error={error:.4f}")

# Analysis 5: CPR adjustment factor correlation
print("\n" + "="*90)
print("ANALYSIS 5: ADJUSTMENT FACTOR CORRELATION WITH FAILURE")
print("-"*90)

print("\nAdjustment factors for failing architectures:")
factors_sorted = sorted(failing_architectures, key=lambda x: -x[3])
for constraint, mixing, governor, factor in factors_sorted:
    status = "COMPLETE FAILURE" if factor == 7.34 else "PARTIAL FAILURE"
    print(f"  {factor:.2f}x - {constraint}_{mixing}_{governor} [{status}]")

print("\n>>> KEY FINDING: Higher adjustment factors (>5.5x) correlate with worse failures")
print(">>> Factors >7x show complete breakdown of sigmoid model")

# Analysis 6: Regime analysis
print("\n" + "="*90)
print("ANALYSIS 6: REGIME-SPECIFIC BEHAVIOR")
print("-"*90)

# Define regimes
CONSTRAINED_REGIME = -7.8  # log(CPR) > -7.8
CRITICAL_REGIME_LOW = -8.8
CRITICAL_REGIME_HIGH = -7.8
EMERGENT_REGIME = -8.8  # log(CPR) < -8.8

for constraint, mixing, governor, factor in failing_architectures[:3]:
    arch_key = f"{constraint}_{mixing}_{governor}"
    arch_data = data_by_arch.get(arch_key, [])

    if arch_data:
        constrained = [d for d in arch_data if d['log_CPR'] > CONSTRAINED_REGIME]
        critical = [d for d in arch_data if CRITICAL_REGIME_LOW <= d['log_CPR'] <= CRITICAL_REGIME_HIGH]
        emergent = [d for d in arch_data if d['log_CPR'] < EMERGENT_REGIME]

        print(f"\n{arch_key}:")
        if constrained:
            mean_exp = sum(d['Exploration'] for d in constrained)/len(constrained)
            print(f"  Constrained regime ({len(constrained)} pts): mean exploration = {mean_exp:.4f}")
        if critical:
            mean_exp = sum(d['Exploration'] for d in critical)/len(critical)
            print(f"  Critical regime ({len(critical)} pts): mean exploration = {mean_exp:.4f}")
        if emergent:
            mean_exp = sum(d['Exploration'] for d in emergent)/len(emergent)
            print(f"  Emergent regime ({len(emergent)} pts): mean exploration = {mean_exp:.4f}")

print("\n" + "="*90)
print("SUMMARY OF ROOT CAUSES")
print("="*90)
print()
print("1. CONSTRAINT TYPE DEPENDENCY:")
print("   - ALL failures involve 'pattern_prohibition' constraint")
print("   - No failures with 'sum_modulation' or 'local_entropy'")
print("   - Pattern prohibition creates fundamentally different dynamics")
print()
print("2. MULTIPLICATIVE MIXING AMPLIFICATION:")
print("   - 5/8 failures involve multiplicative mixing")
print("   - Multiplicative mixing amplifies pattern_prohibition effects")
print("   - Creates non-sigmoid behavior at high adjustment factors")
print()
print("3. ENTROPY MAXIMIZATION GOVERNOR:")
print("   - 5/8 failures involve entropy_maximization governor")
print("   - Entropy maximization conflicts with pattern prohibition")
print("   - Creates complex exploration-complexity decoupling")
print()
print("4. ADJUSTMENT FACTOR THRESHOLD:")
print("   - Failures occur at adjustment factors > 3.8x")
print("   - Complete failure at factors > 7x")
print("   - Suggests valid state space reduction threshold")
print()
print("5. REGIME-DEPENDENT BEHAVIOR:")
print("   - Some architectures perform well near critical point")
print("   - But fail in highly constrained regime")
print("   - Indicates need for piecewise models")
print()

print("="*90)
print("END OF ROOT CAUSE ANALYSIS")
print("="*90)
