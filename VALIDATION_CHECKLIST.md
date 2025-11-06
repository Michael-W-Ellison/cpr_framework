# Chart Validation Checklist

Use this checklist to verify the phase transition visualization chart is working correctly.

## Pre-Testing Setup

- [ ] Open `/home/ubuntu/Desktop/CPR/cpr_framework_demo.html` in a modern browser (Chrome, Firefox, Edge, Safari)
- [ ] Open browser Developer Tools (F12 or right-click → Inspect)
- [ ] Navigate to the Console tab
- [ ] Clear the console for clean output

## Phase 1: Initial Load

### Chart.js Loading
- [ ] Console shows no red errors
- [ ] Console shows "Annotation plugin registered" (or warning message)
- [ ] Console shows "Chart created successfully"
- [ ] No "Chart.js failed to load" error message

### Visual Inspection
- [ ] Navigate to "Interactive Explorer" tab
- [ ] Chart section is visible below the control and results panels
- [ ] "Loading chart..." message disappears quickly
- [ ] Chart canvas is visible and fills its container

## Phase 2: Chart Components

### Sigmoid Curve
- [ ] Purple/blue sigmoid curve is visible
- [ ] Curve is smooth (no jagged edges or gaps)
- [ ] Curve shows characteristic S-shape
- [ ] Curve extends across the visible x-axis range
- [ ] Fill area below curve is visible (light purple/blue)

### Current Point Marker
- [ ] A marker point is visible on the chart
- [ ] For "local_entropy" or "sum_modulation": marker is a RED CIRCLE
- [ ] For "pattern_prohibition": marker is a BLUE TRIANGLE
- [ ] Marker appears at a reasonable position on the curve

### Axis Labels
- [ ] X-axis label reads "log₁₀(Adjusted CPR)"
- [ ] Y-axis label reads "Predicted Exploration"
- [ ] Y-axis shows percentage values (0%, 20%, 40%, 60%, 80%, 100%)
- [ ] Axis labels are bold and clearly visible

### Legend
- [ ] Legend appears at top of chart
- [ ] Shows "Universal Sigmoid (Density-Based)" with purple color
- [ ] Shows current config label ("Current Config (CPR Model)" or "Current Config (Complexity Model)")
- [ ] Legend items have appropriate symbols (line for sigmoid, dot/triangle for current point)

### Regime Boundary Lines (if visible)
- [ ] Vertical dashed line may appear at x = -7.8 (red, labeled "Constrained→Critical")
- [ ] Vertical dashed line may appear at x = -8.8 (green, labeled "Critical→Emergent")
- [ ] Lines have labels with colored backgrounds
- [ ] Note: These lines only appear if they fall within the visible x-axis range

## Phase 3: Interactive Features

### Slider Updates - System Size
- [ ] Move "System Size (n)" slider to 6
- [ ] Chart updates immediately (no delay > 1 second)
- [ ] Current point moves to new position
- [ ] Console shows "Chart created successfully"
- [ ] Move slider to 30
- [ ] Chart updates again
- [ ] Current point moves to very far left (low CPR region)

### Slider Updates - Base
- [ ] Move "Base (b)" slider to 2
- [ ] Chart updates immediately
- [ ] Current point moves to new position
- [ ] Move slider to 20
- [ ] Chart updates again
- [ ] Current point moves to different position

### Slider Updates - Complexity
- [ ] Set constraint type to "Pattern Prohibition"
- [ ] Move "Complexity (C)" slider
- [ ] Chart updates (current point y-value changes)
- [ ] Marker changes to BLUE TRIANGLE

### Dropdown Updates - Constraint Type
- [ ] Select "Pattern Prohibition (Structure-Based)"
- [ ] Marker changes to blue triangle
- [ ] Model indicator shows "Using Complexity Model (Structure-Based)"
- [ ] Select "Local Entropy (Density-Based)"
- [ ] Marker changes to red circle
- [ ] Model indicator shows "Using CPR Sigmoid Model (Density-Based)"
- [ ] Select "Sum Modulation (Density-Based)"
- [ ] Marker stays as red circle
- [ ] Model indicator shows "Using CPR Sigmoid Model (Density-Based)"

### Dropdown Updates - Mixing Type
- [ ] Change "Mixing Type" dropdown
- [ ] Chart updates for each selection
- [ ] Adjustment factor value changes in results panel
- [ ] Current point position may shift

### Dropdown Updates - Governor Type
- [ ] Change "Governor Type" dropdown
- [ ] Chart updates for each selection
- [ ] Adjustment factor value may change
- [ ] Current point position may shift

## Phase 4: Preset Configurations

### Constrained Preset
- [ ] Click "Constrained" button
- [ ] All parameters update instantly
- [ ] Chart shows point in constrained regime (high x-value, low y-value)
- [ ] Regime indicator shows "Constrained Regime" with red background
- [ ] Constraint type is "Pattern Prohibition"

### Critical Preset
- [ ] Click "Critical" button
- [ ] All parameters update instantly
- [ ] Chart shows point near critical region (x around -8 to -9)
- [ ] Regime indicator shows "Critical Regime" with orange background
- [ ] Constraint type is "Local Entropy"
- [ ] Both boundary lines should be visible

### Emergent Preset
- [ ] Click "Emergent" button
- [ ] All parameters update instantly
- [ ] Chart shows point in emergent regime (very low x-value, high y-value)
- [ ] Regime indicator shows "Emergent Regime" with green background
- [ ] Constraint type is "Sum Modulation"

## Phase 5: Tooltip and Hover

### Sigmoid Curve Hover
- [ ] Hover mouse over purple sigmoid curve
- [ ] Tooltip appears showing "Sigmoid Model: XX.XX%"
- [ ] Tooltip follows mouse along the curve
- [ ] Percentage value changes as you move along curve

### Current Point Hover
- [ ] Hover mouse over current point marker
- [ ] Tooltip appears showing configuration details
- [ ] Shows current exploration percentage
- [ ] Shows current log₁₀(CPR) value
- [ ] Format: "Current: XX.XX% @ log₁₀(CPR) = -X.XX"

## Phase 6: Responsive Behavior

### Window Resize
- [ ] Resize browser window to narrower width
- [ ] Chart resizes to fit container
- [ ] Chart remains readable and proportional
- [ ] Resize window to wider width
- [ ] Chart expands appropriately
- [ ] No distortion or clipping occurs

## Phase 7: Console Validation

### Expected Console Messages (in order)
- [ ] "Annotation plugin registered" (or warning if not found)
- [ ] "Generated XXX chart data points" (should be 600+)
- [ ] "Current config: logCPR=X.XX, exploration=X.XXXX, model=xxx"
- [ ] "Creating chart with x range [X, Y]"
- [ ] "Annotations:" followed by object keys
- [ ] "Chart created successfully"

### No Error Messages
- [ ] No red error messages in console
- [ ] No "undefined" errors
- [ ] No "Chart.js failed to load" errors
- [ ] No "Canvas element not found" errors
- [ ] No "TypeError" or "ReferenceError" messages

## Phase 8: Edge Cases

### Maximum Values
- [ ] Set System Size to 50 (maximum)
- [ ] Set Base to 50 (maximum)
- [ ] Chart handles without errors
- [ ] Console shows no overflow warnings
- [ ] Current point appears (may be far off to the left)

### Minimum Values
- [ ] Set System Size to 3 (minimum)
- [ ] Set Base to 2 (minimum)
- [ ] Chart handles without errors
- [ ] Current point appears in constrained region

### All Constraint Type Combinations
For each constraint type, test with different mixing and governor types:
- [ ] pattern_prohibition + additive + uniform_distribution
- [ ] pattern_prohibition + multiplicative + entropy_maximization
- [ ] local_entropy + additive + novelty_seeking
- [ ] sum_modulation + triple_sum + uniform_distribution
- [ ] (Test at least 4-5 different combinations)
- [ ] All combinations work without errors
- [ ] Chart updates correctly for each

## Phase 9: Tab Switching

### Navigation Test
- [ ] Switch to "Overview" tab - no errors
- [ ] Switch back to "Interactive Explorer" tab
- [ ] Chart is still visible and functional
- [ ] Switch to "Theory" tab - no errors
- [ ] Switch back to "Interactive Explorer" tab
- [ ] Chart updates when parameters change

## Phase 10: Performance

### Responsiveness
- [ ] Slider movements feel smooth (< 100ms update lag)
- [ ] Chart redraws quickly (< 500ms)
- [ ] No visible flickering during updates
- [ ] Browser remains responsive (no freezing)

### Memory
- [ ] Open multiple tabs and switch between them repeatedly
- [ ] No memory leaks (check browser Task Manager if available)
- [ ] Chart continues to work after 10+ parameter changes

## Summary Checklist

Core Functionality:
- [ ] Chart renders on page load
- [ ] Sigmoid curve is visible and smooth
- [ ] Current point marker appears correctly
- [ ] Chart updates when parameters change
- [ ] Regime boundary lines appear (when in range)

Visual Quality:
- [ ] Chart is visually appealing
- [ ] Colors are clear and distinct
- [ ] Text is readable
- [ ] No visual artifacts or rendering issues

Interactivity:
- [ ] Tooltips work correctly
- [ ] Sliders trigger updates
- [ ] Dropdowns trigger updates
- [ ] Preset buttons work
- [ ] Responsive to window resize

Technical:
- [ ] No console errors
- [ ] Proper logging messages appear
- [ ] Chart.js and plugins load correctly
- [ ] All event listeners function

## If Any Test Fails

1. **Check browser console** for specific error messages
2. **Verify CDN connectivity** - ensure https://cdnjs.cloudflare.com is accessible
3. **Try a different browser** to isolate browser-specific issues
4. **Clear browser cache** and reload the page
5. **Check the browser version** - ensure using modern browser (not IE11)
6. **Review the error stack trace** in console for debugging clues

## Success Criteria

The chart is considered fully functional if:
- ✅ 95%+ of checklist items pass
- ✅ No critical errors in console
- ✅ Chart renders and updates smoothly
- ✅ All three presets work correctly
- ✅ Both model types (CPR and Complexity) work
- ✅ Chart remains stable after 20+ parameter changes

## Files to Reference

- Main file: `/home/ubuntu/Desktop/CPR/cpr_framework_demo.html`
- Test file: `/home/ubuntu/Desktop/CPR/test_chart.html`
- Fix summary: `/home/ubuntu/Desktop/CPR/CHART_FIXES_SUMMARY.md`
- Code reference: `/home/ubuntu/Desktop/CPR/CORRECTED_CHART_CODE.js`
- This checklist: `/home/ubuntu/Desktop/CPR/VALIDATION_CHECKLIST.md`
