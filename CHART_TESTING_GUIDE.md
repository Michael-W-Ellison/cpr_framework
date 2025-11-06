# Phase Transition Chart - Testing Guide

**Status**: ✅ All fixes applied - Ready for testing
**Date**: 2025-10-15

---

## What Was Fixed

The phase transition visualization chart has been completely repaired with the following fixes:

### Critical Fixes Applied ✅

1. **Chart.js v4 Compatibility**
   - Updated to Chart.js v4.4.0
   - Registered annotation plugin explicitly
   - Updated all API calls to v4 syntax

2. **Annotation Plugin**
   - Changed from v3 syntax (`xMin`/`xMax`) to v4 syntax (`scaleID`/`value`)
   - Properly registered plugin on page load
   - Added fallback warning if plugin not loaded

3. **Chart Type**
   - Changed from `'line'` to `'scatter'` for better data handling
   - Added `showLine: true` to sigmoid dataset
   - Properly handles mixed line/scatter data

4. **Data Generation**
   - Base points: -50 to -1 with 0.1 step (~490 points)
   - Dense critical region: -10 to -7 with 0.02 step (~150 extra points)
   - Total: ~650 points for smooth curve
   - Sorted by x-value for proper rendering

5. **Error Handling**
   - Canvas element validation
   - Try-catch around chart creation
   - Comprehensive console logging
   - User-friendly error messages

6. **Visual Improvements**
   - Y-axis shows percentages (0%-100%)
   - Proper color coding (red for CPR model, blue for complexity model)
   - Different marker shapes (circle vs triangle)
   - Smooth curve rendering
   - Professional styling

---

## How to Test

### Step 1: Open the File
```bash
# Navigate to the directory
cd /home/ubuntu/Desktop/CPR

# Open in browser (choose one):
google-chrome cpr_framework_demo.html    # Chrome
firefox cpr_framework_demo.html          # Firefox
open cpr_framework_demo.html             # macOS Safari
```

Or simply double-click `cpr_framework_demo.html` in your file manager.

---

### Step 2: Open Developer Console

**Chrome/Edge**: Press `F12` or `Ctrl+Shift+J` (Windows/Linux) or `Cmd+Option+J` (Mac)
**Firefox**: Press `F12` or `Ctrl+Shift+K` (Windows/Linux) or `Cmd+Option+K` (Mac)
**Safari**: Enable Developer menu first, then press `Cmd+Option+C`

---

### Step 3: Navigate to Interactive Explorer Tab

1. Click on the **"🔬 Interactive Explorer"** tab
2. Look at the console - you should see:
   ```
   Annotation plugin registered
   Generated 640 chart data points
   Current config: logCPR=-4.12, exploration=0.8513, model=cpr_based
   Creating chart with x range [-14.12, 0.88]
   Annotations: constrainedLine
   Chart created successfully
   ```

---

### Step 4: Visual Verification

The chart should display:

#### ✅ Sigmoid Curve
- **Color**: Purple/blue (#667eea)
- **Shape**: Smooth S-curve
- **Range**: Visible across the x-axis
- **Fill**: Light purple fill under curve
- **Line**: Thick (3px), smooth (tension 0.4)

#### ✅ Current Configuration Marker

**For Density-Based Constraints** (sum_modulation, local_entropy):
- **Shape**: Circle
- **Color**: Red (#e53e3e)
- **Size**: Medium (10px radius)
- **Label**: "Current Config (CPR Model)"

**For Structure-Based Constraints** (pattern_prohibition):
- **Shape**: Triangle
- **Color**: Blue (#3b82f6)
- **Size**: Medium (12px radius)
- **Label**: "Current Config (Complexity Model)"

#### ✅ Regime Boundary Lines (when visible)

**Constrained → Critical** (at x = -7.8):
- **Style**: Vertical dashed line
- **Color**: Red (rgba(197, 48, 48, 0.6))
- **Label**: "Constrained→Critical"

**Critical → Emergent** (at x = -8.8):
- **Style**: Vertical dashed line
- **Color**: Green (rgba(56, 161, 105, 0.6))
- **Label**: "Critical→Emergent"

#### ✅ Axes

**X-Axis**:
- **Label**: "log₁₀(Adjusted CPR)"
- **Range**: Dynamic (adjusts to show current point)
- **Grid**: Light gray lines

**Y-Axis**:
- **Label**: "Predicted Exploration"
- **Range**: 0% to 100%
- **Format**: Percentages (0%, 25%, 50%, 75%, 100%)
- **Grid**: Light gray lines

#### ✅ Legend
- **Position**: Top of chart
- **Items**:
  - Universal Sigmoid curve
  - Current configuration marker
- **Style**: Uses point style icons

---

### Step 5: Interactive Testing

Test all interactive features to ensure chart updates properly:

#### A. Preset Buttons
Click each preset and verify chart updates:

1. **"Constrained" button**:
   - Should show point on right side (high log CPR)
   - Point should be low on Y-axis (low exploration)
   - May show red boundary line

2. **"Critical" button**:
   - Should show point in middle region
   - Point should be mid-range on Y-axis
   - Both boundary lines may be visible

3. **"Emergent" button**:
   - Should show point on left side (low log CPR)
   - Point should be high on Y-axis (high exploration)
   - May show green boundary line

#### B. System Size Slider
Move the slider from 3 to 50:
- Chart should update smoothly
- Point should move left (decreasing CPR)
- Console should show: "Generated XXX chart data points"
- Console should show: "Chart created successfully"

#### C. Base Slider
Move the slider from 2 to 50:
- Chart should update smoothly
- Point should move left (decreasing CPR)
- Updates should be responsive

#### D. Constraint Type Selector

**Switch to "Pattern Prohibition"**:
- Marker should change to **blue triangle**
- Label should say "Complexity Model"

**Switch to "Local Entropy"**:
- Marker should change to **red circle**
- Label should say "CPR Model"

**Switch to "Sum Modulation"**:
- Marker should remain **red circle**
- Label should say "CPR Model"

#### E. Mixing and Governor Selectors
- Change various combinations
- Chart should update each time
- Adjustment factor should change in results panel
- Point position may shift slightly

#### F. Complexity Slider (for Pattern Prohibition only)
- Select "Pattern Prohibition" constraint
- Move complexity slider (0 to 2.4467)
- Point should move up/down on Y-axis
- X position stays same (based on CPR)

---

### Step 6: Tooltip Testing

Hover over different parts of the chart:

1. **Hover over sigmoid curve**:
   - Should show: "Sigmoid Model: XX.XX%"

2. **Hover over current point**:
   - Should show: "Current: XX.XX% @ log₁₀(CPR) = -X.XX"

3. **Tooltips should**:
   - Appear quickly
   - Follow cursor
   - Show accurate values
   - Be easy to read

---

## Expected Console Output

### On Page Load
```
Chart.js loaded successfully
Annotation plugin registered
```

### On Initial Chart Render
```
Generated 640 chart data points
Current config: logCPR=-4.12, exploration=0.8513, model=cpr_based
Creating chart with x range [-14.12, 0.88]
Annotations: constrainedLine
Chart created successfully
```

### On Parameter Change
```
Generated 640 chart data points
Current config: logCPR=-7.45, exploration=0.8513, model=cpr_based
Creating chart with x range [-17.45, -2.45]
Annotations: constrainedLine,emergentLine
Chart created successfully
```

---

## Troubleshooting

### Chart Not Visible

**Check 1**: Is the canvas element present?
- Look for `<canvas id="phaseChart">` in the HTML
- Should be in the "Interactive Explorer" tab

**Check 2**: Console errors?
- Any red error messages in console?
- If "Chart.js failed to load" → Check internet connection
- If "Canvas element not found" → HTML structure issue

**Check 3**: Are you on the correct tab?
- Chart only appears in "Interactive Explorer" tab
- Click the 🔬 tab to see it

### Chart Shows Error Message

**"Chart.js failed to load"**:
- Internet connection required for CDN
- Try refreshing page
- Check if https://cdnjs.cloudflare.com is accessible

**"Annotation plugin not found"**:
- Boundary lines won't show but chart should still work
- Plugin CDN might be blocked
- Chart will function without annotations

**"Canvas element not found"**:
- HTML structure issue
- Verify file wasn't corrupted
- Re-download if necessary

### Chart Doesn't Update

**Symptoms**: Moving sliders doesn't update chart

**Check 1**: Console errors?
- Look for JavaScript errors
- Red error messages indicate the problem

**Check 2**: Event listeners attached?
- Should see console logs on parameter change
- If no logs → event listeners not working

**Solution**: Refresh the page

### Curve Looks Jagged

**Symptoms**: Sigmoid curve not smooth

**Check 1**: Data point count
- Console should show "Generated ~640 chart data points"
- If lower number → data generation issue

**Check 2**: Browser performance
- Try in different browser
- Close other tabs to free resources

**Solution**: Should not occur with current implementation (650+ points)

### Boundary Lines Not Showing

**Normal Behavior**: Boundary lines only show when in visible range

**At x = -4**: Only constrained boundary visible (at -7.8)
**At x = -8**: Both boundaries might be visible
**At x = -20**: Only emergent boundary visible (at -8.8)
**At x = -40**: No boundaries visible (both outside range)

**If never showing**:
- Check console for "Annotation plugin not found"
- Not critical - chart still functions

### Performance Issues

**Symptoms**: Slow updates, lag when moving sliders

**Cause**: Chart recreation on every update (necessary for dynamic range)

**Solutions**:
1. Use preset buttons for large jumps
2. Update only when slider released (current implementation updates continuously)
3. Close other browser tabs
4. Use modern browser (Chrome, Firefox, Edge)

### Wrong Values Displayed

**Check 1**: Verify calculations
- Look at "Predictions" panel on right
- Values should match chart
- CPR, Adjusted CPR, Exploration should be consistent

**Check 2**: Console log values
- Console shows current config
- Compare with displayed values
- Should match

---

## Performance Benchmarks

### Expected Performance

- **Initial render**: < 500ms
- **Parameter update**: < 200ms
- **Smooth animations**: 60fps
- **Memory usage**: < 50MB additional

### Data Points

- Base sigmoid: ~490 points
- Dense critical region: ~150 points
- Total: ~640 points
- Update frequency: On every input change

---

## Browser Compatibility

### Fully Supported ✅
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+
- Opera 76+

### Not Supported ❌
- Internet Explorer 11 (requires ES6+)
- Older mobile browsers

### Recommended
- Latest Chrome or Firefox
- Hardware acceleration enabled
- 4GB+ RAM available

---

## Success Criteria

Your chart is working correctly if:

- [x] Chart renders on page load
- [x] Smooth sigmoid S-curve visible
- [x] Current configuration marker appears
- [x] Chart updates when parameters change
- [x] Tooltips work on hover
- [x] Y-axis shows percentages
- [x] X-axis shows log₁₀(Adjusted CPR)
- [x] Legend displays correctly
- [x] Console shows "Chart created successfully"
- [x] No red error messages in console
- [x] Preset buttons work
- [x] Sliders update chart smoothly
- [x] Constraint type selector changes marker style
- [x] Boundary lines appear when in range

**If all boxes can be checked: ✅ Chart is fully functional**

---

## Technical Details

### Chart.js Configuration
- Version: 4.4.0
- Type: scatter
- Plugins: annotation, legend, tooltip
- Responsive: true
- Maintain aspect ratio: false

### Annotation Plugin
- Version: 3.0.1
- API: v4 syntax
- Registration: Explicit via Chart.register()

### Data Generation
- Algorithm: Sigmoid function
- Parameters: L=0.8513, k=46.7978, x₀=-8.2999
- Points: 640 (490 base + 150 dense)
- Sorting: By x-value ascending

### Rendering
- Canvas API: 2D context
- Size: Responsive (fills container)
- Resolution: Device pixel ratio aware
- Animations: Smooth transitions

---

## Additional Resources

- Chart.js Documentation: https://www.chartjs.org/docs/latest/
- Annotation Plugin: https://www.chartjs.org/chartjs-plugin-annotation/
- CPR Framework Theory: See "Theory" tab in demo
- Scientific Validation: See SCIENTIFIC_VALIDATION_REPORT.md

---

**Testing Complete**: If all tests pass, the phase transition visualization is fully functional and ready for use.
