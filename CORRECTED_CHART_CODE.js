// ============================================================================
// CORRECTED CHART CODE FOR CPR FRAMEWORK DEMO
// ============================================================================
// This file contains the key corrected sections of the chart implementation
// All fixes are compatible with Chart.js v4.4.0 and chartjs-plugin-annotation v3.0.1
// ============================================================================

// SECTION 1: IMPROVED DATA GENERATION
// ----------------------------------------------------------------------------
// Generates smooth sigmoid curve with extra density in critical region
function generateChartData() {
    const data = [];

    // Generate points from -50 to -1 with step 0.1
    for (let logCPR = CHART_RANGE.LOG_CPR_MIN; logCPR <= -1; logCPR += CHART_RANGE.LOG_CPR_STEP) {
        const exploration = sigmoid(logCPR, SIGMOID_L, SIGMOID_K, SIGMOID_X0);

        if (isFinite(exploration) && exploration >= 0 && exploration <= 1.1) {
            data.push({ x: logCPR, y: exploration });
        }
    }

    // Add extra dense points around critical region for smooth curve
    for (let logCPR = -10; logCPR <= -7; logCPR += 0.02) {
        const exploration = sigmoid(logCPR, SIGMOID_L, SIGMOID_K, SIGMOID_X0);

        if (isFinite(exploration) && exploration >= 0 && exploration <= 1.1) {
            data.push({ x: logCPR, y: exploration });
        }
    }

    // Sort by x value for proper line rendering
    data.sort((a, b) => a.x - b.x);

    return data;
}

// SECTION 2: CORRECTED CHART UPDATE FUNCTION
// ----------------------------------------------------------------------------
// Main function to create/update the phase transition chart
function updateChart(currentAdjustedCPR, currentExploration, modelType) {
    if (!chartJsLoaded) {
        console.warn('Chart.js not loaded yet');
        return;
    }

    const loadingElement = document.getElementById('chartLoading');
    loadingElement.classList.remove('hidden');
    loadingElement.textContent = 'Loading chart...';

    const ctx = document.getElementById('phaseChart');
    if (!ctx) {
        console.error('Canvas element not found');
        loadingElement.textContent = 'Error: Canvas element not found';
        return;
    }

    if (chart) {
        chart.destroy();
        chart = null;
    }

    const chartData = generateChartData();
    console.log(`Generated ${chartData.length} chart data points`);

    const currentLogCPR = Math.log10(currentAdjustedCPR);
    console.log(`Current config: logCPR=${currentLogCPR.toFixed(2)}, exploration=${currentExploration.toFixed(4)}, model=${modelType}`);

    const xMin = Math.max(-50, Math.min(-15, currentLogCPR - 10));
    const xMax = Math.min(0, Math.max(-5, currentLogCPR + 5));

    // Dataset 1: Sigmoid curve (always shown)
    const datasets = [{
        label: 'Universal Sigmoid (Density-Based)',
        data: chartData,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 3,
        showLine: true  // IMPORTANT: explicitly enable line rendering for scatter chart
    }];

    // Dataset 2: Current point (style varies by model type)
    if (modelType === 'complexity_based') {
        datasets.push({
            label: 'Current Config (Complexity Model)',
            data: [{ x: currentLogCPR, y: currentExploration }],
            backgroundColor: '#3b82f6',
            borderColor: '#1e40af',
            pointRadius: 12,
            pointHoverRadius: 14,
            showLine: false,
            pointStyle: 'triangle'  // Triangle for structure-based constraints
        });
    } else {
        datasets.push({
            label: 'Current Config (CPR Model)',
            data: [{ x: currentLogCPR, y: currentExploration }],
            backgroundColor: '#e53e3e',
            borderColor: '#c53030',
            pointRadius: 10,
            pointHoverRadius: 12,
            showLine: false  // Circle for density-based constraints
        });
    }

    // Build annotations object (Chart.js v4 syntax)
    const annotations = {};

    // Annotation 1: Constrained → Critical boundary
    if (REGIME_BOUNDARIES.CONSTRAINED_CRITICAL >= xMin &&
        REGIME_BOUNDARIES.CONSTRAINED_CRITICAL <= xMax) {
        annotations.constrainedLine = {
            type: 'line',
            scaleID: 'x',  // NEW in v4: specify which axis
            value: REGIME_BOUNDARIES.CONSTRAINED_CRITICAL,  // NEW in v4: single value
            borderColor: 'rgba(197, 48, 48, 0.6)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'Constrained→Critical',
                display: true,  // NEW in v4: display instead of enabled
                position: 'start',
                backgroundColor: 'rgba(197, 48, 48, 0.8)',
                color: 'white',
                font: {
                    size: 10
                }
            }
        };
    }

    // Annotation 2: Critical → Emergent boundary
    if (REGIME_BOUNDARIES.CRITICAL_EMERGENT >= xMin &&
        REGIME_BOUNDARIES.CRITICAL_EMERGENT <= xMax) {
        annotations.emergentLine = {
            type: 'line',
            scaleID: 'x',
            value: REGIME_BOUNDARIES.CRITICAL_EMERGENT,
            borderColor: 'rgba(56, 161, 105, 0.6)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'Critical→Emergent',
                display: true,
                position: 'start',
                backgroundColor: 'rgba(56, 161, 105, 0.8)',
                color: 'white',
                font: {
                    size: 10
                }
            }
        };
    }

    try {
        console.log(`Creating chart with x range [${xMin}, ${xMax}]`);
        console.log(`Annotations:`, Object.keys(annotations));

        chart = new Chart(ctx, {
            type: 'scatter',  // IMPORTANT: use scatter type for mixed line/point data
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',  // Explicit linear scale
                        title: {
                            display: true,
                            text: 'log₁₀(Adjusted CPR)',
                            font: { size: 14, weight: 'bold' }
                        },
                        min: xMin,
                        max: xMax,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Predicted Exploration',
                            font: { size: 14, weight: 'bold' }
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    return `Sigmoid Model: ${(context.parsed.y * 100).toFixed(2)}%`;
                                } else {
                                    return `Current: ${(context.parsed.y * 100).toFixed(2)}% @ log₁₀(CPR) = ${context.parsed.x.toFixed(2)}`;
                                }
                            }
                        }
                    },
                    annotation: {
                        annotations: annotations  // Include annotations from initial config
                    }
                }
            }
        });

        console.log('Chart created successfully');
        loadingElement.classList.add('hidden');
    } catch (e) {
        console.error('Error creating chart:', e);
        console.error('Error stack:', e.stack);
        loadingElement.textContent = 'Error creating chart: ' + e.message;
    }
}

// SECTION 3: INITIALIZATION CODE
// ----------------------------------------------------------------------------
// Proper initialization in DOMContentLoaded event
document.addEventListener('DOMContentLoaded', function() {
    // Check if Chart.js loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js failed to load');
        document.getElementById('chartLoading').textContent = 'Error: Chart.js failed to load. Please refresh the page.';
        return;
    }

    // Register annotation plugin (Chart.js v4 requires explicit registration)
    if (typeof window.ChartAnnotation !== 'undefined') {
        Chart.register(window.ChartAnnotation);
        console.log('Annotation plugin registered');
    } else {
        console.warn('Annotation plugin not found - boundary lines will not be displayed');
    }

    chartJsLoaded = true;

    // Initialize tabs
    initTabs();

    // Add event listeners for sliders
    document.getElementById('systemSize').addEventListener('input', updateCalculations);
    document.getElementById('systemBase').addEventListener('input', updateCalculations);
    document.getElementById('complexitySlider').addEventListener('input', updateCalculations);

    // Add event listeners for selects
    document.getElementById('constraintType').addEventListener('change', updateCalculations);
    document.getElementById('mixingType').addEventListener('change', updateCalculations);
    document.getElementById('governorType').addEventListener('change', updateCalculations);

    // Add event listeners for preset buttons
    document.getElementById('btn-constrained').addEventListener('click', function() {
        loadPreset('constrained');
    });
    document.getElementById('btn-critical').addEventListener('click', function() {
        loadPreset('critical');
    });
    document.getElementById('btn-emergent').addEventListener('click', function() {
        loadPreset('emergent');
    });

    // Initial calculation (triggers first chart render)
    updateCalculations();
});

// ============================================================================
// KEY DIFFERENCES FROM ORIGINAL CODE
// ============================================================================
/*
1. Chart type changed from 'line' to 'scatter' for better handling of mixed datasets
2. Annotation syntax updated for Chart.js v4 (scaleID, value, display)
3. Annotations built as object instead of array
4. Annotations included in initial config instead of updating after creation
5. Data generation improved with denser points in critical region
6. Explicit showLine: true for sigmoid dataset
7. Better error handling and console logging
8. Plugin registration added (required for Chart.js v4)
9. Canvas element validation added
10. Y-axis tick formatting added for percentage display

All changes maintain backward compatibility with the rest of the codebase
and improve robustness and user experience.
*/
