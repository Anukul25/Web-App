document.addEventListener('DOMContentLoaded', function() {
    // Add page load transition effect
    const dashboardContent = document.getElementById('dashboard-content');
    if (dashboardContent) {
        setTimeout(() => {
            dashboardContent.classList.remove('opacity-0');
            dashboardContent.classList.add('opacity-100');
        }, 100);
    }

    const stats = {
        total_customers: parseFloat(document.querySelector('meta[name="total_customers"]')?.content || '0'),
        high_churn_count: parseFloat(document.querySelector('meta[name="high_churn_count"]')?.content || '0'),
        moderate_churn_count: parseFloat(document.querySelector('meta[name="moderate_churn_count"]')?.content || '0'),
        low_churn_count: parseFloat(document.querySelector('meta[name="low_churn_count"]')?.content || '0'),
        high_churn_percent: parseFloat(document.querySelector('meta[name="high_churn_percent"]')?.content || '0'),
        moderate_churn_percent: parseFloat(document.querySelector('meta[name="moderate_churn_percent"]')?.content || '0'),
        low_churn_percent: parseFloat(document.querySelector('meta[name="low_churn_percent"]')?.content || '0'),
        model_metrics: {
            accuracy: parseFloat(document.querySelector('meta[name="model_metrics_accuracy"]')?.content || '0'),
            precision: parseFloat(document.querySelector('meta[name="model_metrics_precision"]')?.content || '0'),
            recall: parseFloat(document.querySelector('meta[name="model_metrics_recall"]')?.content || '0'),
            f1_score: parseFloat(document.querySelector('meta[name="model_metrics_f1_score"]')?.content || '0')
        },
        transaction_volume_dist: {
            high: parseFloat(document.querySelector('meta[name="transaction_volume_high"]')?.content || '50.0'),
            moderate: parseFloat(document.querySelector('meta[name="transaction_volume_moderate"]')?.content || '30.0'),
            low: parseFloat(document.querySelector('meta[name="transaction_volume_low"]')?.content || '20.0')
        },
        online_usage_dist: {
            high: parseFloat(document.querySelector('meta[name="online_usage_high"]')?.content || '60.0'),
            moderate: parseFloat(document.querySelector('meta[name="online_usage_moderate"]')?.content || '25.0'),
            low: parseFloat(document.querySelector('meta[name="online_usage_low"]')?.content || '15.0')
        },
        complaints_dist: {
            high: parseFloat(document.querySelector('meta[name="complaints_high"]')?.content || '45.0'),
            moderate: parseFloat(document.querySelector('meta[name="complaints_moderate"]')?.content || '35.0'),
            low: parseFloat(document.querySelector('meta[name="complaints_low"]')?.content || '20.0')
        },
        complaints_per_transaction_dist: {
            high: parseFloat(document.querySelector('meta[name="complaints_per_transaction_high"]')?.content || '55.0'),
            moderate: parseFloat(document.querySelector('meta[name="complaints_per_transaction_moderate"]')?.content || '25.0'),
            low: parseFloat(document.querySelector('meta[name="complaints_per_transaction_low"]')?.content || '20.0')
        }
    };

    console.log("DEBUG: Stats from Meta Tags:", stats);

    const manualBtn = document.getElementById('manual-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const recordCard = document.getElementById('record-card');
    const runPredictionCard = document.getElementById('run-prediction-card');
    const miniResultCard = document.getElementById('mini-result-card');
    const uploadContainer = document.getElementById('upload-container');
    const uploadErrorCard = document.getElementById('upload-error-card');
    const retryUploadBtn = document.getElementById('retry-upload-btn');
    const datasetStatsCard = document.getElementById('dataset-stats-card');
    const redundancyCard = document.getElementById('redundancy-card');

    const showUploadError = document.querySelector('input[name="show_upload_error"]')?.value === 'true' || false;
    const showMiniResult = document.querySelector('input[name="show_mini_result"]')?.value === 'true' || false;

    let manualWorkflowActive = false;
    let workflowLocked = false;

    function scrollToElement(element) {
        if (!element) return;
        const elementRect = element.getBoundingClientRect();
        const elementHeight = elementRect.height;
        const viewportHeight = window.innerHeight;
        const scrollPosition = elementRect.top + window.scrollY - (viewportHeight / 2) + (elementHeight / 2);
        window.scrollTo({ top: scrollPosition, behavior: 'smooth' });
    }

    function showOnlyCard(cardToShow) {
        const allCards = [recordCard, runPredictionCard, miniResultCard, uploadContainer, uploadErrorCard, datasetStatsCard, redundancyCard];
        allCards.forEach(card => {
            if (card && card !== cardToShow) {
                card.classList.add('hidden');
                card.classList.add('opacity-0');
            }
        });
        if (cardToShow) {
            cardToShow.classList.remove('hidden');
            setTimeout(() => {
                cardToShow.classList.remove('opacity-0');
                scrollToElement(cardToShow);
            }, 10);
        }
    }

    if (manualBtn && uploadBtn && recordCard && uploadContainer) {
        const hasRecordCount = document.querySelector('input[name="record_count"][type="hidden"]');
        if (miniResultCard && (showMiniResult || !miniResultCard.classList.contains('hidden'))) {
            workflowLocked = true;
            showOnlyCard(miniResultCard);
            console.log("DEBUG: Mini result card is visible on load or show_mini_result is true, locking workflow");
        } else if (redundancyCard && !redundancyCard.classList.contains('hidden')) {
            manualWorkflowActive = false;
            showOnlyCard(redundancyCard);
            uploadBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.remove('border-gray-300');
            manualBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.add('border-gray-300');
            console.log("DEBUG: Redundancy card is visible on load, upload workflow active");
        } else if (datasetStatsCard && !datasetStatsCard.classList.contains('hidden')) {
            manualWorkflowActive = false;
            showOnlyCard(datasetStatsCard);
            uploadBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.remove('border-gray-300');
            manualBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.add('border-gray-300');
            console.log("DEBUG: Dataset stats card is visible on load, upload workflow active");
        } else if (runPredictionCard && !runPredictionCard.classList.contains('hidden')) {
            manualWorkflowActive死者active = true;
            showOnlyCard(runPredictionCard);
            manualBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.remove('border-gray-300');
            uploadBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.add('border-gray-300');
            console.log("DEBUG: Run prediction card is visible on load, manual workflow active");
        } else if (hasRecordCount) {
            manualWorkflowActive = true;
            showOnlyCard(recordCard);
            manualBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.remove('border-gray-300');
            uploadBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.add('border-gray-300');
            console.log("DEBUG: Record card is visible on load, manual workflow active");
        } else if (showUploadError && uploadErrorCard) {
            manualWorkflowActive = false;
            showOnlyCard(uploadErrorCard);
            uploadBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.remove('border-gray-300');
            manualBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.add('border-gray-300');
            console.log("DEBUG: Upload error card is visible on load, upload workflow active");
        } else {
            recordCard.classList.add('hidden');
            uploadContainer.classList.add('hidden');
            if (uploadErrorCard) uploadErrorCard.classList.add('hidden');
            if (datasetStatsCard) datasetStatsCard.classList.add('hidden');
            if (redundancyCard) redundancyCard.classList.add('hidden');
            if (runPredictionCard) runPredictionCard.classList.add('hidden');
            if (miniResultCard) miniResultCard.classList.add('hidden');
            console.log("DEBUG: Initial state - all workflow cards hidden except btn-selection");
        }

        manualBtn.addEventListener('click', function() {
            if (workflowLocked) {
                console.log("DEBUG: Workflow locked due to mini_result card, ignoring manual button click");
                return;
            }
            manualWorkflowActive = true;
            manualBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.remove('border-gray-300');
            uploadBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.add('border-gray-300');
            showOnlyCard(recordCard);
            console.log("DEBUG: Manual button clicked, showing record card");
        });

        uploadBtn.addEventListener('click', function() {
            if (workflowLocked) {
                console.log("DEBUG: Workflow locked due to mini_result card, ignoring upload button click");
                return;
            }
            manualWorkflowActive = false;
            uploadBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
            uploadBtn.classList.remove('border-gray-300');
            manualBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
            manualBtn.classList.add('border-gray-300');
            showOnlyCard(uploadContainer);
            console.log("DEBUG: Upload button clicked, showing upload container");
        });

        if (retryUploadBtn) {
            retryUploadBtn.addEventListener('click', function() {
                if (workflowLocked) {
                    console.log("DEBUG: Workflow locked due to mini_result card, ignoring retry upload click");
                    return;
                }
                manualWorkflowActive = false;
                uploadBtn.classList.add('border-blue-500', 'bg-blue-100', 'border-4');
                uploadBtn.classList.remove('border-gray-300');
                manualBtn.classList.remove('border-blue-500', 'bg-blue-100', 'border-4');
                manualBtn.classList.add('border-gray-300');
                showOnlyCard(uploadContainer);
                console.log("DEBUG: Retry upload clicked, showing upload container");
            });
        }
    }

    const observer = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
            if (miniResultCard && !miniResultCard.classList.contains('hidden')) {
                workflowLocked = true;
                showOnlyCard(miniResultCard);
                console.log("DEBUG: Mini result card detected, locking workflow and showing only it");
            } else if (manualWorkflowActive) {
                if (runPredictionCard && !runPredictionCard.classList.contains('hidden')) {
                    showOnlyCard(runPredictionCard);
                    console.log("DEBUG: Run prediction card detected, showing only it");
                } else if (recordCard && !recordCard.classList.contains('hidden')) {
                    showOnlyCard(recordCard);
                    console.log("DEBUG: Record card detected, showing only it");
                }
            } else {
                if (redundancyCard && !redundancyCard.classList.contains('hidden')) {
                    showOnlyCard(redundancyCard);
                    console.log("DEBUG: Redundancy card detected, showing only it");
                } else if (datasetStatsCard && !datasetStatsCard.classList.contains('hidden')) {
                    showOnlyCard(datasetStatsCard);
                    console.log("DEBUG: Dataset stats card detected, showing only it");
                } else if (uploadErrorCard && !uploadErrorCard.classList.contains('hidden')) {
                    showOnlyCard(uploadErrorCard);
                    console.log("DEBUG: Upload error card detected, showing only it");
                } else if (uploadContainer && !uploadContainer.classList.contains('hidden')) {
                    showOnlyCard(uploadContainer);
                    console.log("DEBUG: Upload container detected, showing only it");
                }
            }
        });
    });
    observer.observe(document.querySelector('section') || document.body, { childList: true, subtree: true });

    const churnPieChart = document.getElementById('churnPieChart');
    if (churnPieChart) {
        console.log('DEBUG: Churn pie chart canvas found on page:', window.location.pathname);
        const highChurn = parseFloat(churnPieChart.dataset.highChurn) || 0;
        const moderateChurn = parseFloat(churnPieChart.dataset.moderateChurn) || 0;
        const lowChurn = parseFloat(churnPieChart.dataset.lowChurn) || 0;

        new Chart(churnPieChart, {
            type: 'pie',
            data: {
                labels: ['High Churn', 'Moderate Churn', 'Low Churn'],
                datasets: [{
                    data: [highChurn, moderateChurn, lowChurn],
                    backgroundColor: ['#EF4444', '#FBBF24', '#10B981'],
                    borderColor: ['#EF4444', '#FBBF24', '#10B981'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    title: { display: true, text: 'Churn Risk Distribution' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) label += ': ';
                                label += context.raw + '%';
                                return label;
                            }
                        }
                    }
                }
            }
        });
        console.log('DEBUG: Pie chart rendered successfully');
    }

    const churnBarChartCanvas = document.getElementById('churnBarChart');
    const featureSelect = document.getElementById('featureSelect');
    if (churnBarChartCanvas && featureSelect) {
        const metaTags = {
            transaction_volume: {
                high: stats.transaction_volume_dist.high,
                moderate: stats.transaction_volume_dist.moderate,
                low: stats.transaction_volume_dist.low
            },
            online_usage: {
                high: stats.online_usage_dist.high,
                moderate: stats.online_usage_dist.moderate,
                low: stats.online_usage_dist.low
            },
            complaints: {
                high: stats.complaints_dist.high,
                moderate: stats.complaints_dist.moderate,
                low: stats.complaints_dist.low
            },
            complaints_per_transaction: {
                high: stats.complaints_per_transaction_dist.high,
                moderate: stats.complaints_per_transaction_dist.moderate,
                low: stats.complaints_per_transaction_dist.low
            }
        };

        let barChart = new Chart(churnBarChartCanvas, {
            type: 'bar',
            data: {
                labels: ['High', 'Moderate', 'Low'],
                datasets: [{
                    label: 'Distribution',
                    data: [metaTags.transaction_volume.high, metaTags.transaction_volume.moderate, metaTags.transaction_volume.low],
                    backgroundColor: ['#EF4444', '#FBBF24', '#10B981']
                }]
            },
            options: {
                scales: {
                    x: { stacked: true },
                    y: { stacked: true, beginAtZero: true, max: 100, title: { display: true, text: 'Percentage (%)' } }
                },
                plugins: { legend: { display: false } }
            }
        });

        featureSelect.addEventListener('change', function() {
            const feature = featureSelect.value;
            barChart.data.datasets[0].data = [
                metaTags[feature].high,
                metaTags[feature].moderate,
                metaTags[feature].low
            ];
            barChart.update();
            console.log(`DEBUG: Bar chart updated for feature: ${feature}`);
        });
        console.log('DEBUG: Stacked bar chart initialized');
    }

    const riskFilter = document.getElementById('riskFilter');
    const customerSearch = document.getElementById('customerSearch');
    const churnTableBody = document.getElementById('churnTableBody');
    const downloadFilteredCsv = document.getElementById('downloadFilteredCsv');
    const rows = Array.from(churnTableBody?.querySelectorAll('tr') || []);

    function applyFilters() {
        const riskValue = riskFilter.value;
        const searchValue = customerSearch.value.toLowerCase();

        rows.forEach(row => {
            const riskCategory = row.dataset.risk;
            const customerId = row.dataset.customerId.toLowerCase();

            const matchesRisk = riskValue === 'all' || riskCategory.toLowerCase().includes(riskValue);
            const matchesSearch = !searchValue || customerId.includes(searchValue);

            row.style.display = matchesRisk && matchesSearch ? '' : 'none';
        });

        const baseUrl = downloadFilteredCsv.getAttribute('href').split('?')[0];
        const params = new URLSearchParams();
        params.set('risk_filter', riskValue);
        if (searchValue) params.set('customer_id', searchValue);
        downloadFilteredCsv.setAttribute('href', `${baseUrl}?${params.toString()}`);
        console.log(`DEBUG: Updated download URL: ${downloadFilteredCsv.href}`);
    }

    if (riskFilter && customerSearch && churnTableBody && downloadFilteredCsv) {
        console.log('DEBUG: Table filtering elements found');
        riskFilter.addEventListener('change', applyFilters);
        customerSearch.addEventListener('input', applyFilters);
        applyFilters();
    }

    // Updated PDF Export Logic for Multi-Page Support
    const exportPdfBtn = document.getElementById('export-pdf-btn');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', function() {
            const element = document.getElementById('dashboard-content');
            const dateElement = document.getElementById('dashboard-date');
            const rawDate = dateElement ? dateElement.textContent : 'unknown_date';
            const date = rawDate.replace('Dashboard Created: ', '').trim();

            rows.forEach(row => row.style.display = ''); // Show all rows for export

            html2canvas(element, {
                scale: 2, // Higher resolution for better quality
                useCORS: true,
                logging: true,
                width: element.scrollWidth, // Capture full width
                height: element.scrollHeight // Capture full height
            }).then(canvas => {
                const imgData = canvas.toDataURL('image/jpeg', 0.98);
                const { jsPDF } = window.jspdf;
                const pdf = new jsPDF({
                    orientation: 'portrait',
                    unit: 'in',
                    format: 'letter'
                });

                const imgProps = pdf.getImageProperties(imgData);
                const pdfWidth = pdf.internal.pageSize.getWidth();
                const pdfPageHeight = pdf.internal.pageSize.getHeight();
                const imgWidth = imgProps.width;
                const imgHeight = imgProps.height;
                const widthRatio = pdfWidth / imgWidth;
                const scaledImgHeight = imgHeight * widthRatio;

                let heightLeft = scaledImgHeight;
                let position = 0;

                // Add first page
                pdf.addImage(imgData, 'JPEG', 0, position, pdfWidth, scaledImgHeight);
                heightLeft -= pdfPageHeight;

                // Add additional pages if content exceeds one page
                while (heightLeft > 0) {
                    position -= pdfPageHeight;
                    pdf.addPage();
                    pdf.addImage(imgData, 'JPEG', 0, position, pdfWidth, scaledImgHeight);
                    heightLeft -= pdfPageHeight;
                }

                const filename = `Churn_Dashboard_${date.replace(/[: ]/g, '_')}.pdf`;
                pdf.save(filename);
                console.log(`DEBUG: Dashboard exported to PDF as "${filename}" with ${pdf.internal.getNumberOfPages()} pages`);
                applyFilters(); // Reapply filters after export
            }).catch(err => {
                console.error('DEBUG: Error exporting to PDF:', err.message, err.stack);
                alert('Failed to export PDF. Check console for details.');
                applyFilters();
            });
        });
        console.log('DEBUG: Export PDF button initialized with multi-page support');
    }

    setActiveLink();
});

function setActiveLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (linkPath === currentPath) {
            link.classList.add('border-blue-500', 'text-black');
            link.classList.remove('border-transparent');
        } else {
            link.classList.remove('border-blue-500', 'text-black');
            link.classList.add('border-transparent');
        }
    });
}

function openNav() {
    const sidenav = document.getElementById('mySidenav');
    if (sidenav) {
        sidenav.style.width = '250px';
        document.body.style.backgroundColor = 'rgba(0,0,0,0.4)';
    } else {
        console.warn('Sidenav element not found.');
    }
}

function closeNav() {
    const sidenav = document.getElementById('mySidenav');
    if (sidenav) {
        sidenav.style.width = '0';
        document.body.style.backgroundColor = 'white';
    } else {
        console.warn('Sidenav element not found.');
    }
}