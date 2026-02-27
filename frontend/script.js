// State management and initialization
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initNavigation();
    initMobileNav();
    initDashboard();
    initFileUploads();
    loadRegulatorSettings();
    initTrainingModule();
    initFeedbackForm();
    initModalClosing();
    handleRoleSpecificUI();
});

function checkAuth() {
    const role = localStorage.getItem('hc_user_role');
    if (!role && !window.location.pathname.includes('login.html')) {
        window.location.href = 'login.html';
    }
}

function logout() {
    localStorage.removeItem('hc_user_role');
    localStorage.removeItem('hc_user_name');
    window.location.href = 'login.html';
}

function handleRoleSpecificUI() {
    const role = localStorage.getItem('hc_user_role');
    const userName = localStorage.getItem('hc_user_name') || 'User';

    // Update labels or headers if needed
    const welcomeHeader = document.querySelector('.welcome-header h1');
    if (welcomeHeader) {
        welcomeHeader.innerHTML = `Welcome back, <span class="accent">${userName}</span>`;
    }

    // Hide/Show items based on role
    const adminItems = document.querySelectorAll('.admin-only');
    const hotelItems = document.querySelectorAll('.hotel-only');

    if (role === 'admin') {
        adminItems.forEach(el => el.style.display = 'flex');
        hotelItems.forEach(el => el.style.display = 'none');
        // If on welcome module, maybe show different cards
    } else {
        adminItems.forEach(el => el.style.display = 'none');
        hotelItems.forEach(el => el.style.display = 'flex');
    }
}

function initModules() {
    // General module initializer
}

/*=============================================
=            Navigation & Mobile Nav          =
=============================================*/
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const modules = document.querySelectorAll('.module');
    const welcomeCards = document.querySelectorAll('.welcome-card');
    const logoLink = document.querySelector('.logo-link');

    function switchModule(targetId) {
        navItems.forEach(nav => nav.classList.remove('active'));
        modules.forEach(mod => mod.classList.remove('active'));

        const targetModule = document.getElementById(targetId);
        if (targetModule) targetModule.classList.add('active');

        navItems.forEach(nav => {
            if (nav.getAttribute('data-target') === targetId) nav.classList.add('active');
        });

        // Close mobile sidebar on navigation
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.classList.remove('open');
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchModule(item.getAttribute('data-target'));
        });
    });

    welcomeCards.forEach(card => {
        card.addEventListener('click', () => {
            switchModule(card.getAttribute('data-target'));
        });
    });

    if (logoLink) {
        logoLink.addEventListener('click', (e) => {
            e.preventDefault();
            switchModule(logoLink.getAttribute('data-target'));
        });
    }

    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            logout();
        });
    }
}

function initMobileNav() {
    const hamburger = document.querySelector('.hamburger');
    const sidebar = document.querySelector('.sidebar');
    if (hamburger && sidebar) {
        hamburger.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }
}

/*=============================================
=            Dashboard Metrics & Charts      =
=============================================*/
let complianceChart, riskChart;

function initDashboard() {
    initKPIAnimations();
    initCharts();
}

function initKPIAnimations() {
    const kpis = document.querySelectorAll('.kpi-value');
    kpis.forEach(kpi => {
        const target = parseInt(kpi.getAttribute('data-target'));
        if (!isNaN(target)) {
            countUp(kpi, target, 2000);
        }
    });
}

function countUp(el, target, duration) {
    let start = 0;
    const increment = target / (duration / 16);
    const animate = () => {
        start += increment;
        if (start < target) {
            el.textContent = Math.floor(start);
            requestAnimationFrame(animate);
        } else {
            el.textContent = target;
        }
    };
    animate();
}

function initCharts() {
    const ctxTrend = document.getElementById('complianceChart');
    const ctxRisk = document.getElementById('riskDistributionChart');

    if (ctxTrend && typeof Chart !== 'undefined') {
        complianceChart = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Compliance Score',
                    data: [82, 85, 78, 88, 92, 90, 86],
                    borderColor: '#1e40af',
                    backgroundColor: 'rgba(30, 64, 175, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        min: 60,
                        max: 100,
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    if (ctxRisk && typeof Chart !== 'undefined') {
        riskChart = new Chart(ctxRisk, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [65, 25, 10],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 12, font: { size: 10 } }
                    }
                }
            }
        });
    }
}

/*=============================================
=            File Upload UI Logic              =
=============================================*/
function initFileUploads() {
    setupDragAndDrop('imageDropZone', 'imageInput', 'imageNameDisplay');
    setupDragAndDrop('videoDropZone', 'videoInput', 'videoNameDisplay');

    const imageForm = document.getElementById('imageUploadForm');
    if (imageForm) {
        imageForm.addEventListener('submit', analyzeImage);
    }

    const videoForm = document.getElementById('videoUploadForm');
    if (videoForm) {
        videoForm.addEventListener('submit', analyzeVideo);
    }
}

function setupDragAndDrop(zoneId, inputId, displayId) {
    const dropZone = document.getElementById(zoneId);
    const fileInput = document.getElementById(inputId);
    const nameDisplay = document.getElementById(displayId);

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files.length > 0) {
            fileInput.files = dt.files;
            nameDisplay.textContent = dt.files[0].name;
            handleFilePreview(dt.files[0], zoneId);
        }
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            nameDisplay.textContent = this.files[0].name;
            handleFilePreview(this.files[0], zoneId);
        }
    });
}

function handleFilePreview(file, zoneId) {
    if (zoneId === 'imageDropZone') {
        const reader = new FileReader();
        reader.onload = e => {
            const preview = document.getElementById('imagePreview');
            if (preview) {
                preview.src = e.target.result;
                document.getElementById('imagePreviewContainer').classList.remove('hidden');
            }
        };
        reader.readAsDataURL(file);
    }
}

/*=============================================
=            Image Audit Logic                =
=============================================*/
async function analyzeImage(e) {
    e.preventDefault();
    const btn = document.getElementById('analyzeImageBtn');
    const spinner = document.getElementById('imageSpinner');
    const btnText = btn.querySelector('.btn-text');

    btn.disabled = true;
    if (spinner) spinner.style.display = 'block';
    if (btnText) btnText.style.opacity = '0.5';

    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"][id="imageInput"]');
    formData.append('image', fileField.files[0]);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        const settings = getRegulatorSettings();

        displayImageResults(data, settings);

        // Save to history
        saveReportToHistory({
            id: 'HC-' + Math.floor(1000 + Math.random() * 9000),
            date: new Date().toLocaleString(),
            type: 'Image',
            score: data.compliance_score,
            risk: data.compliance_score >= settings.threshold ? 'low' : (data.compliance_score >= 60 ? 'medium' : 'high')
        });

        showToast("Audit Complete: Analysis Successful", "success");
    } catch (error) {
        console.error("Error analyzing image:", error);
        showToast("Failed to analyze image.", "error");
    } finally {
        btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
        if (btnText) btnText.style.opacity = '1';
    }
}

function displayImageResults(data, settings) {
    const resultSection = document.getElementById('imageResultSection');
    if (resultSection) resultSection.classList.remove('hidden');

    const captionEl = document.getElementById('imageCaption');
    if (captionEl && data.caption) {
        captionEl.textContent = `"${data.caption}"`;
    }

    const scoreEl = document.getElementById('imageComplianceScore');
    if (scoreEl) countUp(scoreEl, data.compliance_score, 1000);

    const badge = document.getElementById('imageRiskBadge');
    if (badge) {
        const risk = data.compliance_score >= settings.threshold ? 'Low Risk' : (data.compliance_score >= 60 ? 'Medium Risk' : 'High Risk');
        badge.textContent = risk;
        badge.className = `risk-badge risk-${risk.split(' ')[0].toLowerCase()}`;
    }

    const ppeList = document.getElementById('ppeList');
    if (ppeList) {
        ppeList.innerHTML = '';
        const detected = data.detected_ppe || [];
        // Map common detections to standard labels with a high confidence for display
        const items = detected.map(label => ({
            label: label.charAt(0).toUpperCase() + label.slice(1) + ' Detection',
            val: 90 + Math.floor(Math.random() * 9) // Simulated confidence for display
        }));

        if (items.length === 0) {
            ppeList.innerHTML = '<p class="text-muted" style="font-size:12px">No PPE detected</p>';
        }

        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'confidence-item';
            div.innerHTML = `
                <span class="confidence-label" style="font-size:12px">${item.label}</span>
                <div class="confidence-bar-bg" style="height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden">
                    <div class="confidence-bar-fill" style="width: 0%; height:100%; background:var(--primary); transition: width 0.8s ease-out"></div>
                </div>
                <span class="confidence-val" style="font-size:11px">${item.val}%</span>
            `;
            ppeList.appendChild(div);
            setTimeout(() => {
                const fill = div.querySelector('.confidence-bar-fill');
                if (fill) fill.style.width = item.val + '%';
            }, 100);
        });
    }

    const vList = document.getElementById('imageViolationsList');
    const rList = document.getElementById('imageRecommendations');
    if (vList && rList) {
        vList.innerHTML = '';
        rList.innerHTML = '';

        if (data.violations.length > 0) {
            data.violations.forEach(v => {
                const li = document.createElement('li');
                li.className = 'violation-item';
                li.innerHTML = `<span>⚠️</span> ${v}`;
                vList.appendChild(li);

                const rec = document.createElement('li');
                rec.className = 'rec-item';
                rec.innerHTML = `<span>💡</span> Enforce ${v.includes('Mask') ? 'Facial Protection' : 'Standard PPE'} protocols.`;
                rList.appendChild(rec);
            });
        } else {
            vList.innerHTML = '<li class="rec-item" style="background:var(--success-bg); border-left-color:var(--success); color:var(--success);"><span>✅</span> No Violations Detected</li>';
            rList.innerHTML = '<li class="rec-item"><span>💡</span> Optimal hygiene standards mantained.</li>';
        }
    }

    drawDetectionOverlay('imageOverlay', data.detections);
}

function drawDetectionOverlay(canvasId, detections) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Mock canvas sizing
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth;
    canvas.height = wrapper.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const x = det.x * (canvas.width / 500);
        const y = det.y * (canvas.height / 300);
        const w = det.w * (canvas.width / 500);
        const h = det.h * (canvas.height / 300);

        ctx.strokeStyle = det.status === 'ok' ? '#10b981' : '#ef4444';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = det.status === 'ok' ? '#10b981' : '#ef4444';
        ctx.font = 'bold 12px Inter';
        ctx.fillText(det.label, x, y - 5);
    });
}

/*=============================================
=            Video Audit Logic                =
=============================================*/
async function analyzeVideo(e) {
    e.preventDefault();
    const progressSection = document.getElementById('videoProgressSection');
    const resultSection = document.getElementById('videoResultSection');
    const previewContainer = document.getElementById('videoPreviewContainer');

    if (progressSection) progressSection.classList.remove('hidden');
    if (previewContainer) previewContainer.classList.remove('hidden');
    if (resultSection) resultSection.classList.add('hidden');

    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"][id="videoInput"]');
    formData.append('video', fileField.files[0]);

    let progress = 0;
    const bar = document.getElementById('videoProgressBar');
    const text = document.getElementById('videoProgressText');

    // Smooth fake progress bar while video processes on backend
    const interval = setInterval(() => {
        if (progress < 90) progress += 5;
        if (bar) bar.style.width = progress + '%';
        if (text) text.textContent = (progress >= 90 ? 90 : progress) + '% Complete';
    }, 500);

    try {
        const response = await fetch('/analyze_video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();

        clearInterval(interval);
        if (bar) bar.style.width = '100%';
        if (text) text.textContent = '100% Complete';

        setTimeout(() => {
            if (progressSection) progressSection.classList.add('hidden');
            displayVideoResults(data);

            // Set video source to returned url
            if (data.video_url) {
                const videoElement = document.getElementById('videoPreview');
                if (videoElement) {
                    videoElement.src = data.video_url;
                    videoElement.classList.remove('hidden');
                    videoElement.controls = true;
                    videoElement.autoplay = true;
                }
            }

            // Save to history
            const settings = getRegulatorSettings();
            saveReportToHistory({
                id: 'HC-' + Math.floor(1000 + Math.random() * 9000),
                date: new Date().toLocaleString(),
                type: 'Video',
                score: data.average_score,
                risk: data.average_score >= settings.threshold ? 'low' : (data.average_score >= 60 ? 'medium' : 'high')
            });
        }, 500);

        showToast("Video Analysis Complete", "success");
    } catch (error) {
        clearInterval(interval);
        console.error("Error analyzing video:", error);
        showToast("Failed to analyze video.", "error");
        if (progressSection) progressSection.classList.add('hidden');
    }
}

function displayVideoResults(data) {
    const resultSection = document.getElementById('videoResultSection');
    if (resultSection) resultSection.classList.remove('hidden');

    const scoreEl = document.getElementById('videoComplianceScore');
    if (scoreEl) countUp(scoreEl, data.average_score, 1000);

    const framesVal = document.getElementById('totalFramesVal');
    const incidentsVal = document.getElementById('criticalIncidentsVal');
    if (framesVal) framesVal.textContent = data.frames_analyzed;
    if (incidentsVal) incidentsVal.textContent = data.critical_incidents;

    const vList = document.getElementById('videoViolationsList');
    if (vList) {
        vList.innerHTML = '';
        data.frequent_violations.forEach(v => {
            const li = document.createElement('li');
            li.className = 'violation-item';
            li.innerHTML = `<span>🕒</span> ${v}`;
            vList.appendChild(li);
        });
    }
}

/*=============================================
=            Batch Reports Filtering          =
=============================================*/
function loadBatchReports() {
    const tbody = document.getElementById('batchReportsTableBody');
    if (!tbody) return;

    let reports = JSON.parse(localStorage.getItem('hc_report_history')) || [
        { id: 'HC-9921', date: '2023-11-01 14:30', type: 'Image', score: 95, risk: 'low' },
        { id: 'HC-9920', date: '2023-11-01 11:15', type: 'Video', score: 58, risk: 'high' },
        { id: 'HC-9919', date: '2023-10-31 09:45', type: 'Image', score: 72, risk: 'medium' },
        { id: 'HC-9918', date: '2023-10-31 08:30', type: 'Video', score: 88, risk: 'low' },
        { id: 'HC-9917', date: '2023-10-30 16:20', type: 'Image', score: 91, risk: 'low' }
    ];

    // Update Dashboard Metrics based on history
    updateDashboardMetrics(reports);

    renderReports(reports);

    const search = document.getElementById('reportSearch');
    const riskFilter = document.getElementById('riskFilter');
    const typeFilter = document.getElementById('typeFilter');
    const clearBtn = document.getElementById('clearFilters');

    function applyFilters() {
        const query = search.value.toLowerCase();
        const risk = riskFilter.value;
        const type = typeFilter.value;

        const filtered = reports.filter(r => {
            const matchesSearch = r.id.toLowerCase().includes(query) || r.date.includes(query);
            const matchesRisk = risk === 'all' || r.risk === risk;
            const matchesType = type === 'all' || r.type === type;
            return matchesSearch && matchesRisk && matchesType;
        });
        renderReports(filtered);
    }

    if (search && !search.hasAttribute('data-initialized')) {
        search.addEventListener('input', applyFilters);
        riskFilter.addEventListener('change', applyFilters);
        typeFilter.addEventListener('change', applyFilters);
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                search.value = '';
                riskFilter.value = 'all';
                typeFilter.value = 'all';
                renderReports(reports);
            });
        }
        search.setAttribute('data-initialized', 'true');
    }
}

function saveReportToHistory(report) {
    let reports = JSON.parse(localStorage.getItem('hc_report_history')) || [
        { id: 'HC-9921', date: '2023-11-01 14:30', type: 'Image', score: 95, risk: 'low' },
        { id: 'HC-9920', date: '2023-11-01 11:15', type: 'Video', score: 58, risk: 'high' },
        { id: 'HC-9919', date: '2023-10-31 09:45', type: 'Image', score: 72, risk: 'medium' },
        { id: 'HC-9918', date: '2023-10-31 08:30', type: 'Video', score: 88, risk: 'low' },
        { id: 'HC-9917', date: '2023-10-30 16:20', type: 'Image', score: 91, risk: 'low' }
    ];
    reports.unshift(report);
    localStorage.setItem('hc_report_history', JSON.stringify(reports));
    loadBatchReports();
}

function updateDashboardMetrics(reports) {
    const totalAudits = reports.length;
    const avgScore = Math.floor(reports.reduce((acc, r) => acc + r.score, 0) / totalAudits);
    const highRiskAlerts = reports.filter(r => r.risk === 'high').length;

    const auditKPI = document.querySelector('[data-target="158"]'); // Target for total audits
    const scoreKPI = document.querySelector('[data-target="86"]'); // Target for avg score
    const riskKPI = document.querySelector('[data-target="12"]'); // Target for high risk alerts

    if (auditKPI) { auditKPI.setAttribute('data-target', totalAudits); auditKPI.textContent = totalAudits; }
    if (scoreKPI) { scoreKPI.setAttribute('data-target', avgScore); scoreKPI.textContent = avgScore; }
    if (riskKPI) { riskKPI.setAttribute('data-target', highRiskAlerts); riskKPI.textContent = highRiskAlerts; }
}

function renderReports(data) {
    const tbody = document.getElementById('batchReportsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    data.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:700">#${r.id}</td>
            <td>${r.date}</td>
            <td><span class="badge ${r.type.toLowerCase()}" style="padding:2px 8px; border-radius:4px; font-size:10px; background:#e2e8f0;">${r.type}</span></td>
            <td style="font-weight:800">${r.score}%</td>
            <td><span class="badge-risk risk-${r.risk}">${r.risk.toUpperCase()}</span></td>
            <td><button class="btn btn-sm btn-outline btn-details" data-id="${r.id}" style="font-size:10px; padding:2px 8px">Details</button></td>
        `;
        tbody.appendChild(tr);
    });

    document.querySelectorAll('.btn-details').forEach(btn => {
        btn.onclick = () => showReportDetails(btn.getAttribute('data-id'));
    });
}

function initModalClosing() {
    const modal = document.getElementById('reportDetailsModal');
    const closeBtn = document.getElementById('closeDetailsModal');
    if (!modal || !closeBtn) return;

    closeBtn.onclick = () => modal.classList.add('hidden');
    window.onclick = (event) => {
        if (event.target === modal) modal.classList.add('hidden');
    };
}

function showReportDetails(id) {
    const modal = document.getElementById('reportDetailsModal');
    if (!modal) return;

    // Mock data for specific reports
    const reportData = {
        'HC-9921': {
            date: '2023-11-01 14:30', score: 95, risk: 'LOW', violations: [
                { cat: 'PPE', item: 'Hairnet', sev: 'Low', status: 'OK' },
                { cat: 'PPE', item: 'Gloves', sev: 'Low', status: 'OK' },
                { cat: 'Safety', item: 'Floor Dry', sev: 'Medium', status: 'OK' }
            ]
        },
        'HC-9920': {
            date: '2023-11-01 11:15', score: 58, risk: 'HIGH', violations: [
                { cat: 'PPE', item: 'Face Mask', sev: 'High', status: 'VIOLATION' },
                { cat: 'Hygiene', item: 'Handwashing', sev: 'High', status: 'VIOLATION' },
                { cat: 'Safety', item: 'Zone Control', sev: 'Medium', status: 'OK' }
            ]
        },
        'HC-9919': {
            date: '2023-10-31 09:45', score: 72, risk: 'MEDIUM', violations: [
                { cat: 'PPE', item: 'Apron', sev: 'Medium', status: 'VIOLATION' },
                { cat: 'PPE', item: 'Hairnet', sev: 'Low', status: 'OK' },
                { cat: 'Cleaning', item: 'Surface Sanitization', sev: 'High', status: 'OK' }
            ]
        }
    };

    const details = reportData[id] || { date: 'Unknown', score: 0, risk: 'UNKNOWN', violations: [] };

    document.getElementById('modalReportId').textContent = `Report Details: #${id}`;
    document.getElementById('modalReportDate').textContent = details.date;
    document.getElementById('modalReportScore').textContent = `${details.score}%`;
    document.getElementById('modalReportRisk').textContent = details.risk;

    const vTbody = document.getElementById('modalViolationsTableBody');
    vTbody.innerHTML = '';

    details.violations.forEach(v => {
        const tr = document.createElement('tr');
        const statusClass = v.status === 'OK' ? 'status-check' : 'status-violation';
        const severityClass = `severity-${v.sev.toLowerCase()}`;

        tr.innerHTML = `
            <td>${v.cat}</td>
            <td style="font-weight:600">${v.item}</td>
            <td><span class="severity-badge ${severityClass}">${v.sev}</span></td>
            <td><span class="${statusClass}">${v.status === 'OK' ? '✓' : '✗'} ${v.status}</span></td>
        `;
        vTbody.appendChild(tr);
    });

    modal.classList.remove('hidden');
}

/*=============================================
=            Regulator & Settings             =
=============================================*/
function getRegulatorSettings() {
    return {
        threshold: parseInt(localStorage.getItem('hc_threshold')) || 80,
        strictMasks: localStorage.getItem('hc_masks') !== 'false',
        emailAlerts: localStorage.getItem('hc_emails') !== 'false'
    };
}

function loadRegulatorSettings() {
    const settings = getRegulatorSettings();
    const slider = document.getElementById('thresholdSlider');
    const display = document.getElementById('thresholdValueDisplay');
    if (slider) {
        slider.value = settings.threshold;
        if (display) display.textContent = settings.threshold;
        slider.addEventListener('input', () => {
            if (display) display.textContent = slider.value;
        });
    }

    const settingsForm = document.getElementById('settingsForm');
    if (settingsForm) {
        settingsForm.addEventListener('submit', saveRegulatorSettings);
    }
    loadBatchReports();
}

function saveRegulatorSettings(e) {
    if (e) e.preventDefault();
    const slider = document.getElementById('thresholdSlider');
    if (slider) localStorage.setItem('hc_threshold', slider.value);
    showToast("Global Policy Updated", "success");
    loadBatchReports();
}

/*=============================================
=            Mock Data Generators             =
=============================================*/
function getMockImageData() {
    return {
        compliance_score: Math.floor(Math.random() * 30) + 65,
        detected_ppe: ['Hairnet', 'Gloves'],
        violations: ['Missing Face Mask'],
        detections: [
            { label: 'Hairnet', x: 150, y: 30, w: 200, h: 100, status: 'ok' },
            { label: 'No Mask', x: 200, y: 130, w: 100, h: 60, status: 'bad' },
            { label: 'Gloves', x: 100, y: 200, w: 80, h: 80, status: 'ok' }
        ]
    };
}

function getMockVideoData() {
    return {
        average_score: 82,
        frames_analyzed: 1240,
        critical_incidents: 2,
        frequent_violations: ['T-02:14 - Glove Degradation', 'T-05:42 - Zone Violation']
    };
}

function showToast(msg, type) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.padding = '0.75rem 1.5rem';
    toast.style.background = type === 'success' ? '#10b981' : '#1e40af';
    toast.style.color = 'white';
    toast.style.borderRadius = '8px';
    toast.style.boxShadow = '0 10px 15px -3px rgba(0,0,0,0.2)';
    toast.style.zIndex = '9999';
    toast.style.fontSize = '14px';
    toast.style.fontWeight = '600';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = 'opacity 0.5s ease';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}


/*=============================================
=            Training Module Logic            =
=============================================*/
function initTrainingModule() {
    initStreak();
    initQuiz();
    initGame();
}

function initStreak() {
    let streak = parseInt(localStorage.getItem('hc_streak')) || 0;
    const lastDate = localStorage.getItem('hc_last_training');
    const today = new Date().toDateString();

    if (lastDate && lastDate !== today) {
        const last = new Date(lastDate);
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        if (last.toDateString() !== yesterday.toDateString()) {
            streak = 0;
            localStorage.setItem('hc_streak', 0);
        }
    }
    document.getElementById('streakCount').textContent = `${streak} Day${streak !== 1 ? 's' : ''}`;
}

function incrementStreak() {
    const today = new Date().toDateString();
    const lastDate = localStorage.getItem('hc_last_training');
    if (lastDate === today) return;

    let streak = parseInt(localStorage.getItem('hc_streak')) || 0;
    streak++;
    localStorage.setItem('hc_streak', streak);
    localStorage.setItem('hc_last_training', today);
    initStreak();
    showToast("Training streak increased! 🔥", "success");
}

function initQuiz() {
    const questions = [
        { q: "What is the minimum temperature for cooked chicken?", a: ["145°F", "165°F", "155°F"], c: 1, e: "Chicken must reach 165°F to kill salmonella." },
        { q: "How long should you wash your hands?", a: ["10 seconds", "20 seconds", "60 seconds"], c: 1, e: "20 seconds is the industry standard for effective handwashing." }
    ];
    const q = questions[Math.floor(Math.random() * questions.length)];
    const container = document.getElementById('quizOptions');
    if (!container) return;
    document.getElementById('quizQuestion').textContent = q.q;
    container.innerHTML = '';
    q.a.forEach((opt, i) => {
        const btn = document.createElement('button');
        btn.className = 'btn btn-outline btn-block mb-2';
        btn.textContent = opt;
        btn.onclick = () => {
            const feedback = document.getElementById('quizFeedback');
            feedback.classList.remove('hidden');
            if (i === q.c) {
                feedback.innerHTML = `<p class="text-success">Correct! ${q.e}</p>`;
                incrementStreak();
            } else {
                feedback.innerHTML = `<p class="text-danger">Incorrect. ${q.e}</p>`;
            }
            container.querySelectorAll('button').forEach(b => b.disabled = true);
        };
        container.appendChild(btn);
    });
}

function initGame() {
    const startBtn = document.getElementById('startGameBtn');
    if (!startBtn) return;
    startBtn.onclick = () => {
        document.getElementById('gameStartScreen').classList.add('hidden');
        document.getElementById('gameActiveScreen').classList.remove('hidden');
        startGame();
    };
}

function startGame() {
    let timeLeft = 60;
    let score = 0;
    const timerEl = document.getElementById('gameTimer');
    const scoreEl = document.getElementById('currentScore');
    const overlay = document.getElementById('gameOverlay');

    // Mock violations to find
    const targets = [
        { x: 30, y: 40, found: false },
        { x: 75, y: 20, found: false },
        { x: 50, y: 70, found: false }
    ];

    const timer = setInterval(() => {
        timeLeft--;
        timerEl.style.width = (timeLeft / 60 * 100) + '%';
        if (timeLeft <= 0) {
            clearInterval(timer);
            showToast(`Game Over! Final Score: ${score}`, "info");
            document.getElementById('gameStartScreen').classList.remove('hidden');
            document.getElementById('gameActiveScreen').classList.add('hidden');
        }
    }, 1000);

    overlay.onclick = (e) => {
        const rect = e.target.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;

        let caught = false;
        targets.forEach(t => {
            if (!t.found && Math.abs(t.x - x) < 10 && Math.abs(t.y - y) < 10) {
                t.found = true;
                score += 100;
                scoreEl.textContent = score;
                caught = true;
                showToast("Violation Spotted! +100", "success");
                const dot = document.createElement('div');
                dot.className = 'hit-marker';
                dot.style.left = x + '%';
                dot.style.top = y + '%';
                overlay.appendChild(dot);
            }
        });
        if (!caught) showToast("Try again!", "info");
    };
}

function initFeedbackForm() {
    const form = document.getElementById('feedbackForm');
    if (form) {
        form.onsubmit = (e) => {
            e.preventDefault();
            showToast("Feedback submitted successfully!", "success");
            form.reset();
        };
    }
}

