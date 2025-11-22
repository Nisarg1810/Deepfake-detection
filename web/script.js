// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const videoInput = document.getElementById('videoInput');
const videoPreview = document.getElementById('videoPreview');
const previewVideo = document.getElementById('previewVideo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const loadingStatus = document.getElementById('loadingStatus');
const progressFill = document.getElementById('progressFill');
const resultsSection = document.getElementById('resultsSection');
const resetBtn = document.getElementById('resetBtn');

let selectedFile = null;

// Upload Box Click
uploadBox.addEventListener('click', () => {
    videoInput.click();
});

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File Input Change
videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a valid video file');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    
    uploadBox.style.display = 'none';
    videoPreview.style.display = 'block';
}

// Analyze Button Click
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Hide preview, show loading
    videoPreview.style.display = 'none';
    loadingSection.style.display = 'block';
    
    // Simulate analysis steps
    await simulateAnalysis();
});

// Simulate Analysis Process
async function simulateAnalysis() {
    const steps = [
        { status: 'Uploading video...', progress: 10 },
        { status: 'Extracting frames...', progress: 25 },
        { status: 'Detecting faces...', progress: 40 },
        { status: 'Running CNN analysis...', progress: 55 },
        { status: 'Analyzing frequency patterns...', progress: 70 },
        { status: 'Checking temporal consistency...', progress: 85 },
        { status: 'Computing final verdict...', progress: 100 }
    ];
    
    for (const step of steps) {
        loadingStatus.textContent = step.status;
        progressFill.style.width = step.progress + '%';
        await sleep(800);
    }
    
    // Upload and analyze
    await uploadAndAnalyze();
}

// Upload and Analyze Video
async function uploadAndAnalyze() {
    const formData = new FormData();
    formData.append('video', selectedFile);
    
    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing video. Make sure the Flask server is running.');
        reset();
    }
}

// Display Results
function displayResults(result) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    const verdict = result.verdict;
    const aggregation = result.aggregation;
    
    // Verdict Card
    const verdictCard = document.getElementById('verdictCard');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictLabel = document.getElementById('verdictLabel');
    const scoreValue = document.getElementById('scoreValue');
    
    // Determine verdict type
    let verdictType = 'uncertain';
    let icon = '⚠️';
    
    if (verdict.final_label.includes('AUTHENTIC')) {
        verdictType = 'authentic';
        icon = '✓';
        verdictLabel.textContent = 'Likely Authentic';
    } else if (verdict.final_label.includes('MANIPULATED')) {
        verdictType = 'fake';
        icon = '✗';
        verdictLabel.textContent = verdict.final_label.includes('LIKELY') ? 'Likely Fake' : 'Possibly Fake';
    } else {
        verdictLabel.textContent = 'Uncertain';
    }
    
    verdictCard.className = `verdict-card ${verdictType}`;
    verdictIcon.className = `verdict-icon ${verdictType}`;
    verdictIcon.textContent = icon;
    scoreValue.textContent = (verdict.final_score * 100).toFixed(1) + '%';
    
    // Metrics
    updateMetric('cnn', aggregation.max_score);
    updateMetric('freq', aggregation.frequency_score);
    updateMetric('temp', aggregation.temporal_max);
    updateMetric('lip', aggregation.lip_sync_score || 0.5);
    
    // Details
    document.getElementById('framesCount').textContent = result.frames || 'N/A';
    document.getElementById('facesCount').textContent = result.faces || 'N/A';
    document.getElementById('processingTime').textContent = '~10s';
}

// Update Metric
function updateMetric(type, value) {
    const scoreEl = document.getElementById(`${type}Score`);
    const barEl = document.getElementById(`${type}Bar`);
    
    scoreEl.textContent = (value * 100).toFixed(1) + '%';
    
    setTimeout(() => {
        barEl.style.width = (value * 100) + '%';
    }, 100);
}

// Reset Button
resetBtn.addEventListener('click', reset);

// Reset Function
function reset() {
    selectedFile = null;
    videoInput.value = '';
    previewVideo.src = '';
    
    uploadBox.style.display = 'block';
    videoPreview.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    progressFill.style.width = '0%';
}

// Utility: Sleep
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
