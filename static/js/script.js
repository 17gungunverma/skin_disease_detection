// Global state
let selectedFile = null;
let predictionResults = null;
let modelComparison = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const predictBtn = document.getElementById('predictBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadModelInfo();
    checkModelsStatus();
});

function initializeEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });
    
    // Predict button
    predictBtn.addEventListener('click', handlePrediction);
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

function handleFileSelect() {
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please select a valid image file.');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    
    clearError();
}

async function handlePrediction() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }
    
    showLoading(true);
    resultsSection.classList.add('hidden');
    clearError();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }
        
        const data = await response.json();
        predictionResults = data;
        
        displayResults(data);
        resultsSection.classList.remove('hidden');
        
    } catch (error) {
        console.error('[v0] Prediction error:', error);
        showError(error.message || 'An error occurred during prediction.');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    if (!data.predictions) return;
    
    // Display predictions
    const predictionsContainer = document.getElementById('predictionsContainer');
    predictionsContainer.innerHTML = '';
    
    for (const [modelName, prediction] of Object.entries(data.predictions)) {
        if (prediction.error) {
            predictionsContainer.innerHTML += `
                <div class="prediction-card">
                    <h3>${formatModelName(modelName)}</h3>
                    <div class="prediction-result">
                        <div class="prediction-label">Error</div>
                        <p style="color: #dc2626;">${prediction.error}</p>
                    </div>
                </div>
            `;
            continue;
        }
        
        const confidence = (prediction.confidence * 100).toFixed(2);
        
        predictionsContainer.innerHTML += `
            <div class="prediction-card">
                <h3>${formatModelName(modelName)}</h3>
                <div class="prediction-result">
                    <div class="prediction-label">Predicted Disease</div>
                    <div class="prediction-value">${prediction.class_name}</div>
                </div>
                <div class="prediction-result">
                    <div class="prediction-label">Confidence</div>
                    <div style="font-size: 1.1rem; color: #059669; font-weight: bold;">
                        ${confidence}%
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Display detailed probabilities
    displayDetailedAnalysis(data.predictions);
    
    // Display model comparison if available
    if (modelComparison) {
        displayModelComparison();
    }
}

function displayDetailedAnalysis(predictions) {
    const detailsContainer = document.getElementById('detailsContainer');
    detailsContainer.innerHTML = '';
    
    for (const [modelName, prediction] of Object.entries(predictions)) {
        if (prediction.error) continue;
        
        let probabilitiesHtml = '<div style="margin-top: 1rem;">';
        const sortedProbs = Object.entries(prediction.all_probabilities)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);
        
        for (const [disease, prob] of sortedProbs) {
            const percentage = (prob * 100).toFixed(1);
            probabilitiesHtml += `
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span>${disease}</span>
                        <span style="font-weight: bold;">${percentage}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        }
        probabilitiesHtml += '</div>';
        
        detailsContainer.innerHTML += `
            <div class="detail-card">
                <h4>${formatModelName(modelName)}</h4>
                <p style="color: #6b7280; margin-bottom: 1rem;">Top predictions with probabilities</p>
                ${probabilitiesHtml}
            </div>
        `;
    }
}

function displayModelComparison() {
    const comparisonContainer = document.getElementById('comparisonContainer');
    
    if (!modelComparison || Object.keys(modelComparison).length === 0) {
        comparisonContainer.innerHTML = '<p>No model comparison data available.</p>';
        return;
    }
    
    let tableHtml = `
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    for (const [model, metrics] of Object.entries(modelComparison)) {
        tableHtml += `
            <tr>
                <td><strong>${formatModelName(model)}</strong></td>
                <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
                <td>${(metrics.precision * 100).toFixed(2)}%</td>
                <td>${(metrics.recall * 100).toFixed(2)}%</td>
                <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
            </tr>
        `;
    }
    
    tableHtml += `
            </tbody>
        </table>
    `;
    
    comparisonContainer.innerHTML = tableHtml;
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        if (!response.ok) throw new Error('Failed to load model info');
        
        const data = await response.json();
        modelComparison = data.model_info;
        
        // Populate disease list
        const diseaseList = document.getElementById('disease-list');
        if (data.disease_classes) {
            Object.values(data.disease_classes).forEach(disease => {
                const li = document.createElement('li');
                li.textContent = disease;
                diseaseList.appendChild(li);
            });
        }
        
    } catch (error) {
        console.error('[v0] Error loading model info:', error);
    }
}

async function checkModelsStatus() {
    try {
        const response = await fetch('/api/models-status');
        if (!response.ok) throw new Error('Failed to check models status');
        
        const data = await response.json();
        const statusElement = document.getElementById('model-status');
        
        if (data.loaded_models && data.loaded_models.length > 0) {
            statusElement.textContent = `✓ ${data.loaded_models.length} model(s) loaded`;
            statusElement.classList.add('ready');
        } else {
            statusElement.textContent = '✗ No models loaded. Run train_models.py first.';
            statusElement.classList.add('error');
        }
        
    } catch (error) {
        console.error('[v0] Error checking models status:', error);
        document.getElementById('model-status').textContent = '✗ Unable to connect to server';
        document.getElementById('model-status').classList.add('error');
    }
}

function showLoading(show) {
    if (show) {
        loadingSection.classList.remove('hidden');
    } else {
        loadingSection.classList.add('hidden');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

function clearError() {
    errorSection.classList.add('hidden');
    errorMessage.textContent = '';
}

function formatModelName(name) {
    return name
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
