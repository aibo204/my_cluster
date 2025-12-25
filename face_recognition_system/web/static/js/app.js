/**
 * DGFC Face Recognition System - Frontend JavaScript
 */

// API Base URL
const API_BASE = '/api';

// State
const state = {
    currentPage: 'dashboard',
    cameraStream: null,
    isRealtimeRunning: false,
    clusterImages: [],
    compareImages: [null, null],
    personPhotos: [],
};

// ==================== Utility Functions ====================

function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

function imageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ==================== Navigation ====================

function showPage(pageName) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === pageName) {
            item.classList.add('active');
        }
    });
    
    // Update pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageName).classList.add('active');
    
    state.currentPage = pageName;
    
    // Page-specific initialization
    if (pageName === 'dashboard') {
        loadStatistics();
    } else if (pageName === 'persons') {
        loadPersons();
    }
}

// Initialize navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        showPage(item.dataset.page);
    });
});

// ==================== Dashboard ====================

async function loadStatistics() {
    try {
        const data = await apiCall('/stats');
        if (data.success) {
            const stats = data.statistics;
            document.getElementById('stat-persons').textContent = stats.num_persons || 0;
            document.getElementById('stat-faces').textContent = stats.num_faces || 0;
            document.getElementById('stat-index').textContent = stats.index_size || 0;
            document.getElementById('stat-unassigned').textContent = stats.num_unassigned_faces || 0;
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// ==================== Face Recognition ====================

const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const resultPanel = document.getElementById('result-panel');

uploadArea.addEventListener('click', () => imageInput.click());

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
        processRecognitionImage(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        processRecognitionImage(e.target.files[0]);
    }
});

async function processRecognitionImage(file) {
    showLoading();
    
    try {
        const base64 = await imageToBase64(file);
        
        const data = await apiCall('/recognize', {
            method: 'POST',
            body: JSON.stringify({ image: base64 }),
        });
        
        if (data.success) {
            displayRecognitionResults(data);
        }
    } catch (error) {
        showToast('识别失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayRecognitionResults(data) {
    uploadArea.style.display = 'none';
    resultPanel.style.display = 'grid';
    
    // Display annotated image
    document.getElementById('result-image').src = 'data:image/jpeg;base64,' + data.annotated_image;
    
    // Display face results
    const resultsContainer = document.getElementById('face-results');
    resultsContainer.innerHTML = '';
    
    if (data.results.length === 0) {
        resultsContainer.innerHTML = '<p class="text-muted">未检测到人脸</p>';
        return;
    }
    
    data.results.forEach((result, idx) => {
        const item = document.createElement('div');
        item.className = `face-result-item ${result.is_known ? 'known' : 'unknown'}`;
        
        item.innerHTML = `
            <div class="face-name">${result.person_name || 'Unknown'}</div>
            <div class="face-confidence">
                置信度: ${(result.recognition_confidence * 100).toFixed(1)}%
            </div>
        `;
        
        resultsContainer.appendChild(item);
    });
    
    showToast(`检测到 ${data.num_faces} 张人脸`, 'success');
}

// ==================== Real-time Recognition ====================

const videoStream = document.getElementById('video-stream');
const videoCanvas = document.getElementById('video-canvas');
const videoOverlay = document.getElementById('video-overlay');
const btnStartCamera = document.getElementById('btn-start-camera');
const btnCapture = document.getElementById('btn-capture');
const btnToggleRealtime = document.getElementById('btn-toggle-realtime');

btnStartCamera.addEventListener('click', startCamera);
btnCapture.addEventListener('click', captureAndRecognize);
btnToggleRealtime.addEventListener('click', toggleRealtime);

async function startCamera() {
    try {
        state.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 1280, height: 720 },
            audio: false,
        });
        
        videoStream.srcObject = state.cameraStream;
        videoOverlay.classList.add('hidden');
        showToast('摄像头已启动', 'success');
    } catch (error) {
        showToast('无法访问摄像头: ' + error.message, 'error');
    }
}

async function captureAndRecognize() {
    if (!state.cameraStream) {
        showToast('请先启动摄像头', 'error');
        return;
    }
    
    // Capture frame
    videoCanvas.width = videoStream.videoWidth;
    videoCanvas.height = videoStream.videoHeight;
    const ctx = videoCanvas.getContext('2d');
    ctx.drawImage(videoStream, 0, 0);
    
    const base64 = videoCanvas.toDataURL('image/jpeg');
    
    showLoading();
    
    try {
        const data = await apiCall('/recognize', {
            method: 'POST',
            body: JSON.stringify({ image: base64 }),
        });
        
        if (data.success) {
            displayRealtimeResults(data.results);
        }
    } catch (error) {
        showToast('识别失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayRealtimeResults(results) {
    const container = document.getElementById('realtime-results');
    container.innerHTML = '';
    
    results.forEach(result => {
        const item = document.createElement('div');
        item.className = `face-result-item ${result.is_known ? 'known' : 'unknown'}`;
        item.innerHTML = `
            <strong>${result.person_name || 'Unknown'}</strong>
            <span>置信度: ${(result.recognition_confidence * 100).toFixed(1)}%</span>
        `;
        container.appendChild(item);
    });
}

let realtimeInterval = null;

function toggleRealtime() {
    if (state.isRealtimeRunning) {
        stopRealtime();
    } else {
        startRealtime();
    }
}

function startRealtime() {
    if (!state.cameraStream) {
        showToast('请先启动摄像头', 'error');
        return;
    }
    
    state.isRealtimeRunning = true;
    btnToggleRealtime.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="6" y="4" width="4" height="16"/>
            <rect x="14" y="4" width="4" height="16"/>
        </svg>
        停止实时识别
    `;
    
    realtimeInterval = setInterval(async () => {
        if (!state.isRealtimeRunning) return;
        
        videoCanvas.width = videoStream.videoWidth;
        videoCanvas.height = videoStream.videoHeight;
        const ctx = videoCanvas.getContext('2d');
        ctx.drawImage(videoStream, 0, 0);
        
        const base64 = videoCanvas.toDataURL('image/jpeg');
        
        try {
            const data = await apiCall('/recognize', {
                method: 'POST',
                body: JSON.stringify({ image: base64 }),
            });
            
            if (data.success) {
                displayRealtimeResults(data.results);
            }
        } catch (error) {
            console.error('Realtime recognition error:', error);
        }
    }, 500);
}

function stopRealtime() {
    state.isRealtimeRunning = false;
    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }
    
    btnToggleRealtime.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        开始实时识别
    `;
}

// ==================== Face Clustering ====================

const clusterUploadZone = document.getElementById('cluster-upload-zone');
const clusterImagesInput = document.getElementById('cluster-images-input');
const selectedImagesContainer = document.getElementById('selected-images');
const btnStartCluster = document.getElementById('btn-start-cluster');

clusterUploadZone.addEventListener('click', () => clusterImagesInput.click());

clusterUploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    clusterUploadZone.style.borderColor = 'var(--accent-primary)';
});

clusterUploadZone.addEventListener('dragleave', () => {
    clusterUploadZone.style.borderColor = '';
});

clusterUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    clusterUploadZone.style.borderColor = '';
    handleClusterFiles(e.dataTransfer.files);
});

clusterImagesInput.addEventListener('change', (e) => {
    handleClusterFiles(e.target.files);
});

async function handleClusterFiles(files) {
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const base64 = await imageToBase64(file);
            state.clusterImages.push(base64);
            
            const thumb = document.createElement('div');
            thumb.className = 'selected-image-thumb';
            thumb.innerHTML = `<img src="${base64}" alt="Selected">`;
            selectedImagesContainer.appendChild(thumb);
        }
    }
    
    btnStartCluster.disabled = state.clusterImages.length < 2;
}

btnStartCluster.addEventListener('click', async () => {
    if (state.clusterImages.length < 2) {
        showToast('请至少上传2张图片', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const autoRegister = document.getElementById('auto-register').checked;
        
        const data = await apiCall('/cluster', {
            method: 'POST',
            body: JSON.stringify({
                images: state.clusterImages,
                auto_register: autoRegister,
            }),
        });
        
        if (data.success) {
            displayClusterResults(data);
        }
    } catch (error) {
        showToast('聚类失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
});

function displayClusterResults(data) {
    const resultsContainer = document.getElementById('cluster-results');
    resultsContainer.style.display = 'block';
    
    document.getElementById('cluster-count').textContent = data.num_clusters;
    document.getElementById('face-count').textContent = data.num_faces;
    
    const grid = document.getElementById('clusters-grid');
    grid.innerHTML = '';
    
    data.clusters.forEach(cluster => {
        const item = document.createElement('div');
        item.className = 'cluster-item';
        
        let faceHtml = '';
        if (cluster.representative_face) {
            faceHtml = `<img src="data:image/jpeg;base64,${cluster.representative_face}" alt="Cluster ${cluster.cluster_id}">`;
        } else {
            faceHtml = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="8" r="5"/>
                <path d="M20 21a8 8 0 10-16 0"/>
            </svg>`;
        }
        
        item.innerHTML = `
            <div class="cluster-face">${faceHtml}</div>
            <div class="cluster-info">
                <strong>身份 #${cluster.cluster_id}</strong>
                <span>${cluster.num_faces} 张人脸</span>
            </div>
        `;
        
        grid.appendChild(item);
    });
    
    showToast(`聚类完成，发现 ${data.num_clusters} 个身份`, 'success');
}

// ==================== Person Management ====================

const btnAddPerson = document.getElementById('btn-add-person');
const modalAddPerson = document.getElementById('modal-add-person');
const personPhotosUpload = document.getElementById('person-photos-upload');
const personPhotosInput = document.getElementById('person-photos-input');
const personPhotosPreview = document.getElementById('person-photos-preview');
const btnSavePerson = document.getElementById('btn-save-person');

btnAddPerson.addEventListener('click', () => {
    modalAddPerson.classList.add('active');
    state.personPhotos = [];
    document.getElementById('person-name').value = '';
    personPhotosPreview.innerHTML = '';
});

modalAddPerson.querySelectorAll('.modal-close, .modal-cancel').forEach(el => {
    el.addEventListener('click', () => {
        modalAddPerson.classList.remove('active');
    });
});

personPhotosUpload.addEventListener('click', () => personPhotosInput.click());

personPhotosInput.addEventListener('change', async (e) => {
    for (const file of e.target.files) {
        if (file.type.startsWith('image/')) {
            const base64 = await imageToBase64(file);
            state.personPhotos.push(base64);
            
            const thumb = document.createElement('div');
            thumb.className = 'selected-image-thumb';
            thumb.innerHTML = `<img src="${base64}" alt="Photo">`;
            personPhotosPreview.appendChild(thumb);
        }
    }
});

btnSavePerson.addEventListener('click', async () => {
    const name = document.getElementById('person-name').value.trim();
    
    if (!name) {
        showToast('请输入姓名', 'error');
        return;
    }
    
    if (state.personPhotos.length === 0) {
        showToast('请添加至少一张照片', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const data = await apiCall('/register', {
            method: 'POST',
            body: JSON.stringify({
                name: name,
                images: state.personPhotos,
            }),
        });
        
        if (data.success) {
            showToast(`已成功注册 ${name}`, 'success');
            modalAddPerson.classList.remove('active');
            loadPersons();
        }
    } catch (error) {
        showToast('注册失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
});

async function loadPersons() {
    try {
        const data = await apiCall('/persons');
        
        if (data.success) {
            const grid = document.getElementById('persons-grid');
            grid.innerHTML = '';
            
            if (data.persons.length === 0) {
                grid.innerHTML = '<p class="text-muted" style="text-align: center; grid-column: 1/-1;">暂无注册人员</p>';
                return;
            }
            
            data.persons.forEach(person => {
                const card = document.createElement('div');
                card.className = 'person-card';
                card.innerHTML = `
                    <div class="person-avatar">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="8" r="5"/>
                            <path d="M20 21a8 8 0 10-16 0"/>
                        </svg>
                    </div>
                    <div class="person-name">${person.name}</div>
                    <div class="person-info">${person.num_faces || 0} 张人脸</div>
                `;
                grid.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Failed to load persons:', error);
    }
}

// Search persons
document.getElementById('search-persons').addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    document.querySelectorAll('.person-card').forEach(card => {
        const name = card.querySelector('.person-name').textContent.toLowerCase();
        card.style.display = name.includes(query) ? 'block' : 'none';
    });
});

// ==================== Face Comparison ====================

const compareBox1 = document.getElementById('compare-box-1');
const compareBox2 = document.getElementById('compare-box-2');
const compareInput1 = document.getElementById('compare-input-1');
const compareInput2 = document.getElementById('compare-input-2');
const compareImg1 = document.getElementById('compare-img-1');
const compareImg2 = document.getElementById('compare-img-2');
const btnCompare = document.getElementById('btn-compare');

compareBox1.addEventListener('click', () => compareInput1.click());
compareBox2.addEventListener('click', () => compareInput2.click());

compareInput1.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        const base64 = await imageToBase64(e.target.files[0]);
        state.compareImages[0] = base64;
        compareImg1.src = base64;
        compareImg1.style.display = 'block';
        compareBox1.querySelector('.compare-placeholder').style.display = 'none';
        updateCompareButton();
    }
});

compareInput2.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        const base64 = await imageToBase64(e.target.files[0]);
        state.compareImages[1] = base64;
        compareImg2.src = base64;
        compareImg2.style.display = 'block';
        compareBox2.querySelector('.compare-placeholder').style.display = 'none';
        updateCompareButton();
    }
});

function updateCompareButton() {
    btnCompare.disabled = !(state.compareImages[0] && state.compareImages[1]);
}

btnCompare.addEventListener('click', async () => {
    showLoading();
    
    try {
        const data = await apiCall('/compare', {
            method: 'POST',
            body: JSON.stringify({
                image1: state.compareImages[0],
                image2: state.compareImages[1],
            }),
        });
        
        if (data.success) {
            displayCompareResult(data);
        }
    } catch (error) {
        showToast('比对失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
});

function displayCompareResult(data) {
    const resultContainer = document.getElementById('compare-result');
    resultContainer.style.display = 'block';
    
    const similarity = data.similarity * 100;
    const progress = document.getElementById('similarity-progress');
    const value = document.getElementById('similarity-value');
    const text = document.getElementById('result-text');
    
    // Animate progress
    setTimeout(() => {
        progress.style.strokeDasharray = `${similarity}, 100`;
    }, 100);
    
    value.textContent = similarity.toFixed(1) + '%';
    
    if (data.is_same_person) {
        text.className = 'result-text match';
        text.textContent = '✓ 判定为同一人';
        document.getElementById('result-indicator').style.color = 'var(--accent-primary)';
    } else {
        text.className = 'result-text no-match';
        text.textContent = '✗ 判定为不同人';
        document.getElementById('result-indicator').style.color = 'var(--danger)';
    }
}

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    // Load initial data
    loadStatistics();
    
    // Health check
    apiCall('/health').then(data => {
        if (data.status === 'ok') {
            document.querySelector('.status-indicator').classList.add('online');
        }
    }).catch(() => {
        document.querySelector('.status-indicator').classList.remove('online');
    });
});

