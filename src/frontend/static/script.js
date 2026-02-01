/**
 * Cogito Frontend JavaScript
 * Handles API interactions and UI updates
 */

// ============================================================================
// Tab Navigation
// ============================================================================

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.dataset.tab;
        document.getElementById(`${tabId}-tab`).classList.add('active');

        // Refresh status when switching to status tab
        if (tabId === 'status') {
            refreshStatus();
        }
    });
});

// ============================================================================
// Loading Overlay
// ============================================================================

function showLoading(text = 'PROCESSING...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
}

// ============================================================================
// Status Updates
// ============================================================================

function showStatus(elementId, message, type = 'info') {
    const statusBox = document.getElementById(elementId);
    statusBox.textContent = message;
    statusBox.className = `status-box ${type}`;
    statusBox.classList.remove('hidden');
}

// ============================================================================
// File Upload
// ============================================================================

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const files = document.getElementById('files').files;
    if (files.length === 0) {
        showStatus('upload-status', 'ERROR: No files selected', 'error');
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    formData.append('collection', document.getElementById('upload-collection').value);
    formData.append('chunking_strategy', document.getElementById('chunking').value);

    showLoading('UPLOADING AND PROCESSING...');

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('upload-status',
                `SUCCESS: Processed ${data.files_processed} files, created ${data.chunks_created} chunks`,
                'success');
        } else {
            showStatus('upload-status', `ERROR: ${data.detail}`, 'error');
        }
    } catch (error) {
        showStatus('upload-status', `ERROR: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
});

// ============================================================================
// URL Ingestion
// ============================================================================

document.getElementById('url-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const url = document.getElementById('url').value;
    if (!url) {
        showStatus('url-status', 'ERROR: No URL provided', 'error');
        return;
    }

    const requestData = {
        url: url,
        collection: document.getElementById('url-collection').value,
        max_pages: parseInt(document.getElementById('max-pages').value)
    };

    showStatus('url-status', 'Starting ingestion...', 'loading');

    try {
        const response = await fetch('/api/ingest-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('url-status',
                `Ingestion started (Job ID: ${data.job_id}). This may take several minutes...`,
                'loading');

            // Poll for status
            pollIngestionStatus(data.job_id);
        } else {
            showStatus('url-status', `ERROR: ${data.detail}`, 'error');
        }
    } catch (error) {
        showStatus('url-status', `ERROR: ${error.message}`, 'error');
    }
});

async function pollIngestionStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/ingestion-status/${jobId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showStatus('url-status',
                    `SUCCESS: Ingested ${data.stats.total_nodes} nodes from ${data.stats.pages_visited} pages`,
                    'success');
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                showStatus('url-status', `FAILED: ${data.error}`, 'error');
            } else {
                showStatus('url-status',
                    `Processing... ${data.message || 'Please wait'}`,
                    'loading');
            }
        } catch (error) {
            clearInterval(pollInterval);
            showStatus('url-status', `ERROR: ${error.message}`, 'error');
        }
    }, 3000);
}

// ============================================================================
// Query
// ============================================================================

document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = document.getElementById('question').value;
    if (!question) {
        return;
    }

    const requestData = {
        question: question,
        collection: document.getElementById('query-collection').value
    };

    // Hide previous results
    document.getElementById('trace-container').classList.add('hidden');
    document.getElementById('answer-container').classList.add('hidden');

    showLoading('EXECUTING QUERY...');

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (response.ok) {
            renderTrace(data.trace);
            renderAnswer(data);
        } else {
            alert(`Error: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
});

function renderTrace(trace) {
    const container = document.getElementById('trace-steps');
    container.innerHTML = '';

    trace.forEach(step => {
        const icon = step.status === 'pass' ? '✓' : (step.status === 'fail' ? '✗' : '...');

        const stepEl = document.createElement('div');
        stepEl.className = `trace-step ${step.status}`;
        stepEl.innerHTML = `
            <span class="trace-icon ${step.status}">[${icon}]</span>
            <span class="trace-action">${step.action}</span>
            <span class="trace-details">— ${step.details}</span>
        `;
        container.appendChild(stepEl);
    });

    document.getElementById('trace-container').classList.remove('hidden');
}

function renderAnswer(data) {
    // Status badge
    const badge = document.getElementById('status-badge');
    if (data.audit_status === 'pass') {
        badge.textContent = 'VERIFIED ✓';
        badge.className = 'verified';
    } else {
        badge.textContent = 'UNVERIFIED ✗';
        badge.className = 'unverified';
    }

    // Answer content
    document.getElementById('answer-content').textContent = data.answer;

    // Sources
    document.getElementById('source-count').textContent =
        `Vector Documents: ${data.sources.vector_docs} | Graph Documents: ${data.sources.graph_docs}`;

    // Retry info
    const retryInfo = document.getElementById('retry-info');
    if (data.retry_count > 0) {
        retryInfo.textContent = `⚡ Self-corrected after ${data.retry_count} retry(s)`;
        retryInfo.classList.remove('hidden');
    } else {
        retryInfo.classList.add('hidden');
    }

    document.getElementById('answer-container').classList.remove('hidden');
}

// ============================================================================
// System Status
// ============================================================================

async function refreshStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const statusContainer = document.getElementById('system-status');
        statusContainer.innerHTML = `
            <div class="status-item">
                <span class="status-label">SYSTEM:</span>
                <span class="status-value ${data.initialized ? '' : 'inactive'}">
                    ${data.initialized ? 'INITIALIZED' : 'NOT INITIALIZED'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">COLLECTION:</span>
                <span class="status-value">${data.collection || 'N/A'}</span>
            </div>
            <div class="status-item">
                <span class="status-label">DOCUMENTS:</span>
                <span class="status-value">${data.document_count}</span>
            </div>
            <div class="status-item">
                <span class="status-label">GRAPH NODES:</span>
                <span class="status-value">${data.graph_nodes}</span>
            </div>
        `;
    } catch (error) {
        document.getElementById('system-status').innerHTML =
            `<p style="color: #FF0000;">Error loading status: ${error.message}</p>`;
    }
}

document.getElementById('refresh-status').addEventListener('click', refreshStatus);

document.getElementById('init-btn').addEventListener('click', async () => {
    const collection = prompt('Enter collection name:', 'cogito_docs');
    if (!collection) return;

    showLoading('INITIALIZING SYSTEM...');

    try {
        const response = await fetch(`/api/initialize?collection=${collection}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            alert('System initialized successfully!');
            refreshStatus();
        } else {
            alert(`Error: ${data.detail}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
});

// ============================================================================
// Initial Load
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    refreshStatus();
});
