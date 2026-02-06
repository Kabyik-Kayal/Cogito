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

    // Show progress panel, hide status
    document.getElementById('upload-progress').classList.remove('hidden');
    document.getElementById('upload-status').classList.add('hidden');
    resetUploadProgressPanel();

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.job_id) {
            // Poll for status with enhanced display
            pollUploadStatus(data.job_id);
        } else {
            document.getElementById('upload-progress').classList.add('hidden');
            showStatus('upload-status', `ERROR: ${data.detail || 'Upload failed'}`, 'error');
        }
    } catch (error) {
        document.getElementById('upload-progress').classList.add('hidden');
        showStatus('upload-status', `ERROR: ${error.message}`, 'error');
    }
});

function resetUploadProgressPanel() {
    // Reset step indicators
    document.querySelectorAll('.upload-step').forEach(item => {
        item.classList.remove('active', 'complete');
    });
    // Reset progress bar
    document.getElementById('upload-progress-bar').style.width = '0%';
    document.getElementById('upload-progress-percent').textContent = '0%';
    // Reset counters
    document.getElementById('files-count').textContent = '0';
    document.getElementById('chunks-count').textContent = '0';
    document.getElementById('upload-nodes-count').textContent = '0';
    // Reset status
    document.getElementById('upload-status-msg').textContent = 'Initializing...';
    // Clear activity log
    document.getElementById('upload-activity-log').innerHTML = '';
}

function updateUploadStepIndicators(step) {
    const stepMap = { 'upload': 1, 'chunk': 2, 'graph': 3, 'vectors': 4, 'done': 5 };
    const stepNumber = stepMap[step] || 1;

    document.querySelectorAll('.upload-step').forEach(item => {
        const itemStep = parseInt(item.dataset.step);
        item.classList.remove('active', 'complete');
        if (itemStep < stepNumber) {
            item.classList.add('complete');
        } else if (itemStep === stepNumber) {
            item.classList.add('active');
        }
    });
}

async function pollUploadStatus(jobId) {
    let lastLogCount = 0;

    const updateUI = (data) => {
        // Update progress bar
        const progress = data.progress || 0;
        document.getElementById('upload-progress-bar').style.width = `${progress}%`;
        document.getElementById('upload-progress-percent').textContent = `${progress}%`;

        // Update step indicators
        if (data.step) {
            updateUploadStepIndicators(data.step);
        }

        // Update counters
        if (data.files_processed !== undefined) {
            document.getElementById('files-count').textContent = data.files_processed;
        }
        if (data.chunks_created !== undefined) {
            document.getElementById('chunks-count').textContent = data.chunks_created;
        }
        if (data.nodes_created !== undefined) {
            document.getElementById('upload-nodes-count').textContent = data.nodes_created;
        }

        // Update current status message
        if (data.message) {
            document.getElementById('upload-status-msg').textContent = data.message;
        }

        // Update activity log (only new entries)
        if (data.activity_log && data.activity_log.length > lastLogCount) {
            const newEntries = data.activity_log.slice(lastLogCount);
            const log = document.getElementById('upload-activity-log');
            newEntries.forEach(entry => {
                const entryEl = document.createElement('div');
                entryEl.className = 'activity-entry';
                entryEl.textContent = entry;
                log.appendChild(entryEl);
                log.scrollTop = log.scrollHeight;
            });
            lastLogCount = data.activity_log.length;
        }
    };

    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/upload-status/${jobId}`);
            
            if (response.status === 404) {
                clearInterval(pollInterval);
                return;
            }
            
            const data = await response.json();

            updateUI(data);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                document.getElementById('upload-status-msg').innerHTML =
                    '<span class="success-text">✓ UPLOAD COMPLETE</span>';
                // Refresh collections dropdown
                await loadCollections();
                
                // Auto-hide after 20 seconds
                setTimeout(() => {
                    document.getElementById('upload-progress').classList.add('hidden');
                }, 20000);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                document.getElementById('upload-status-msg').innerHTML =
                    `<span class="error-text">✗ FAILED: ${data.error}</span>`;
            }
        } catch (error) {
            clearInterval(pollInterval);
            document.getElementById('upload-progress').classList.add('hidden');
            showStatus('upload-status', `ERROR: ${error.message}`, 'error');
        }
    }, 500);
}

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

    // Show progress panel, hide status
    document.getElementById('url-progress').classList.remove('hidden');
    document.getElementById('url-status').classList.add('hidden');
    resetProgressPanel();

    try {
        const response = await fetch('/api/ingest-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (response.ok) {
            // Poll for status with enhanced display
            pollIngestionStatus(data.job_id);
        } else {
            document.getElementById('url-progress').classList.add('hidden');
            showStatus('url-status', `ERROR: ${data.detail}`, 'error');
        }
    } catch (error) {
        document.getElementById('url-progress').classList.add('hidden');
        showStatus('url-status', `ERROR: ${error.message}`, 'error');
    }
});

function resetProgressPanel() {
    // Reset step indicators
    document.querySelectorAll('.step-item').forEach(item => {
        item.classList.remove('active', 'complete');
    });
    // Reset progress bar
    document.getElementById('ingestion-progress-bar').style.width = '0%';
    document.getElementById('progress-percent').textContent = '0%';
    // Reset counters
    document.getElementById('pages-count').textContent = '0';
    document.getElementById('nodes-count').textContent = '0';
    // Reset status
    document.getElementById('current-status-msg').textContent = 'Initializing...';
    // Clear activity log
    document.getElementById('activity-log').innerHTML = '';
}

function updateStepIndicators(stepNumber) {
    document.querySelectorAll('.step-item').forEach(item => {
        const step = parseInt(item.dataset.step);
        item.classList.remove('active', 'complete');
        if (step < stepNumber) {
            item.classList.add('complete');
        } else if (step === stepNumber) {
            item.classList.add('active');
        }
    });
}

function addActivityLogEntry(message) {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'activity-entry';
    entry.textContent = message;
    log.appendChild(entry);
    // Auto-scroll to bottom
    log.scrollTop = log.scrollHeight;
}

async function pollIngestionStatus(jobId) {
    let lastLogCount = 0;

    const updateUI = (data) => {
        // Update step indicators
        if (data.step_number !== undefined) {
            updateStepIndicators(data.step_number);
        }

        // Update progress bar
        if (data.progress !== undefined) {
            document.getElementById('ingestion-progress-bar').style.width = `${data.progress}%`;
            document.getElementById('progress-percent').textContent = `${data.progress}%`;
        }

        // Update counters
        if (data.pages_scraped !== undefined) {
            document.getElementById('pages-count').textContent = data.pages_scraped;
        }
        if (data.nodes_created !== undefined) {
            document.getElementById('nodes-count').textContent = data.nodes_created;
        }

        // Update current status message
        if (data.message) {
            document.getElementById('current-status-msg').textContent = data.message;
        }

        // Update activity log (only new entries)
        if (data.activity_log && data.activity_log.length > lastLogCount) {
            const newEntries = data.activity_log.slice(lastLogCount);
            newEntries.forEach(entry => addActivityLogEntry(entry));
            lastLogCount = data.activity_log.length;
        }
    };

    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/ingestion-status/${jobId}`);
            
            if (response.status === 404) {
                clearInterval(pollInterval);
                return;
            }
            
            const data = await response.json();

            updateUI(data);

            if (data.status === 'completed') {
                clearInterval(pollInterval);

                // Final fetch to ensure we have all activity log entries
                const finalResponse = await fetch(`/api/ingestion-status/${jobId}`);
                const finalData = await finalResponse.json();
                updateUI(finalData);

                updateStepIndicators(5); // All steps complete
                document.getElementById('current-status-msg').innerHTML =
                    '<span class="success-text">✓ INGESTION COMPLETE</span>';
                // Refresh collections dropdown to show new collection
                await loadCollections();
                // Keep progress panel visible to show final stats
                
                // Auto-hide after 20 seconds
                setTimeout(() => {
                    document.getElementById('url-progress').classList.add('hidden');
                }, 20000);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                document.getElementById('current-status-msg').innerHTML =
                    `<span class="error-text">✗ FAILED: ${data.error}</span>`;
            }
        } catch (error) {
            clearInterval(pollInterval);
            document.getElementById('url-progress').classList.add('hidden');
            showStatus('url-status', `ERROR: ${error.message}`, 'error');
        }
    }, 500);  // Poll every 500ms for more responsive updates
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

    // UI Reset
    document.getElementById('trace-container').classList.add('hidden');
    document.getElementById('answer-container').classList.add('hidden');
    document.getElementById('query-progress').classList.remove('hidden');
    resetQueryProgress();

    // Hide default loading spinner since we have custom progress now
    // showLoading('EXECUTING QUERY...'); 

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (response.ok) {
            pollQueryStatus(data.job_id);
        } else {
            alert(`Error: ${data.detail}`);
            document.getElementById('query-progress').classList.add('hidden');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        document.getElementById('query-progress').classList.add('hidden');
    }
});

function resetQueryProgress() {
    document.querySelectorAll('.query-step').forEach(s => s.classList.remove('active', 'complete'));
    document.getElementById('query-activity-log').innerHTML = '';
    document.getElementById('query-status-msg').textContent = 'Initializing...';
}

function pollQueryStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/query-status/${jobId}`);
            if (!res.ok) {
                clearInterval(pollInterval);
                return;
            }
            const job = await res.json();

            updateQueryProgress(job);

            if (job.status === 'completed') {
                clearInterval(pollInterval);
                // Show final answer
                renderAnswer(job.response);

                // Keep progress panel visible (User request)
                // Do NOT hide 'query-progress'
                // Do NOT show 'trace-container' (User said "remove traceback")

                document.getElementById('answer-container').classList.remove('hidden');
                
                // Auto-scroll to bottom to show the final answer
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

            } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                alert('Query Failed: ' + (job.error || 'Unknown error'));
            }
        } catch (e) {
            console.error("Polling error", e);
            clearInterval(pollInterval);
        }
    }, 500);
}

function updateQueryProgress(job) {
    // Update logs
    const logContainer = document.getElementById('query-activity-log');
    // We only append new logs? Or assume full replacement is fast enough?
    // job.logs is a list of strings.
    // For simplicity, verify length and append ONLY new ones if possible.
    // But replacing innerHTML is safer to avoid duplicates if polling overlaps.
    // Optimization: check existing child count.

    // Simple approach: Clear and rebuild. It's text, it's fast.
    const currentLines = logContainer.children.length;
    if (job.logs.length > currentLines) {
        for (let i = currentLines; i < job.logs.length; i++) {
            const div = document.createElement('div');
            div.textContent = `> ${job.logs[i]}`;
            logContainer.appendChild(div);
        }
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    // Steps logic
    const steps = ['RETRIEVE', 'GRAPH', 'GENERATE', 'AUDIT', 'DONE'];
    const currentStep = job.step;
    const currentIdx = steps.indexOf(currentStep);

    document.querySelectorAll('.query-step').forEach(el => {
        const stepName = el.dataset.step;
        const stepIdx = steps.indexOf(stepName);

        el.classList.remove('active', 'complete');

        if (currentIdx === -1) {
            // Unknown step (START etc)
            return;
        }

        if (stepIdx < currentIdx) {
            el.classList.add('complete');
        } else if (stepIdx === currentIdx) {
            el.classList.add('active');
        }
    });

    document.getElementById('query-status-msg').textContent = `Status: ${job.step} - ${job.logs[job.logs.length - 1] || ''}`;
}

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
// System Status & Collections
// ============================================================================

async function loadCollections() {
    try {
        const response = await fetch('/api/collections');
        const data = await response.json();

        // Update selectors that should show available collections
        const selectors = ['collection-select', 'query-collection'];

        selectors.forEach(id => {
            const select = document.getElementById(id);
            if (!select) return;

            select.innerHTML = '';

            if (data.collections.length === 0) {
                select.innerHTML = '<option value="">No collections found</option>';
            } else {
                data.collections.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col.name;

                    // Show count only for the main status selector
                    if (id === 'collection-select') {
                        option.textContent = `${col.name} (${col.count} docs)`;
                    } else {
                        option.textContent = col.name;
                    }

                    if (col.name === data.current) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
            }

            // Refresh custom dropdown UI if it exists
            if (typeof refreshCustomDropdown === 'function') {
                refreshCustomDropdown(id);
            }
        });

        // Also populate the datalist for combo-box inputs (ingestion forms)
        const datalist = document.getElementById('collections-list');
        if (datalist) {
            datalist.innerHTML = '';
            data.collections.forEach(col => {
                const option = document.createElement('option');
                option.value = col.name;
                datalist.appendChild(option);
            });
        }

        // Refresh ComboBoxes to reflect new data
        if (typeof refreshCustomComboBox === 'function') {
            refreshCustomComboBox('upload-collection', 'collections-list');
            refreshCustomComboBox('url-collection', 'collections-list');
        }

        // Set default value for ingestion inputs if empty
        const ingestionInputs = ['upload-collection', 'url-collection'];
        ingestionInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input && !input.value && data.current) {
                input.value = data.current;
            }
        });

    } catch (error) {
        console.error('Error loading collections:', error);
        const statusSelect = document.getElementById('collection-select');
        if (statusSelect) statusSelect.innerHTML = '<option value="">Error loading</option>';
    }
}

async function refreshStatus() {
    try {
        const selectedCollection = document.getElementById('collection-select').value;
        const url = selectedCollection
            ? `/api/status?collection=${encodeURIComponent(selectedCollection)}`
            : '/api/status';

        const response = await fetch(url);
        const data = await response.json();

        const statusContainer = document.getElementById('system-status');
        statusContainer.innerHTML = `
            <div class="status-item">
                <span class="status-label">RAG SYSTEM:</span>
                <span class="status-value ${data.initialized ? '' : 'inactive'}">
                    ${data.initialized ? 'READY' : 'NOT INITIALIZED'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">COLLECTION:</span>
                <span class="status-value">${data.collection || 'N/A'}</span>
            </div>
            <div class="status-item">
                <span class="status-label">VECTOR DOCS:</span>
                <span class="status-value highlight">${data.document_count}</span>
            </div>
            <div class="status-item">
                <span class="status-label">GRAPH NODES:</span>
                <span class="status-value">${data.graph_nodes}</span>
            </div>
            <div class="status-item">
                <span class="status-label">GRAPH EDGES:</span>
                <span class="status-value">${data.graph_edges || 0}</span>
            </div>
        `;
    } catch (error) {
        document.getElementById('system-status').innerHTML =
            `<p style="color: #FF0000;">Error loading status: ${error.message}</p>`;
    }
}

document.getElementById('refresh-status').addEventListener('click', async () => {
    await loadCollections();
    await refreshStatus();
});

// Helper to auto-hide success messages after 3 seconds
function showCollectionStatus(message, type) {
    showStatus('collection-status', message, type);
    if (type === 'success') {
        setTimeout(() => {
            document.getElementById('collection-status').classList.add('hidden');
        }, 3000);
    }
}

// Auto-switch when collection is selected
document.getElementById('collection-select').addEventListener('change', async () => {
    const selectedCollection = document.getElementById('collection-select').value;
    if (!selectedCollection) return;

    // Show inline progress
    const progressContainer = document.getElementById('switch-progress-container');
    const progressBar = document.getElementById('switch-progress-bar');
    progressContainer.classList.remove('hidden');
    progressBar.style.width = '10%';

    // Fake progress animation
    let progress = 10;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += 5;
            progressBar.style.width = `${progress}%`;
        }
    }, 100);

    try {
        const response = await fetch(`/api/switch-collection?collection=${encodeURIComponent(selectedCollection)}`, {
            method: 'POST'
        });

        clearInterval(progressInterval);
        progressBar.style.width = '100%';

        const data = await response.json();

        if (response.ok) {
            // Update all collection inputs across the app
            document.getElementById('upload-collection').value = selectedCollection;
            document.getElementById('url-collection').value = selectedCollection;
            document.getElementById('query-collection').value = selectedCollection;

            await refreshStatus();
            showCollectionStatus(`Switched to collection: ${selectedCollection}`, 'success');
        } else {
            showCollectionStatus(`Error: ${data.detail}`, 'error');
            progressBar.style.backgroundColor = '#FF0000';
        }
    } catch (error) {
        clearInterval(progressInterval);
        showCollectionStatus(`Error: ${error.message}`, 'error');
        progressBar.style.backgroundColor = '#FF0000';
    } finally {
        setTimeout(() => {
            progressContainer.classList.add('hidden');
            progressBar.style.width = '0%';
            progressBar.style.backgroundColor = '';
        }, 1000);
    }
});

document.getElementById('init-btn').addEventListener('click', async () => {
    const selectedCollection = document.getElementById('collection-select').value;

    if (!selectedCollection) {
        showCollectionStatus('Please select a collection to initialize', 'error');
        return;
    }

    showLoading('INITIALIZING RAG SYSTEM...');

    try {
        const response = await fetch(`/api/initialize?collection=${encodeURIComponent(selectedCollection)}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            await loadCollections();
            await refreshStatus();
            showCollectionStatus('RAG system initialized successfully!', 'success');
        } else {
            showCollectionStatus(`Error: ${data.detail}`, 'error');
        }
    } catch (error) {
        showCollectionStatus(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
});
// Delete collection modal handling
let pendingDeleteCollection = null;

document.getElementById('delete-collection-btn').addEventListener('click', () => {
    const selectedCollection = document.getElementById('collection-select').value;
    if (!selectedCollection) {
        showCollectionStatus('Please select a collection to delete', 'error');
        return;
    }

    // Store the collection to delete and show confirmation modal
    pendingDeleteCollection = selectedCollection;
    document.getElementById('delete-modal-collection').textContent = selectedCollection;
    document.getElementById('delete-modal').classList.remove('hidden');
});

document.getElementById('delete-cancel-btn').addEventListener('click', () => {
    document.getElementById('delete-modal').classList.add('hidden');
    pendingDeleteCollection = null;
});

document.getElementById('delete-confirm-btn').addEventListener('click', async () => {
    if (!pendingDeleteCollection) return;

    document.getElementById('delete-modal').classList.add('hidden');
    showLoading('DELETING COLLECTION...');

    try {
        const response = await fetch(
            `/api/delete-collection?collection=${encodeURIComponent(pendingDeleteCollection)}`,
            { method: 'DELETE' }
        );

        const data = await response.json();

        if (response.ok) {
            await loadCollections();
            await refreshStatus();
            showCollectionStatus(`Collection "${pendingDeleteCollection}" deleted successfully!`, 'success');
        } else {
            showCollectionStatus(`Error: ${data.detail}`, 'error');
        }
    } catch (error) {
        showCollectionStatus(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
        pendingDeleteCollection = null;
    }
});

// ============================================================================
// Custom Dropdowns Logic (Select & ComboBox)
// ============================================================================

function setupCustomDropdown(selectId) {
    const originalSelect = document.getElementById(selectId);
    if (!originalSelect) return;

    // Check if already converted
    if (originalSelect.classList.contains('hidden-select')) {
        refreshCustomDropdown(selectId);
        return;
    }

    // Create wrapper if not exists
    let wrapper = originalSelect.parentElement.querySelector('.custom-select-wrapper');
    if (!wrapper) {
        wrapper = document.createElement('div');
        wrapper.className = 'custom-select-wrapper';
        originalSelect.parentNode.insertBefore(wrapper, originalSelect);
        wrapper.appendChild(originalSelect);
    }

    // Create custom UI
    const oldCustom = wrapper.querySelector('.custom-select');
    if (oldCustom) oldCustom.remove();

    const customSelect = document.createElement('div');
    customSelect.className = 'custom-select';
    customSelect.id = `custom-${selectId}`;

    const trigger = document.createElement('div');
    trigger.className = 'custom-select-trigger';
    trigger.innerHTML = '<span>Loading...</span>';

    const options = document.createElement('div');
    options.className = 'custom-options';

    customSelect.appendChild(trigger);
    customSelect.appendChild(options);
    wrapper.appendChild(customSelect);

    originalSelect.classList.add('hidden-select');

    // Event listeners
    trigger.addEventListener('click', function (e) {
        // Close all other selects and comboboxes
        closeAllDropdowns(customSelect);
        customSelect.classList.toggle('open');
        e.stopPropagation();
    });

    // Populate options
    refreshCustomDropdown(selectId);
}

function refreshCustomDropdown(selectId) {
    const originalSelect = document.getElementById(selectId);
    const customSelect = document.getElementById(`custom-${selectId}`);
    if (!originalSelect || !customSelect) return;

    const trigger = customSelect.querySelector('.custom-select-trigger');
    const optionsContainer = customSelect.querySelector('.custom-options');

    // Update trigger text
    const selectedOption = originalSelect.options[originalSelect.selectedIndex];
    if (selectedOption) {
        trigger.innerHTML = `<span>${selectedOption.text}</span>`;
    } else {
        trigger.innerHTML = `<span>Select...</span>`;
    }

    // Rebuild options
    optionsContainer.innerHTML = '';
    Array.from(originalSelect.options).forEach(opt => {
        const optionDiv = document.createElement('div');
        optionDiv.className = 'custom-option';
        if (opt.selected) optionDiv.classList.add('selected');
        optionDiv.dataset.value = opt.value;
        optionDiv.textContent = opt.text;

        optionDiv.addEventListener('click', function () {
            originalSelect.value = this.dataset.value;

            // Trigger change event on original select
            const event = new Event('change');
            originalSelect.dispatchEvent(event);

            // Update UI
            trigger.innerHTML = `<span>${this.textContent}</span>`;
            customSelect.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
            this.classList.add('selected');
            customSelect.classList.remove('open');
        });

        optionsContainer.appendChild(optionDiv);
    });
}

function setupCustomComboBox(inputId, datalistId) {
    const input = document.getElementById(inputId);
    if (!input) return;

    // Check if already converted
    if (input.dataset.customCombobox) {
        refreshCustomComboBox(inputId, datalistId);
        return;
    }

    // Mark as converted
    input.dataset.customCombobox = "true";

    // Remove native list attribute to prevent double dropdown
    input.removeAttribute('list');

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'custom-combobox-wrapper';
    input.parentNode.insertBefore(wrapper, input);
    wrapper.appendChild(input);

    input.classList.add('custom-combobox-input');

    // Arrow Trigger
    const arrow = document.createElement('div');
    arrow.className = 'custom-combobox-arrow';
    arrow.innerHTML = '▼';
    wrapper.appendChild(arrow);

    // Dropdown List
    const optionsList = document.createElement('div');
    optionsList.className = 'custom-options';
    optionsList.id = `options-${inputId}`;
    wrapper.appendChild(optionsList);

    // Events
    const toggle = (e) => {
        if (wrapper.classList.contains('open')) {
            wrapper.classList.remove('open');
        } else {
            closeAllDropdowns(wrapper);
            refreshCustomComboBox(inputId, datalistId); // Refresh filtering
            wrapper.classList.add('open');
        }
        e.stopPropagation();
    };

    arrow.addEventListener('click', toggle);

    input.addEventListener('focus', () => {
        closeAllDropdowns(wrapper);
        wrapper.classList.add('open');
    });

    input.addEventListener('input', () => {
        const filter = input.value.toLowerCase();
        const opts = optionsList.querySelectorAll('.custom-option');
        let hasVisible = false;

        opts.forEach(opt => {
            const text = opt.textContent.toLowerCase();
            if (text.includes(filter)) {
                opt.style.display = 'block';
                hasVisible = true;
            } else {
                opt.style.display = 'none';
            }
        });

        if (hasVisible) {
            wrapper.classList.add('open');
        } else {
            // wrapper.classList.remove('open'); // Keep open even if empty? No.
        }
    });

    // Populate
    refreshCustomComboBox(inputId, datalistId);
}

function refreshCustomComboBox(inputId, datalistId) {
    const input = document.getElementById(inputId);
    const datalist = document.getElementById(datalistId);
    if (!input) return;

    const optionsContainer = document.getElementById(`options-${inputId}`);
    if (!optionsContainer) return;

    optionsContainer.innerHTML = '';

    if (datalist && datalist.options.length > 0) {
        Array.from(datalist.options).forEach(opt => {
            const div = document.createElement('div');
            div.className = 'custom-option';
            div.textContent = opt.value;
            div.dataset.value = opt.value;

            div.addEventListener('click', () => {
                input.value = div.dataset.value;
                input.parentElement.classList.remove('open');
            });

            optionsContainer.appendChild(div);
        });
    } else {
        const div = document.createElement('div');
        div.className = 'custom-option';
        div.textContent = "No collections found";
        div.style.color = '#666';
        div.style.cursor = 'default';
        div.style.borderBottom = 'none';
        optionsContainer.appendChild(div);
    }
}

function closeAllDropdowns(exceptElement) {
    document.querySelectorAll('.custom-select').forEach(s => {
        if (s !== exceptElement) s.classList.remove('open');
    });
    document.querySelectorAll('.custom-combobox-wrapper').forEach(s => {
        if (s !== exceptElement) s.classList.remove('open');
    });
}

// Close dropdowns when clicking outside
window.addEventListener('click', function (e) {
    if (!e.target.closest('.custom-select') && !e.target.closest('.custom-combobox-wrapper')) {
        closeAllDropdowns(null);
    }
});


// ============================================================================
// Model Download
// ============================================================================

async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-status');
        const data = await response.json();
        
        const btn = document.getElementById('download-model-btn');
        if (!btn) return; // Guard against missing button
        
        if (data.downloaded) {
            btn.classList.add('downloaded');
            btn.disabled = true;
            btn.querySelector('.btn-icon').textContent = '';
            btn.querySelector('.btn-text').textContent = 'MODEL READY';
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
    }
}

function setupModelDownloadButton() {
    const btn = document.getElementById('download-model-btn');
    if (!btn) return; // Guard against missing button
    
    btn.addEventListener('click', async () => {
        if (btn.disabled || btn.classList.contains('downloaded')) return;
        
        btn.disabled = true;
        btn.querySelector('.btn-text').textContent = 'STARTING...';
        
        try {
            const response = await fetch('/api/download-model', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'exists') {
                btn.classList.add('downloaded');
                btn.querySelector('.btn-icon').textContent = '';
                btn.querySelector('.btn-text').textContent = 'MODEL READY';
                alert('Model already downloaded!');
                return;
            }
            
            if (data.status === 'started') {
                // Poll for download progress
                pollModelDownload(data.job_id);
            }
        } catch (error) {
            console.error('Model download failed:', error);
            alert('Failed to start model download: ' + error.message);
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'MODEL';
        }
    });
}

async function pollModelDownload(jobId) {
    const btn = document.getElementById('download-model-btn');
    
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/model-download-status/${jobId}`);
            
            // If job not found (404), stop polling
            if (response.status === 404) {
                clearInterval(pollInterval);
                btn.disabled = false;
                btn.querySelector('.btn-text').textContent = 'MODEL';
                console.error('Job not found, stopping poll');
                return;
            }
            
            const data = await response.json();
            
            // Update button text with progress and message
            if (data.progress >= 0 && data.progress < 100) {
                btn.querySelector('.btn-text').textContent = `${data.progress}%`;
            }
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                btn.classList.add('downloaded');
                btn.querySelector('.btn-icon').textContent = '';
                btn.querySelector('.btn-text').textContent = 'MODEL READY';
                alert('Model downloaded successfully!');
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                btn.disabled = false;
                btn.querySelector('.btn-text').textContent = 'MODEL';
                alert('Model download failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            clearInterval(pollInterval);
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = 'MODEL';
            console.error('Status poll failed:', error);
        }
    }, 1000);
}

// ============================================================================
// Initial Load
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // Setup static dropdowns first
    setupCustomDropdown('chunking');

    // Setup ComboBoxes (Inputs with Datalist)
    setupCustomComboBox('upload-collection', 'collections-list');
    setupCustomComboBox('url-collection', 'collections-list');

    // Load data and setup dynamic dropdowns
    await loadCollections();
    await refreshStatus();

    // Setup remaining dropdowns if not covered by loadCollections
    setupCustomDropdown('query-collection');
    setupCustomDropdown('collection-select');
    
    // Setup model download button and check status
    setupModelDownloadButton();
    await checkModelStatus();
});
