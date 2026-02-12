/**
 * Produce Identifier – Frontend logic
 *
 * Handles:
 *  - System status check & bootstrap banner
 *  - Auto-setup via /setup-demo
 *  - Image upload, drag-and-drop, prediction requests, result rendering
 */

(() => {
    "use strict";

    // ─── DOM references ──────────────────────────────────────
    const uploadArea      = document.getElementById("uploadArea");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");
    const imagePreview    = document.getElementById("imagePreview");
    const fileInput       = document.getElementById("fileInput");
    const predictBtn      = document.getElementById("predictBtn");
    const clearBtn        = document.getElementById("clearBtn");
    const loader          = document.getElementById("loader");
    const errorMsg        = document.getElementById("errorMsg");
    const results         = document.getElementById("results");
    const predictionCards = document.getElementById("predictionCards");
    const summaryContent  = document.getElementById("summaryContent");

    // Status banner
    const statusBanner = document.getElementById("statusBanner");
    const statusIcon   = document.getElementById("statusIcon");
    const statusText   = document.getElementById("statusText");
    const statusCmd    = document.getElementById("statusCmd");
    const setupBtn     = document.getElementById("setupBtn");

    // Setup panel
    const setupPanel   = document.getElementById("setupPanel");
    const setupLogs    = document.getElementById("setupLogs");
    const setupResult  = document.getElementById("setupResult");

    let selectedFile = null;

    // ─── On page load: check status ──────────────────────────
    checkStatus();

    async function checkStatus() {
        try {
            const resp = await fetch("/status");
            if (!resp.ok) return;
            const data = await resp.json();

            if (data.model_loaded) {
                // Model is ready – hide banner or show success
                statusBanner.classList.add("hidden");
                return;
            }

            // Show warning banner
            statusBanner.classList.remove("hidden");
            statusBanner.classList.remove("success");

            if (!data.raw_data_present) {
                statusText.textContent = "No dataset or trained model found. Click below to auto-download data and train a demo model.";
            } else if (!data.processed_data_present) {
                statusText.textContent = "Raw data found but not split yet. Click below to prepare and train.";
            } else {
                statusText.textContent = "Data is ready but no trained model found. Click below to train.";
            }

            if (data.suggested_next_command) {
                statusCmd.textContent = data.suggested_next_command;
                statusCmd.classList.remove("hidden");
            } else {
                statusCmd.classList.add("hidden");
            }
        } catch (e) {
            // Server might not be running
        }
    }

    // ─── Setup button ────────────────────────────────────────
    setupBtn.addEventListener("click", runSetup);

    async function runSetup() {
        setupBtn.disabled = true;
        setupPanel.classList.remove("hidden");
        setupLogs.innerHTML = "";
        setupResult.classList.add("hidden");

        addLog("info", "Starting auto setup (download + prepare + train) ...");

        try {
            const resp = await fetch("/setup-demo", { method: "POST" });
            const data = await resp.json();

            // Render logs
            if (data.logs) {
                data.logs.forEach((entry) => {
                    const icon = entry.success ? "success" : "fail";
                    addLog(icon, `[${entry.step}] ${entry.message}`);
                });
            }

            // Result
            setupResult.classList.remove("hidden");
            if (data.success) {
                setupResult.className = "setup-result ok";
                setupResult.textContent = data.message;

                // Update banner
                statusBanner.classList.add("success");
                statusIcon.textContent = "\u2713";
                statusText.textContent = "Model is trained and ready! Upload an image to classify.";
                statusCmd.classList.add("hidden");
                setupBtn.classList.add("hidden");
            } else {
                setupResult.className = "setup-result fail";
                setupResult.textContent = "Setup failed: " + data.message;
                setupBtn.disabled = false;
            }

            // Hide spinner in panel header
            const hdr = setupPanel.querySelector(".setup-header");
            if (hdr) hdr.innerHTML = "<span>" + (data.success ? "Setup Complete" : "Setup Failed") + "</span>";

        } catch (err) {
            addLog("fail", "Request failed: " + err.message);
            setupResult.classList.remove("hidden");
            setupResult.className = "setup-result fail";
            setupResult.textContent = "Network error – is the server running?";
            setupBtn.disabled = false;
        }
    }

    function addLog(type, message) {
        const line = document.createElement("div");
        line.className = "log-line";
        const icons = { success: "\u2714", fail: "\u2718", info: "\u25B6" };
        const cls   = { success: "log-success", fail: "log-fail", info: "log-info" };
        line.innerHTML = `<span class="log-icon ${cls[type] || 'log-info'}">${icons[type] || "\u25B6"}</span><span>${escapeHtml(message)}</span>`;
        setupLogs.appendChild(line);
        setupLogs.scrollTop = setupLogs.scrollHeight;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ─── File selection ──────────────────────────────────────
    uploadArea.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // ─── Drag & drop ─────────────────────────────────────────
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ─── Handle selected file ────────────────────────────────
    function handleFile(file) {
        const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!allowed.includes(file.type)) {
            showError("Unsupported file type. Please use JPG, PNG, or WebP.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            showError("File is too large. Maximum size is 10 MB.");
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("hidden");
            uploadPlaceholder.classList.add("hidden");
        };
        reader.readAsDataURL(file);

        predictBtn.disabled = false;
        clearBtn.classList.remove("hidden");
        hideError();
        results.classList.add("hidden");
    }

    // ─── Clear ───────────────────────────────────────────────
    clearBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        imagePreview.classList.add("hidden");
        uploadPlaceholder.classList.remove("hidden");
        predictBtn.disabled = true;
        clearBtn.classList.add("hidden");
        results.classList.add("hidden");
        hideError();
    });

    // ─── Predict ─────────────────────────────────────────────
    predictBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        hideError();
        results.classList.add("hidden");
        loader.classList.remove("hidden");
        predictBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            showError(err.message || "Something went wrong. Please try again.");
        } finally {
            loader.classList.add("hidden");
            predictBtn.disabled = false;
        }
    });

    // ─── Render results ──────────────────────────────────────
    function renderResults(data) {
        // Top-k prediction cards
        predictionCards.innerHTML = "";
        data.top_k.forEach((item, idx) => {
            const pct = (item.probability * 100).toFixed(1);
            const barClass = item.category === "fruit" ? "fruit" : "vegetable";
            const badgeClass = item.category === "fruit" ? "badge-fruit" : "badge-vegetable";

            const card = document.createElement("div");
            card.className = "pred-card";
            card.innerHTML = `
                <span class="pred-rank">${idx + 1}</span>
                <div class="pred-info">
                    <div>
                        <span class="pred-label">${item.label.replace(/_/g, " ")}</span>
                        <span class="badge ${badgeClass}">${item.category}</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar ${barClass}" style="width: ${pct}%"></div>
                    </div>
                </div>
                <span class="pred-pct">${pct}%</span>
            `;
            predictionCards.appendChild(card);
        });

        // Summary
        const s = data.summary;
        const fruitPct = (s.fruit_probability * 100).toFixed(1);
        const vegPct = (s.vegetable_probability * 100).toFixed(1);

        summaryContent.innerHTML = `
            <div class="summary-card fruit">
                <div class="cat-label">Fruit</div>
                <div class="cat-prob">${fruitPct}%</div>
            </div>
            <div class="summary-card vegetable">
                <div class="cat-label">Vegetable</div>
                <div class="cat-prob">${vegPct}%</div>
            </div>
            <div class="summary-winner">
                Predicted super-category: <strong>${s.predicted_supercategory}</strong>
            </div>
        `;

        results.classList.remove("hidden");
    }

    // ─── Error helpers ───────────────────────────────────────
    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove("hidden");
    }

    function hideError() {
        errorMsg.textContent = "";
        errorMsg.classList.add("hidden");
    }
})();
