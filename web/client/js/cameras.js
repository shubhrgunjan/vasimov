/**
 * vasimov/web/client/js/cameras.js
 * Multi-camera Workspace: Real-time grid of all model cameras & viewer camera.
 */

class CamerasWorkspace {
  constructor() {
    this.selectedCamera = "front_camera";
    this.camerasList = [];
    this.gridContainer = document.getElementById("camerasGridContainer");
    this.isPaused = false;
  }

  updateFrame(frame) {
    if (!frame || !frame.cameras) return;

    // Check if cameras list changed
    const newCams = frame.cameras.map((c) => c.name).join(",");
    const oldCams = this.camerasList.map((c) => c.name).join(",");

    if (newCams !== oldCams) {
      this.camerasList = frame.cameras;
      this.renderGrid();
    }

    // Update timestamps and FPS on cards
    for (const c of frame.cameras) {
      const elTime = document.getElementById(`camSimTime_${c.name}`);
      const elSeq = document.getElementById(`camSeq_${c.name}`);
      if (elTime) elTime.innerText = `${frame.meta.sim_time.toFixed(2)}s`;
      if (elSeq) elSeq.innerText = `#${frame.meta.sequence}`;
    }
  }

  renderGrid() {
    if (!this.gridContainer) return;
    let html = "";

    for (const c of this.camerasList) {
      const isViewer = c.name === "viewer_camera";
      const badgeClass = isViewer ? "connecting" : "connected";
      const typeLabel = isViewer ? "VIEWER CAMERA" : "MODEL CAMERA";
      const srcUrl = `/api/camera/frame?camera=${encodeURIComponent(c.name)}`;

      html += `
        <div class="camera-card" id="camCard_${c.name}">
          <div class="camera-card-header">
            <div>
              <strong>${c.name}</strong>
              <span class="badge ${badgeClass}" style="margin-left: 6px;">${typeLabel}</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">
              <span id="camSimTime_${c.name}">0.00s</span>
              <span id="camSeq_${c.name}" style="margin-left: 6px;">#0</span>
            </div>
          </div>
          <div class="camera-card-body" onclick="window.camerasWorkspace.selectCamera('${c.name}')">
            <img id="camImg_${c.name}" src="${srcUrl}" alt="${c.name} live view" />
          </div>
        </div>
      `;
    }

    this.gridContainer.innerHTML = html;
    this.startThumbnailLoop();
  }

  startThumbnailLoop() {
    if (this._thumbnailLoopStarted) return;
    this._thumbnailLoopStarted = true;

    setInterval(() => {
      // Only refresh thumbnails if the cameras tab is currently active
      const isCamerasTab = document.getElementById("tab-cameras")?.classList.contains("active");
      if (!isCamerasTab) return;

      const ts = Date.now();
      for (const c of this.camerasList) {
        const img = document.getElementById(`camImg_${c.name}`);
        if (img) {
          img.src = `/api/camera/frame?camera=${encodeURIComponent(c.name)}&_t=${ts}`;
        }
      }
    }, 150); // ~6.6 FPS thumbnail update
  }

  selectCamera(name) {
    this.selectedCamera = name;
    if (window.liveViewController) {
      window.liveViewController.currentCamera = name;
      const select = document.getElementById("liveCameraSelect");
      if (select) select.value = name;
      window.liveViewController.updateCameraFeed();
    }
    // Switch to LIVE tab to see full view
    document.querySelector('.nav-tab[data-tab="live"]')?.click();
  }
}

window.camerasWorkspace = new CamerasWorkspace();
