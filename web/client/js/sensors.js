/**
 * vasimov/web/client/js/sensors.js
 * Sensors Workspace: Enumerate all 84 MuJoCo sensors, search, filter, and trace.
 */

class SensorsWorkspace {
  constructor() {
    this.selectedSensorId = 0;
    this.currentCategory = "all";
    this.searchQuery = "";
    this.history = {}; // id -> [[val0, val1, ...], ...]
    this.maxHistory = 60;

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.tableBody = document.getElementById("sensorsTableBody");
    this.searchInput = document.getElementById("sensorSearchInput");
    this.canvas = document.getElementById("sensorTraceCanvas");
    this.ctx = this.canvas?.getContext("2d");
  }

  bindEvents() {
    if (this.searchInput) {
      this.searchInput.addEventListener("input", (e) => {
        this.searchQuery = e.target.value.toLowerCase().trim();
        this.renderTable(window.telemetryManager.latestFrame);
      });
    }

    document.querySelectorAll(".sensor-cat-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".sensor-cat-btn").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.currentCategory = e.target.getAttribute("data-cat");
        this.renderTable(window.telemetryManager.latestFrame);
      });
    });
  }

  selectSensor(id) {
    this.selectedSensorId = id;
    this.renderDetail();
    this.renderTable(window.telemetryManager.latestFrame);
  }

  getSensorEntry(id) {
    const frame = window.telemetryManager.latestFrame;
    if (!frame || !frame.sensors) return null;
    return frame.sensors.find((s) => s.id === id);
  }

  updateFrame(frame) {
    if (!frame || !frame.sensors) return;

    // Record history for selected sensor
    const s = frame.sensors.find((x) => x.id === this.selectedSensorId);
    if (s && s.values) {
      if (!this.history[this.selectedSensorId]) {
        this.history[this.selectedSensorId] = [];
      }
      const arr = this.history[this.selectedSensorId];
      arr.push([...s.values]);
      if (arr.length > this.maxHistory) arr.shift();
    }

    this.renderTable(frame);
    this.renderDetail();
    this.renderTrace();
  }

  renderTable(frame) {
    if (!this.tableBody || !frame || !frame.sensors) return;

    let html = "";
    for (const s of frame.sensors) {
      // Category filter
      if (this.currentCategory !== "all" && s.category.toLowerCase() !== this.currentCategory) {
        continue;
      }
      // Search query
      if (this.searchQuery && !s.name.toLowerCase().includes(this.searchQuery) && !s.type.toLowerCase().includes(this.searchQuery)) {
        continue;
      }

      const isSelected = s.id === this.selectedSensorId;
      const formattedVals = s.values.map((v) => (typeof v === "number" ? v.toFixed(3) : v)).join(", ");

      html += `
        <tr class="${isSelected ? "selected" : ""}" onclick="window.sensorsWorkspace.selectSensor(${s.id})">
          <td class="num">${s.id}</td>
          <td><strong>${s.name}</strong></td>
          <td><span class="badge" style="font-size: 9px;">${s.type}</span></td>
          <td><span class="badge ${s.category === "IMU" ? "connecting" : "connected"}" style="font-size: 9px;">${s.category}</span></td>
          <td class="num">${s.dim}</td>
          <td class="num" style="max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">[${formattedVals}]</td>
          <td>${s.source}</td>
        </tr>
      `;
    }
    this.tableBody.innerHTML = html;
  }

  renderDetail() {
    const s = this.getSensorEntry(this.selectedSensorId);
    if (!s) return;

    document.getElementById("detailSensorTitle").innerText = s.name.toUpperCase();
    document.getElementById("detailSensorId").innerText = `#${s.id}`;
    document.getElementById("detailSensorType").innerText = s.type;
    document.getElementById("detailSensorCategory").innerText = s.category;
    document.getElementById("detailSensorDim").innerText = s.dim;
    document.getElementById("detailSensorSource").innerText = s.source;
    document.getElementById("detailSensorTime").innerText = `${s.timestamp.toFixed(3)}s`;

    // Format components list
    const compContainer = document.getElementById("detailSensorComponents");
    if (compContainer) {
      let cHtml = "";
      const labels = s.dim === 3 ? ["X", "Y", "Z"] : s.dim === 4 ? ["W", "X", "Y", "Z"] : ["Val"];
      for (let i = 0; i < s.values.length; i++) {
        const lbl = labels[i] || `[${i}]`;
        const v = s.values[i];
        cHtml += `
          <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid var(--border-subtle); font-family: var(--font-mono); font-size: 11px;">
            <span style="color: var(--text-muted);">${lbl}:</span>
            <strong>${typeof v === "number" ? v.toFixed(4) : v}</strong>
          </div>
        `;
      }
      compContainer.innerHTML = cHtml;
    }
  }

  renderTrace() {
    if (!this.ctx || !this.canvas) return;
    const history = this.history[this.selectedSensorId] || [];
    if (history.length < 2) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    const dim = history[0].length;
    const colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];

    for (let d = 0; d < Math.min(dim, 4); d++) {
      const vals = history.map((row) => row[d] || 0.0);
      let minVal = Math.min(...vals);
      let maxVal = Math.max(...vals);
      if (Math.abs(maxVal - minVal) < 0.01) {
        minVal -= 0.01;
        maxVal += 0.01;
      }

      const getY = (val) => h - ((val - minVal) / (maxVal - minVal)) * (h - 16) - 8;
      const getX = (idx, total) => (idx / (total - 1)) * w;

      this.ctx.strokeStyle = colors[d % colors.length];
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      for (let i = 0; i < vals.length; i++) {
        const x = getX(i, vals.length);
        const y = getY(vals[i]);
        if (i === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
      }
      this.ctx.stroke();
    }
  }
}

window.sensorsWorkspace = new SensorsWorkspace();
