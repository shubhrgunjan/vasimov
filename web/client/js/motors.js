/**
 * vasimov/web/client/js/motors.js
 * Actuators & Motors Workspace: 25 Actuators, filters, manual controls, and live traces.
 */

class MotorsWorkspace {
  constructor() {
    this.selectedJoint = "left_knee_joint";
    this.currentFilter = "all";
    this.history = {
      timestamps: [],
      pos: {},
      target: {},
      torque: {},
    };
    this.maxHistory = 60; // ~3-4 seconds

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.tableBody = document.getElementById("motorsTableBody");
    this.canvas = document.getElementById("motorTraceCanvas");
    this.ctx = this.canvas?.getContext("2d");
  }

  bindEvents() {
    document.querySelectorAll(".filter-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.currentFilter = e.target.getAttribute("data-filter");
        this.renderTable(window.telemetryManager.latestFrame);
      });
    });

    // Manual Adjust Buttons
    document.getElementById("btnJointMinus05")?.addEventListener("click", () => this.adjustSelectedTarget(-0.05));
    document.getElementById("btnJointMinus01")?.addEventListener("click", () => this.adjustSelectedTarget(-0.01));
    document.getElementById("btnJointPlus01")?.addEventListener("click", () => this.adjustSelectedTarget(0.01));
    document.getElementById("btnJointPlus05")?.addEventListener("click", () => this.adjustSelectedTarget(0.05));

    document.getElementById("btnSetJointTarget")?.addEventListener("click", () => {
      const input = document.getElementById("manualJointTargetInput");
      if (input && input.value !== "") {
        const val = parseFloat(input.value);
        window.telemetryManager.sendCommand("joint", { joint: this.selectedJoint, target: val });
      }
    });

    document.getElementById("btnZeroJointTarget")?.addEventListener("click", () => {
      const jEntry = this.getJointEntry(this.selectedJoint);
      if (jEntry) {
        window.telemetryManager.sendCommand("joint", { joint: this.selectedJoint, target: 0.0 });
      }
    });
  }

  adjustSelectedTarget(delta) {
    const jEntry = this.getJointEntry(this.selectedJoint);
    if (!jEntry) return;
    const newTgt = (jEntry.target || 0.0) + delta;
    window.telemetryManager.sendCommand("joint", { joint: this.selectedJoint, target: newTgt });
  }

  getJointEntry(name) {
    const frame = window.telemetryManager.latestFrame;
    if (!frame || !frame.joints) return null;
    return frame.joints.find((j) => j.name === name);
  }

  selectJoint(name) {
    this.selectedJoint = name;
    if (window.liveViewController) {
      document.getElementById("selectedComponentName").innerText = name;
    }
    this.renderDetail();
    this.renderTable(window.telemetryManager.latestFrame);
  }

  updateFrame(frame) {
    if (!frame || !frame.joints) return;

    // Record history for selected joint
    const j = frame.joints.find((x) => x.name === this.selectedJoint);
    if (j) {
      if (!this.history.pos[this.selectedJoint]) {
        this.history.pos[this.selectedJoint] = [];
        this.history.target[this.selectedJoint] = [];
        this.history.torque[this.selectedJoint] = [];
      }
      const pArr = this.history.pos[this.selectedJoint];
      const tArr = this.history.target[this.selectedJoint];
      const trqArr = this.history.torque[this.selectedJoint];

      pArr.push(j.position);
      tArr.push(j.target);
      trqArr.push(j.torque);

      if (pArr.length > this.maxHistory) {
        pArr.shift();
        tArr.shift();
        trqArr.shift();
      }
    }

    this.renderTable(frame);
    this.renderDetail();
    this.renderTrace();
  }

  filterMatches(j) {
    const n = j.name.toLowerCase();
    switch (this.currentFilter) {
      case "legs":
        return n.includes("hip") || n.includes("knee") || n.includes("ankle");
      case "arms":
        return n.includes("shoulder") || n.includes("elbow") || n.includes("wrist");
      case "waist":
        return n.includes("waist") || n.includes("neck");
      case "active":
        return Math.abs(j.velocity) > 0.01 || Math.abs(j.torque) > 0.5;
      case "near_limit":
        return j.pos_pct_range < 15 || j.pos_pct_range > 85;
      case "high_torque":
        return j.effort_pct > 50;
      default:
        return true;
    }
  }

  renderTable(frame) {
    if (!this.tableBody || !frame || !frame.joints) return;

    let html = "";
    for (const j of frame.joints) {
      if (!this.filterMatches(j)) continue;
      const isSelected = j.name === this.selectedJoint;
      const trqColor = j.effort_pct > 75 ? "color: var(--danger);" : j.effort_pct > 40 ? "color: var(--warning);" : "";

      html += `
        <tr class="${isSelected ? "selected" : ""}" onclick="window.motorsWorkspace.selectJoint('${j.name}')">
          <td class="num">${j.index}</td>
          <td><strong>${j.name}</strong></td>
          <td class="num">${j.position.toFixed(3)}</td>
          <td class="num">${j.target.toFixed(3)}</td>
          <td class="num">${(j.target - j.position).toFixed(3)}</td>
          <td class="num">${j.velocity.toFixed(2)}</td>
          <td class="num" style="${trqColor}">${j.torque.toFixed(2)}</td>
          <td class="num" style="${trqColor}">${j.effort_pct.toFixed(0)}%</td>
          <td class="num">${j.effort_limit.toFixed(0)} Nm</td>
          <td><span class="badge ${j.controller_source === "manual" ? "connecting" : "connected"}">${j.controller_source.toUpperCase()}</span></td>
        </tr>
      `;
    }
    this.tableBody.innerHTML = html;
  }

  renderDetail() {
    const j = this.getJointEntry(this.selectedJoint);
    if (!j) return;

    document.getElementById("detailMotorTitle").innerText = j.name.toUpperCase();
    document.getElementById("detailMotorPos").innerText = `${j.position.toFixed(4)} rad`;
    document.getElementById("detailMotorTarget").innerText = `${j.target.toFixed(4)} rad`;
    document.getElementById("detailMotorVel").innerText = `${j.velocity.toFixed(3)} rad/s`;
    document.getElementById("detailMotorTorque").innerText = `${j.torque.toFixed(2)} Nm`;
    document.getElementById("detailMotorEffort").innerText = `${j.effort_pct.toFixed(1)}%`;
    document.getElementById("detailMotorRange").innerText = `[${j.range[0].toFixed(2)}, ${j.range[1].toFixed(2)}] rad`;
    document.getElementById("detailMotorLimit").innerText = `${j.effort_limit.toFixed(1)} Nm`;
    document.getElementById("detailMotorSource").innerText = j.controller_source.toUpperCase();

    const input = document.getElementById("manualJointTargetInput");
    if (input && document.activeElement !== input) {
      input.value = j.target.toFixed(3);
    }
  }

  renderTrace() {
    if (!this.ctx || !this.canvas) return;
    const pArr = this.history.pos[this.selectedJoint] || [];
    const tArr = this.history.target[this.selectedJoint] || [];
    if (pArr.length < 2) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    // Compute min / max
    let minVal = Math.min(...pArr, ...tArr);
    let maxVal = Math.max(...pArr, ...tArr);
    if (Math.abs(maxVal - minVal) < 0.05) {
      minVal -= 0.05;
      maxVal += 0.05;
    }

    const getY = (val) => h - ((val - minVal) / (maxVal - minVal)) * (h - 16) - 8;
    const getX = (idx, total) => (idx / (total - 1)) * w;

    // Draw Target line (cyan dashed)
    this.ctx.strokeStyle = "#06b6d4";
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([4, 4]);
    this.ctx.beginPath();
    for (let i = 0; i < tArr.length; i++) {
      const x = getX(i, tArr.length);
      const y = getY(tArr[i]);
      if (i === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    }
    this.ctx.stroke();

    // Draw Position line (solid blue)
    this.ctx.strokeStyle = "#3b82f6";
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([]);
    this.ctx.beginPath();
    for (let i = 0; i < pArr.length; i++) {
      const x = getX(i, pArr.length);
      const y = getY(pArr[i]);
      if (i === 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    }
    this.ctx.stroke();

    // Trace Legend
    this.ctx.fillStyle = "#8493a8";
    this.ctx.font = "9px monospace";
    this.ctx.fillText(`Target (dash): ${tArr[tArr.length - 1].toFixed(3)}`, 6, 12);
    this.ctx.fillStyle = "#60a5fa";
    this.ctx.fillText(`Pos: ${pArr[pArr.length - 1].toFixed(3)}`, 6, 24);
  }
}

window.motorsWorkspace = new MotorsWorkspace();
