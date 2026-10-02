/**
 * vasimov/web/client/js/live.js
 * Operator LIVE Viewport, Base State, Controls, and Event-driven Dataflow.
 */

class LiveViewController {
  constructor() {
    this.currentCamera = "front_camera";
    this.isXray = false;
    this.orbitParams = { azimuth: 135.0, elevation: -20.0, distance: 3.0 };

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.imgFeed = document.getElementById("liveCameraImg");
    this.camSelect = document.getElementById("liveCameraSelect");
    this.xrayBtn = document.getElementById("xrayToggleBtn");
    this.gantryBadge = document.getElementById("liveGantryBadge");

    // Sliders
    this.sliderVx = document.getElementById("sliderVx");
    this.sliderVy = document.getElementById("sliderVy");
    this.sliderWz = document.getElementById("sliderWz");
    this.valVx = document.getElementById("valVx");
    this.valVy = document.getElementById("valVy");
    this.valWz = document.getElementById("valWz");
  }

  bindEvents() {
    // Camera change
    if (this.camSelect) {
      this.camSelect.addEventListener("change", (e) => {
        this.currentCamera = e.target.value;
        this.updateCameraFeed();
      });
    }

    // X-Ray toggle
    if (this.xrayBtn) {
      this.xrayBtn.addEventListener("click", () => {
        this.isXray = !this.isXray;
        this.xrayBtn.classList.toggle("active", this.isXray);
        this.xrayBtn.innerText = this.isXray ? "X-RAY ACTIVE" : "SOLID";
        this.updateCameraFeed();
      });
    }

    // Velocity Sliders
    const sendVel = () => {
      const vx = parseFloat(this.sliderVx.value);
      const vy = parseFloat(this.sliderVy.value);
      const wz = parseFloat(this.sliderWz.value);
      this.valVx.innerText = `${vx >= 0 ? "+" : ""}${vx.toFixed(2)}`;
      this.valVy.innerText = `${vy >= 0 ? "+" : ""}${vy.toFixed(2)}`;
      this.valWz.innerText = `${wz >= 0 ? "+" : ""}${wz.toFixed(2)}`;
      window.telemetryManager.sendCommand("walk", { vx, vy, wz });
      this.pulsePipelineNode("cmd");
    };

    if (this.sliderVx) this.sliderVx.addEventListener("input", sendVel);
    if (this.sliderVy) this.sliderVy.addEventListener("input", sendVel);
    if (this.sliderWz) this.sliderWz.addEventListener("input", sendVel);

    // Primary action buttons
    document.getElementById("btnStand")?.addEventListener("click", () => {
      window.telemetryManager.sendCommand("stand");
      this.pulsePipelineNode("cmd");
    });
    document.getElementById("btnStop")?.addEventListener("click", () => {
      if (this.sliderVx) this.sliderVx.value = 0;
      if (this.sliderVy) this.sliderVy.value = 0;
      if (this.sliderWz) this.sliderWz.value = 0;
      if (this.valVx) this.valVx.innerText = "+0.00";
      if (this.valVy) this.valVy.innerText = "+0.00";
      if (this.valWz) this.valWz.innerText = "+0.00";
      window.telemetryManager.sendCommand("stop");
      this.pulsePipelineNode("cmd");
    });
    document.getElementById("btnReset")?.addEventListener("click", () => {
      window.telemetryManager.sendCommand("reset", { seed: 42 });
      this.pulsePipelineNode("cmd");
    });
    document.getElementById("btnPause")?.addEventListener("click", () => {
      const isPaused = document.getElementById("btnPause").innerText.includes("RESUME");
      window.telemetryManager.sendCommand(isPaused ? "resume" : "pause");
    });
    document.getElementById("btnStep")?.addEventListener("click", () => {
      window.telemetryManager.sendCommand("step", { n: 1 });
    });

    // Control Modes
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const mode = e.target.getAttribute("data-mode");
        window.telemetryManager.sendCommand("mode", { mode });
      });
    });

    // D-Pad buttons
    document.getElementById("dpadUp")?.addEventListener("click", () => this.adjustVel(0.1, 0, 0));
    document.getElementById("dpadDown")?.addEventListener("click", () => this.adjustVel(-0.1, 0, 0));
    document.getElementById("dpadLeft")?.addEventListener("click", () => this.adjustVel(0, 0.1, 0));
    document.getElementById("dpadRight")?.addEventListener("click", () => this.adjustVel(0, -0.1, 0));
    document.getElementById("dpadTurnL")?.addEventListener("click", () => this.adjustVel(0, 0, 0.2));
    document.getElementById("dpadTurnR")?.addEventListener("click", () => this.adjustVel(0, 0, -0.2));
    document.getElementById("dpadStop")?.addEventListener("click", () => {
      document.getElementById("btnStop")?.click();
    });

    // Keyboard navigation (WASD, QE, Space, P, R)
    window.addEventListener("keydown", (e) => {
      // Don't intercept if user is typing in an input
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;

      const key = e.key.toLowerCase();
      if (key === "w") { e.preventDefault(); this.adjustVel(0.05, 0, 0); }
      else if (key === "s") { e.preventDefault(); this.adjustVel(-0.05, 0, 0); }
      else if (key === "a") { e.preventDefault(); this.adjustVel(0, 0.05, 0); }
      else if (key === "d") { e.preventDefault(); this.adjustVel(0, -0.05, 0); }
      else if (key === "q") { e.preventDefault(); this.adjustVel(0, 0, 0.10); }
      else if (key === "e") { e.preventDefault(); this.adjustVel(0, 0, -0.10); }
      else if (key === " ") { e.preventDefault(); document.getElementById("btnStop")?.click(); }
      else if (key === "p") { e.preventDefault(); document.getElementById("btnPause")?.click(); }
      else if (key === "r") { e.preventDefault(); document.getElementById("btnReset")?.click(); }
    });

    // Fullscreen button
    document.getElementById("btnFullscreen")?.addEventListener("click", () => {
      const vp = document.getElementById("liveViewport");
      if (!document.fullscreenElement) {
        vp?.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    });

    this.updateCameraFeed();
  }

  adjustVel(dvx, dvy, dwz) {
    let vx = parseFloat(this.sliderVx.value) + dvx;
    let vy = parseFloat(this.sliderVy.value) + dvy;
    let wz = parseFloat(this.sliderWz.value) + dwz;
    vx = Math.max(-0.6, Math.min(0.8, vx));
    vy = Math.max(-0.5, Math.min(0.5, vy));
    wz = Math.max(-0.8, Math.min(0.8, wz));

    this.sliderVx.value = vx;
    this.sliderVy.value = vy;
    this.sliderWz.value = wz;
    this.valVx.innerText = `${vx >= 0 ? "+" : ""}${vx.toFixed(2)}`;
    this.valVy.innerText = `${vy >= 0 ? "+" : ""}${vy.toFixed(2)}`;
    this.valWz.innerText = `${wz >= 0 ? "+" : ""}${wz.toFixed(2)}`;

    window.telemetryManager.sendCommand("walk", { vx, vy, wz });
    this.pulsePipelineNode("cmd");
  }

  startCameraLoop() {
    if (this._cameraLoopActive) return;
    this._cameraLoopActive = true;

    let inFlight = false;
    const fetchNext = () => {
      if (!this._cameraLoopActive) return;
      if (inFlight) {
        setTimeout(fetchNext, 30);
        return;
      }
      inFlight = true;
      const url = `/api/camera/frame?camera=${encodeURIComponent(this.currentCamera)}&xray=${this.isXray ? "1" : "0"}&_t=${Date.now()}`;
      const img = new Image();
      img.onload = () => {
        if (this.imgFeed) {
          this.imgFeed.src = img.src;
        }
        inFlight = false;
        setTimeout(fetchNext, 40); // ~25 FPS
      };
      img.onerror = () => {
        inFlight = false;
        setTimeout(fetchNext, 200);
      };
      img.src = url;
    };
    fetchNext();
  }

  updateCameraFeed() {
    // When camera or xray changes, the active camera loop picks it up immediately
    if (!this._cameraLoopActive) {
      this.startCameraLoop();
    }
  }

  pulsePipelineNode(nodeId) {
    const el = document.getElementById(`pipeNode_${nodeId}`);
    if (el) {
      el.classList.add("pulse");
      setTimeout(() => el.classList.remove("pulse"), 250);
    }
  }

  updateFrame(frame) {
    if (!frame) return;

    // 1. Control & Mode
    const ctrl = frame.control || {};
    const mode = ctrl.mode || "POLICY";
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-mode") === mode.toLowerCase());
    });

    // Pause button state
    const pauseBtn = document.getElementById("btnPause");
    if (pauseBtn) {
      pauseBtn.innerText = ctrl.paused ? "RESUME" : "PAUSE";
      pauseBtn.className = ctrl.paused ? "btn btn-primary" : "btn";
    }

    // Gantry status
    if (this.gantryBadge) {
      const active = ctrl.gantry_active;
      this.gantryBadge.innerText = active ? "GANTRY WELDED" : "GANTRY FREE";
      this.gantryBadge.className = active ? "status-badge connecting" : "status-badge connected";
    }

    // 2. Base State
    const base = frame.base || {};
    const pos = base.position || [0, 0, 0];
    const euler = base.euler_deg || [0, 0, 0];
    const bvel = base.body_linear_velocity || [0, 0, 0];
    const bomega = base.body_angular_velocity || [0, 0, 0];

    document.getElementById("liveBaseX").innerText = pos[0].toFixed(3);
    document.getElementById("liveBaseY").innerText = pos[1].toFixed(3);
    document.getElementById("liveBaseZ").innerText = pos[2].toFixed(3);

    document.getElementById("liveRoll").innerText = euler[0].toFixed(2) + "°";
    document.getElementById("livePitch").innerText = euler[1].toFixed(2) + "°";
    document.getElementById("liveYaw").innerText = euler[2].toFixed(2) + "°";

    document.getElementById("liveVx").innerText = bvel[0].toFixed(3);
    document.getElementById("liveVy").innerText = bvel[1].toFixed(3);
    document.getElementById("liveWz").innerText = bomega[2].toFixed(3);

    // 3. Contacts
    const contacts = frame.contacts || {};
    const lf = contacts.left_foot || {};
    const rf = contacts.right_foot || {};

    const elLf = document.getElementById("liveContactLf");
    const elRf = document.getElementById("liveContactRf");
    if (elLf) {
      elLf.className = `contact-pill ${lf.contact ? "contact" : "swing"}`;
      elLf.innerText = lf.contact ? "CONTACT" : "SWING";
    }
    if (elRf) {
      elRf.className = `contact-pill ${rf.contact ? "contact" : "swing"}`;
      elRf.innerText = rf.contact ? "CONTACT" : "SWING";
    }

    const lfFz = lf.force_xyz ? lf.force_xyz[2] : (lf.normal_force || 0);
    const rfFz = rf.force_xyz ? rf.force_xyz[2] : (rf.normal_force || 0);
    document.getElementById("liveForceLf").innerText = `${Math.abs(lfFz).toFixed(1)} N`;
    document.getElementById("liveForceRf").innerText = `${Math.abs(rfFz).toFixed(1)} N`;
    document.getElementById("liveTotalContacts").innerText = contacts.active_contacts_count ?? "-";

    // 4. Update Pipeline Stage Activity
    if (frame.policy && frame.policy.observation_available) {
      this.pulsePipelineNode("obs");
      this.pulsePipelineNode("pol");
      this.pulsePipelineNode("act");
    }
    this.pulsePipelineNode("sim");
    this.pulsePipelineNode("tlm");

    // 5. Populate cameras in dropdown if not already populated
    if (frame.cameras && this.camSelect && this.camSelect.options.length <= 1) {
      this.camSelect.innerHTML = "";
      for (const c of frame.cameras) {
        const opt = document.createElement("option");
        opt.value = c.name;
        opt.innerText = `${c.name} (${c.type})`;
        if (c.name === this.currentCamera) opt.selected = true;
        this.camSelect.appendChild(opt);
      }
    }

    // 6. Camera timestamp overlay
    const overlayTs = document.getElementById("liveCamTimeOverlay");
    if (overlayTs && frame.meta) {
      overlayTs.innerText = `SIM: ${frame.meta.sim_time.toFixed(2)}s | SEQ: ${frame.meta.sequence}`;
    }
  }
}

window.liveViewController = new LiveViewController();
