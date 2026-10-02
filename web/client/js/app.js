/**
 * vasimov/web/client/js/app.js
 * Main Application Orchestrator, Router, and Global Event Handler.
 */

class DashboardApp {
  constructor() {
    this.currentTab = "live";
    this.presentationMode = false;

    this.initTabs();
    this.initTopBar();
    this.initEventLog();
  }

  initTabs() {
    document.querySelectorAll(".nav-tab").forEach((tab) => {
      tab.addEventListener("click", (e) => {
        const target = e.currentTarget.getAttribute("data-tab");
        this.switchTab(target);
      });
    });

    // Presentation view toggle
    document.getElementById("btnTogglePresentation")?.addEventListener("click", () => {
      this.presentationMode = !this.presentationMode;
      const btn = document.getElementById("btnTogglePresentation");
      btn.classList.toggle("active", this.presentationMode);
      btn.innerText = this.presentationMode ? "EXIT PRESENTATION" : "PRESENTATION";
      document.body.classList.toggle("presentation-mode", this.presentationMode);
    });
  }

  switchTab(tabId) {
    this.currentTab = tabId;
    document.querySelectorAll(".nav-tab").forEach((t) => {
      t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".workspace-page").forEach((p) => {
      p.classList.toggle("active", p.id === `page_${tabId}`);
    });
  }

  initTopBar() {
    // Top emergency stop button
    document.getElementById("topEmergencyStop")?.addEventListener("click", () => {
      window.telemetryManager.sendCommand("stop");
      window.telemetryManager.sendCommand("inject", { fault: "fall" }); // Latches FAULT_DAMP
    });
  }

  initEventLog() {
    window.onCommandAck = (ack) => {
      this.renderCommandAck(ack);
    };
  }

  renderCommandAck(ack) {
    if (!ack) return;
    const banner = document.getElementById("commandAckBanner");
    const lc = ack.lifecycle || {};

    if (banner) {
      const isOk = ack.status === "accepted";
      banner.style.display = "block";
      banner.className = `command-ack-box ${isOk ? "ack-ok" : "ack-fail"}`;
      banner.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-weight: 700;">
          <span>CMD #${ack.command_id || 0}: ${ack.action.toUpperCase()} [${ack.status.toUpperCase()}]</span>
          <span>${lc.validation || "OK"}</span>
        </div>
        <div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">
          Applied: ${lc.applied || "-"} | Result: ${lc.simulator_result || "-"}
          ${lc.clipped ? `<br><span style="color: var(--warning);">CLIPPED: ${lc.reason}</span>` : ""}
        </div>
      `;
      // Auto-hide banner after 4 seconds
      clearTimeout(this._ackTimer);
      this._ackTimer = setTimeout(() => {
        banner.style.display = "none";
      }, 4000);
    }
  }

  updateFrame(frame) {
    if (!frame || !frame.meta) return;

    // Top status metrics
    document.getElementById("topSimTime").innerText = `${frame.meta.sim_time.toFixed(2)}s`;
    document.getElementById("topRtf").innerText = `${frame.meta.rtf.toFixed(2)}x`;
    document.getElementById("topPhysicsHz").innerText = `${frame.meta.physics_rate.toFixed(0)} Hz`;
    document.getElementById("topPolicyHz").innerText = `${frame.meta.control_rate.toFixed(0)} Hz`;

    const mode = frame.control?.mode || "POLICY";
    const sub = frame.control?.edge_mode || "STAND";
    document.getElementById("topMode").innerText = `${mode} / ${sub}`;

    // Update event log stream
    const evContainer = document.getElementById("eventLogContainer");
    if (evContainer && frame.events) {
      let eHtml = "";
      for (let i = frame.events.length - 1; i >= Math.max(0, frame.events.length - 10); i--) {
        const ev = frame.events[i];
        eHtml += `
          <div class="event-row ${ev.level || "info"}">
            <span class="event-time">${(ev.time || 0).toFixed(2)}s</span>
            <span class="event-text">${ev.event}</span>
          </div>
        `;
      }
      evContainer.innerHTML = eHtml;
    }

    // Dispatch to active workspace modules
    window.liveViewController?.updateFrame(frame);
    if (this.currentTab === "motors") window.motorsWorkspace?.updateFrame(frame);
    if (this.currentTab === "sensors") window.sensorsWorkspace?.updateFrame(frame);
    if (this.currentTab === "cameras") window.camerasWorkspace?.updateFrame(frame);
    if (this.currentTab === "policy") window.policyWorkspace?.updateFrame(frame);
    if (this.currentTab === "environment") window.environmentWorkspace?.updateFrame(frame);
    if (this.currentTab === "record") window.recordingWorkspace?.updateFrame(frame);
    if (this.currentTab === "raw") window.rawWorkspace?.updateFrame(frame);
  }
}

// Bootstrap
window.addEventListener("DOMContentLoaded", () => {
  window.app = new DashboardApp();
  window.telemetryManager.onFrame((frame) => window.app.updateFrame(frame));
  window.telemetryManager.connect();
});
