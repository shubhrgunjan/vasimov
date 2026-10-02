/**
 * vasimov/web/client/js/environment.js
 * Environment & Scenarios Workspace: Preset loading, friction, mass scale, and obstacle inspection.
 */

class EnvironmentWorkspace {
  constructor() {
    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.presetSelect = document.getElementById("envPresetSelect");
    this.btnLoadEnv = document.getElementById("btnLoadEnv");
    this.btnResetEnv = document.getElementById("btnResetEnv");
    this.btnPush = document.getElementById("btnApplyPush");
    this.obstaclesTableBody = document.getElementById("envObstaclesTableBody");
  }

  bindEvents() {
    this.btnLoadEnv?.addEventListener("click", () => {
      const preset = this.presetSelect?.value || "flat";
      window.telemetryManager.sendCommand("environment", { preset });
    });

    this.btnResetEnv?.addEventListener("click", () => {
      window.telemetryManager.sendCommand("reset", { seed: 42 });
    });

    this.btnPush?.addEventListener("click", () => {
      const force = parseFloat(document.getElementById("pushForceInput")?.value || 40.0);
      const dir = document.getElementById("pushDirSelect")?.value || "x";
      const duration = parseFloat(document.getElementById("pushDurInput")?.value || 0.1);
      window.telemetryManager.sendCommand("push", { force, dir, duration });
    });
  }

  updateFrame(frame) {
    if (!frame || !frame.environment) return;
    const env = frame.environment;

    document.getElementById("envCurrentName").innerText = env.name.toUpperCase();
    document.getElementById("envCurrentDesc").innerText = env.description || "Nominal flat ground";
    document.getElementById("envFriction").innerText = env.ground_friction?.toFixed(2) || "1.00";
    document.getElementById("envMassScale").innerText = env.mass_scale?.toFixed(2) || "1.00";
    document.getElementById("envGravity").innerText = `[${(env.gravity || [0, 0, -9.81]).map((g) => g.toFixed(2)).join(", ")}]`;
    document.getElementById("envObstacleCount").innerText = env.obstacle_count || 0;

    // Populate presets selector if not already populated
    if (this.presetSelect && this.presetSelect.options.length <= 1 && env.available_presets) {
      this.presetSelect.innerHTML = "";
      for (const p of env.available_presets) {
        const opt = document.createElement("option");
        opt.value = p;
        opt.innerText = p;
        if (p === env.name) opt.selected = true;
        this.presetSelect.appendChild(opt);
      }
    }

    // Render obstacles
    if (this.obstaclesTableBody) {
      if (!env.obstacles || env.obstacles.length === 0) {
        this.obstaclesTableBody.innerHTML = `<tr><td colspan="4" style="color: var(--text-dim); text-align: center;">No active obstacles in current environment</td></tr>`;
      } else {
        let html = "";
        for (const obs of env.obstacles) {
          html += `
            <tr>
              <td><strong>${obs.name}</strong></td>
              <td><span class="badge">${obs.type}</span></td>
              <td class="num">[${(obs.pos || []).map((x) => x.toFixed(2)).join(", ")}]</td>
              <td class="num">[${(obs.size || []).map((x) => x.toFixed(3)).join(", ")}]</td>
            </tr>
          `;
        }
        this.obstaclesTableBody.innerHTML = html;
      }
    }
  }
}

window.environmentWorkspace = new EnvironmentWorkspace();
