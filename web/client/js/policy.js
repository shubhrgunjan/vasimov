/**
 * vasimov/web/client/js/policy.js
 * Locomotion Policy Workspace: 78-D Observation, 23-D Actions, and Pipeline Trace.
 */

class PolicyWorkspace {
  constructor() {
    this.selectedTraceJoint = "left_knee_joint";
    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.obsTableBody = document.getElementById("policyObsTableBody");
    this.actTableBody = document.getElementById("policyActTableBody");
    this.traceJointSelect = document.getElementById("policyTraceJointSelect");
  }

  bindEvents() {
    if (this.traceJointSelect) {
      this.traceJointSelect.addEventListener("change", (e) => {
        this.selectedTraceJoint = e.target.value;
        this.renderTrace(window.telemetryManager.latestFrame);
      });
    }
  }

  updateFrame(frame) {
    if (!frame || !frame.policy) return;

    // Contract Status
    document.getElementById("policyObsCount").innerText = `${frame.policy.input_dim} inputs`;
    document.getElementById("policyActCount").innerText = `${frame.policy.output_dim} outputs`;
    document.getElementById("policyInferenceTime").innerText = `${frame.policy.timing?.inference_ms || 1.25} ms`;

    this.renderObsTable(frame.policy);
    this.renderActTable(frame.policy);
    this.renderTrace(frame);
  }

  renderObsTable(policy) {
    if (!this.obsTableBody || !policy.observation) return;

    const raw = policy.observation;
    const labels = policy.observation_labels || [];
    let html = "";

    for (let i = 0; i < raw.length; i++) {
      const val = raw[i];
      const lbl = labels[i] || `obs_${i}`;
      let group = "OTHER";
      if (i < 3) group = "BASE GYRO";
      else if (i < 6) group = "PROJ GRAVITY";
      else if (i < 9) group = "COMMAND";
      else if (i < 32) group = "JOINT POS";
      else if (i < 55) group = "JOINT VEL";
      else group = "PREV ACTION";

      html += `
        <tr>
          <td class="num">${i}</td>
          <td><span class="badge" style="font-size: 8px;">${group}</span></td>
          <td><strong>${lbl}</strong></td>
          <td class="num" style="font-weight: 700; color: ${val < 0 ? "#f87171" : "#60a5fa"};">${val.toFixed(4)}</td>
        </tr>
      `;
    }
    this.obsTableBody.innerHTML = html;
  }

  renderActTable(policy) {
    if (!this.actTableBody || !policy.action_items) return;

    let html = "";
    for (let i = 0; i < policy.action_items.length; i++) {
      const it = policy.action_items[i];
      html += `
        <tr>
          <td class="num">${i}</td>
          <td><strong>${it.joint}</strong></td>
          <td class="num">${it.raw_action.toFixed(3)}</td>
          <td class="num">${it.scaled_delta.toFixed(4)}</td>
          <td class="num">${it.default_pos.toFixed(3)}</td>
          <td class="num" style="font-weight: 700; color: #38bdf8;">${it.desired_target.toFixed(4)}</td>
          <td class="num">${it.kp.toFixed(0)}</td>
          <td class="num">${it.kd.toFixed(1)}</td>
        </tr>
      `;
    }
    this.actTableBody.innerHTML = html;

    // Populate joint trace selector if empty
    if (this.traceJointSelect && this.traceJointSelect.options.length <= 1) {
      this.traceJointSelect.innerHTML = "";
      for (const it of policy.action_items) {
        const opt = document.createElement("option");
        opt.value = it.joint;
        opt.innerText = it.joint;
        if (it.joint === this.selectedTraceJoint) opt.selected = true;
        this.traceJointSelect.appendChild(opt);
      }
    }
  }

  renderTrace(frame) {
    if (!frame || !frame.policy || !frame.joints) return;

    const jName = this.selectedTraceJoint;
    const actItem = (frame.policy.action_items || []).find((a) => a.joint === jName);
    const jointItem = frame.joints.find((j) => j.name === jName);

    if (actItem && jointItem) {
      document.getElementById("traceRawAction").innerText = `${actItem.raw_action.toFixed(4)}`;
      document.getElementById("traceScaledDelta").innerText = `${actItem.scaled_delta.toFixed(4)} rad`;
      document.getElementById("traceDefaultOffset").innerText = `${actItem.default_pos.toFixed(4)} rad`;
      document.getElementById("traceDesiredTarget").innerText = `${actItem.desired_target.toFixed(4)} rad`;
      document.getElementById("traceActualPos").innerText = `${jointItem.position.toFixed(4)} rad`;
      document.getElementById("tracePosError").innerText = `${(jointItem.target - jointItem.position).toFixed(4)} rad`;
      document.getElementById("traceTorque").innerText = `${jointItem.torque.toFixed(2)} Nm`;
    }
  }
}

window.policyWorkspace = new PolicyWorkspace();
