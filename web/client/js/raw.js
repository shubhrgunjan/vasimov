/**
 * vasimov/web/client/js/raw.js
 * RAW Data Inspector, Model Metadata, and Export.
 */

class RawWorkspace {
  constructor() {
    this.paused = false;
    this.latestSnapshot = "{}";

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.rawPre = document.getElementById("rawJsonPre");
    this.btnCopy = document.getElementById("btnCopyRawJson");
    this.btnDownload = document.getElementById("btnDownloadRawJson");
    this.btnPause = document.getElementById("btnPauseRawStream");
  }

  bindEvents() {
    this.btnCopy?.addEventListener("click", () => {
      navigator.clipboard.writeText(this.latestSnapshot);
      this.btnCopy.innerText = "COPIED!";
      setTimeout(() => (this.btnCopy.innerText = "COPY JSON"), 1500);
    });

    this.btnDownload?.addEventListener("click", () => {
      const blob = new Blob([this.latestSnapshot], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `vasimov_frame_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });

    this.btnPause?.addEventListener("click", () => {
      this.paused = !this.paused;
      this.btnPause.innerText = this.paused ? "RESUME STREAM" : "PAUSE STREAM";
      this.btnPause.className = this.paused ? "btn btn-primary" : "btn";
    });
  }

  updateFrame(frame) {
    if (!frame) return;

    // Model Metadata Card
    const modelMeta = frame.diagnostics?.provenance?.model || {};
    document.getElementById("metaModelBodies").innerText = modelMeta.bodies ?? "-";
    document.getElementById("metaModelJoints").innerText = modelMeta.joints ?? "-";
    document.getElementById("metaModelActuators").innerText = modelMeta.actuators ?? "-";
    document.getElementById("metaModelSensors").innerText = modelMeta.sensors ?? "-";
    document.getElementById("metaModelCameras").innerText = modelMeta.cameras ?? "-";
    document.getElementById("metaModelGeoms").innerText = modelMeta.geoms ?? "-";

    if (this.paused) return;

    this.latestSnapshot = JSON.stringify(frame, null, 2);
    if (this.rawPre) {
      this.rawPre.innerText = this.latestSnapshot;
    }
  }
}

window.rawWorkspace = new RawWorkspace();
