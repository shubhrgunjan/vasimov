/**
 * vasimov/web/client/js/recording.js
 * Flight Recording & Replay Workspace.
 */

class RecordingWorkspace {
  constructor() {
    this.isRecording = false;
    this.recordingStartTime = 0;
    this.recTimerInterval = null;
    this.availableRecordings = [];

    this.initElements();
    this.bindEvents();
    this.refreshRecordings();
  }

  initElements() {
    this.btnRecord = document.getElementById("btnToggleRecord");
    this.recTimerBadge = document.getElementById("recTimerBadge");
    this.recNameInput = document.getElementById("recSessionNameInput");
    this.recFilesOutput = document.getElementById("recFilesOutput");

    this.recsSelect = document.getElementById("replaySelect");
    this.btnLoadReplay = document.getElementById("btnLoadReplay");
    this.btnPlayPause = document.getElementById("btnReplayPlayPause");
    this.btnStepFwd = document.getElementById("btnReplayStepFwd");
    this.btnStepBack = document.getElementById("btnReplayStepBack");
    this.seekSlider = document.getElementById("replaySeekSlider");
    this.replayStatusText = document.getElementById("replayStatusText");
  }

  bindEvents() {
    this.btnRecord?.addEventListener("click", () => {
      if (!this.isRecording) {
        const name = this.recNameInput?.value.trim() || `session_${Math.floor(Date.now() / 1000)}`;
        window.telemetryManager.sendCommand("record", { command: "start", name });
        this.startTimer();
      } else {
        window.telemetryManager.sendCommand("record", { command: "stop" });
        this.stopTimer();
      }
    });

    this.btnLoadReplay?.addEventListener("click", async () => {
      const name = this.recsSelect?.value;
      if (!name) return;
      try {
        const res = await fetch("/api/replay", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "load", name }),
        });
        const data = await res.json();
        if (data.status === "ok") {
          this.replayStatusText.innerText = `LOADED: ${name}`;
          this.refreshRecordings();
        }
      } catch (e) {
        console.error("Replay load error:", e);
      }
    });

    this.btnPlayPause?.addEventListener("click", async () => {
      const isPlay = this.btnPlayPause.innerText.includes("PLAY");
      await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: isPlay ? "play" : "pause" }),
      });
      this.btnPlayPause.innerText = isPlay ? "PAUSE" : "PLAY";
    });

    this.seekSlider?.addEventListener("input", async (e) => {
      const idx = parseInt(e.target.value, 10);
      await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "seek", index: idx }),
      });
    });

    this.btnStepFwd?.addEventListener("click", async () => {
      const cur = parseInt(this.seekSlider?.value || 0, 10);
      this.seekSlider.value = cur + 1;
      await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "seek", index: cur + 1 }),
      });
    });

    this.btnStepBack?.addEventListener("click", async () => {
      const cur = parseInt(this.seekSlider?.value || 0, 10);
      this.seekSlider.value = Math.max(0, cur - 1);
      await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "seek", index: Math.max(0, cur - 1) }),
      });
    });

    document.getElementById("btnRefreshRecordings")?.addEventListener("click", () => {
      this.refreshRecordings();
    });
  }

  startTimer() {
    this.isRecording = true;
    this.recordingStartTime = Date.now();
    this.btnRecord.innerText = "STOP RECORDING";
    this.btnRecord.className = "btn btn-danger";
    this.recTimerBadge.style.display = "inline-flex";

    this.recTimerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
      const mins = Math.floor(elapsed / 60).toString().padStart(2, "0");
      const secs = (elapsed % 60).toString().padStart(2, "0");
      this.recTimerBadge.innerText = `REC ${mins}:${secs}`;
    }, 1000);
  }

  stopTimer() {
    this.isRecording = false;
    clearInterval(this.recTimerInterval);
    this.btnRecord.innerText = "START RECORDING";
    this.btnRecord.className = "btn btn-primary";
    this.recTimerBadge.style.display = "none";
    setTimeout(() => this.refreshRecordings(), 500);
  }

  async refreshRecordings() {
    try {
      const res = await fetch("/api/recordings");
      const list = await res.json();
      this.availableRecordings = list;

      if (this.recsSelect) {
        this.recsSelect.innerHTML = "";
        for (const r of list) {
          const opt = document.createElement("option");
          opt.value = r.name;
          opt.innerText = `${r.filename} (${r.frame_count} frames, ${r.duration_s.toFixed(1)}s)`;
          this.recsSelect.appendChild(opt);
        }
      }
    } catch (e) {
      console.debug("Could not refresh recordings:", e);
    }
  }

  updateFrame(frame) {
    if (frame.replay && frame.replay_data) {
      const rd = frame.replay_data;
      if (this.seekSlider && rd.total_frames) {
        this.seekSlider.max = rd.total_frames - 1;
        this.seekSlider.value = rd.frame_index;
      }
      if (this.replayStatusText) {
        this.replayStatusText.innerText = `FRAME: ${rd.frame_index} / ${rd.total_frames} | REC SIM: ${rd.data?.sim_time?.toFixed(2) || "0.00"}s`;
      }
    }
  }
}

window.recordingWorkspace = new RecordingWorkspace();
