/**
* vasimov/web/client/js/telemetry.js
* WebSocket Telemetry Manager & Command Dispatcher.
*/

class TelemetryManager {
  constructor() {
    this.ws = null;
    this.connected = false;
    this.latestFrame = null;
    this.lastFrameWallTime = 0;
    this.reconnectTimer = null;
    this.listeners = [];
    this.pendingCommands = new Map();
    this.staleThresholdMs = 1500;

    // Monitor for stale data every 500ms
    setInterval(() => this.checkStale(), 500);
  }

  connect() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname || "127.0.0.1";
    // Connect to WebSocket port 8854 (or default)
    const wsUrl = `${proto}//${host}:8854`;

    this.updateStatus("connecting", "CONNECTING");

    try {
      this.ws = new WebSocket(wsUrl);
    } catch (e) {
      this.updateStatus("disconnected", "WS FAIL");
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.updateStatus("connected", "CONNECTED");
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "ack") {
          const ack = msg.ack;
          if (ack && ack.command_id && this.pendingCommands.has(ack.command_id)) {
            const resolver = this.pendingCommands.get(ack.command_id);
            resolver(ack);
            this.pendingCommands.delete(ack.command_id);
          }
          if (window.onCommandAck) {
            window.onCommandAck(ack);
          }
          return;
        }

        // Canonical TelemetryFrame
        this.latestFrame = msg;
        this.lastFrameWallTime = Date.now();
        this.updateStatus("connected", "CONNECTED");

        for (const cb of this.listeners) {
          try {
            cb(msg);
          } catch (err) {
            console.error("Telemetry listener error:", err);
          }
        }
      } catch (err) {
        console.error("Frame parse error:", err);
      }
    };

    this.ws.onerror = () => {
      this.updateStatus("disconnected", "ERROR");
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.updateStatus("disconnected", "DISCONNECTED");
      this.scheduleReconnect();
    };
  }

  scheduleReconnect() {
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), 2000);
  }

  checkStale() {
    if (this.connected && this.lastFrameWallTime > 0) {
      const elapsed = Date.now() - this.lastFrameWallTime;
      if (elapsed > this.staleThresholdMs) {
        this.updateStatus("stale", "STALE");
      }
    }
  }

  updateStatus(state, text) {
    const badge = document.getElementById("connectionBadge");
    const label = document.getElementById("connectionText");
    if (badge && label) {
      badge.className = `status-badge ${state}`;
      label.innerText = text;
    }
  }

  onFrame(callback) {
    this.listeners.push(callback);
  }

  async sendCommand(action, params = {}) {
    // Try sending over active WebSocket
    if (this.connected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      return new Promise((resolve) => {
        const cmdId = Date.now();
        this.pendingCommands.set(cmdId, resolve);
        this.ws.send(JSON.stringify({ action, params, command_id: cmdId }));
        // Timeout fallback after 3s
        setTimeout(() => {
          if (this.pendingCommands.has(cmdId)) {
            this.pendingCommands.delete(cmdId);
            resolve({ status: "timeout", action });
          }
        }, 3000);
      });
    }

    // Fallback: send via REST POST /api/control
    try {
      const res = await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, params }),
      });
      const data = await res.json();
      if (window.onCommandAck) window.onCommandAck(data);
      return data;
    } catch (e) {
      const errAck = { status: "error", action, lifecycle: { validation: e.message } };
      if (window.onCommandAck) window.onCommandAck(errAck);
      return errAck;
    }
  }
}

window.telemetryManager = new TelemetryManager();
