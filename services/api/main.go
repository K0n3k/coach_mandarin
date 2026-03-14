package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ── WebSocket upgrader ──────────────────────────────────────

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// ── Training hub ────────────────────────────────────────────
// Trainer connects as "source", browsers connect as "viewers".
// Events from trainer are relayed to all viewers.

type trainingHub struct {
	mu        sync.RWMutex
	viewers   map[*websocket.Conn]struct{}
	source    *websocket.Conn
	sim       bool   // true if source is the internal simulator
	lastCfg   []byte // last config event — replayed to new viewers
	lastStep  []byte // last step event — replayed to new viewers
}

var hub = &trainingHub{
	viewers: make(map[*websocket.Conn]struct{}),
}

func (h *trainingHub) addViewer(c *websocket.Conn) {
	h.mu.Lock()
	h.viewers[c] = struct{}{}
	h.mu.Unlock()
}

func (h *trainingHub) removeViewer(c *websocket.Conn) {
	h.mu.Lock()
	delete(h.viewers, c)
	h.mu.Unlock()
	c.Close()
}

func (h *trainingHub) setSource(c *websocket.Conn) {
	h.mu.Lock()
	if h.source != nil {
		h.source.Close()
	}
	h.source = c
	h.sim = false
	h.mu.Unlock()
}

func (h *trainingHub) setSimSource() {
	h.mu.Lock()
	if h.source != nil {
		h.source.Close()
	}
	h.source = nil
	h.sim = true
	h.mu.Unlock()
}

func (h *trainingHub) clearSource(c *websocket.Conn) {
	h.mu.Lock()
	if h.source == c {
		h.source = nil
	}
	h.mu.Unlock()
}

func (h *trainingHub) clearSim() {
	h.mu.Lock()
	h.sim = false
	h.mu.Unlock()
}

func (h *trainingHub) broadcast(msg []byte) {
	h.mu.Lock()
	// Snapshot config and step events for late-joining viewers
	var peek struct{ Type string `json:"type"` }
	if json.Unmarshal(msg, &peek) == nil {
		switch peek.Type {
		case "config":
			h.lastCfg = append([]byte(nil), msg...)
		case "step":
			h.lastStep = append([]byte(nil), msg...)
		case "trainer_disconnected":
			h.lastCfg = nil
			h.lastStep = nil
		}
	}
	h.mu.Unlock()

	h.mu.RLock()
	defer h.mu.RUnlock()
	for c := range h.viewers {
		if err := c.WriteMessage(websocket.TextMessage, msg); err != nil {
			go h.removeViewer(c)
		}
	}
}

// replayTo sends cached config + last step to a newly connected viewer.
func (h *trainingHub) replayTo(c *websocket.Conn) {
	h.mu.RLock()
	cfg := h.lastCfg
	step := h.lastStep
	h.mu.RUnlock()
	if cfg != nil {
		c.WriteMessage(websocket.TextMessage, cfg)
	}
	if step != nil {
		c.WriteMessage(websocket.TextMessage, step)
	}
}

func (h *trainingHub) viewerCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.viewers)
}

func (h *trainingHub) hasSource() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.source != nil || h.sim
}

// ── HTTP handlers ───────────────────────────────────────────

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprint(w, `{"status":"ok"}`)
}

func handleAPIHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	src := hub.hasSource()
	vc := hub.viewerCount()

	resp := map[string]any{
		"status":       "ok",
		"trainer":      src,
		"viewer_count": vc,
	}
	json.NewEncoder(w).Encode(resp)
}

// /ws/training?role=trainer  — trainer pushes events
// /ws/training?role=viewer   — browser receives events (default)
// /ws/training               — browser (viewer)
func handleTrainingWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws upgrade: %v", err)
		return
	}

	role := r.URL.Query().Get("role")

	if role == "trainer" {
		handleTrainerConn(conn)
	} else {
		handleViewerConn(conn)
	}
}

func handleTrainerConn(conn *websocket.Conn) {
	hub.setSource(conn)
	log.Println("trainer connected")

	// Notify viewers
	hub.broadcast([]byte(`{"type":"trainer_connected"}`))

	defer func() {
		hub.clearSource(conn)
		conn.Close()
		log.Println("trainer disconnected")
		hub.broadcast([]byte(`{"type":"trainer_disconnected"}`))
	}()

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}
		hub.broadcast(msg)
	}
}

func handleViewerConn(conn *websocket.Conn) {
	hub.addViewer(conn)
	log.Printf("viewer connected (%d total)", hub.viewerCount())

	defer func() {
		hub.removeViewer(conn)
		log.Printf("viewer disconnected (%d remaining)", hub.viewerCount())
	}()

	// Send current status
	status := "idle"
	if hub.hasSource() {
		status = "running"
	}
	conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf(
		`{"type":"status","status":"%s"}`, status,
	)))

	// Replay last config + step so late-joining viewers get full state
	hub.replayTo(conn)

	// Keep connection alive — read pongs / detect close
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

// ── Simulation ──────────────────────────────────────────────
// Activated by SIM=1 env var. Produces realistic training events
// that flow through the hub exactly like a real trainer would.

var simStop chan struct{}

func startSimulation() {
	simStop = make(chan struct{})
	hub.setSimSource()
	hub.broadcast([]byte(`{"type":"trainer_connected"}`))
	log.Println("simulation started")

	go runSimulation(simStop)
}

func stopSimulation() {
	if simStop != nil {
		close(simStop)
		simStop = nil
	}
	hub.clearSim()
	hub.broadcast([]byte(`{"type":"trainer_disconnected"}`))
	log.Println("simulation stopped")
}

func handleSimAPI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	action := r.URL.Query().Get("action")
	switch action {
	case "start":
		if hub.hasSource() {
			w.WriteHeader(409)
			fmt.Fprint(w, `{"error":"trainer already connected"}`)
			return
		}
		startSimulation()
		fmt.Fprint(w, `{"status":"started"}`)
	case "stop":
		stopSimulation()
		fmt.Fprint(w, `{"status":"stopped"}`)
	default:
		w.WriteHeader(400)
		fmt.Fprint(w, `{"error":"use ?action=start or ?action=stop"}`)
	}
}

func runSimulation(stop chan struct{}) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	runID := fmt.Sprintf("sim-%d", time.Now().Unix())

	// Config matching realistic Phase 2 training
	totalEpochs := 20
	stepsPerEpoch := 34423
	batchSize := 64
	baseLR := 5e-4
	warmupSteps := int(float64(totalEpochs*stepsPerEpoch) * 0.03)

	config := map[string]any{
		"type":   "config",
		"run_id": runID,
		"config": map[string]any{
			"model_name":   "mel-cnn-bigru",
			"phase":        2,
			"device":       "cuda:0",
			"amp":          true,
			"batch_size":   batchSize,
			"lr":           baseLR,
			"total_epochs": totalEpochs,
			"warmup_steps": warmupSteps,
			"datasets": []map[string]any{
				{"name": "TTS-ph1", "active": true, "phases": []int{1, 3}, "samples": 8136},
				{"name": "AISHELL-1", "active": true, "phases": []int{2, 3}, "samples": 120098},
				{"name": "THCHS-30", "active": true, "phases": []int{2}, "samples": 10000},
				{"name": "CV zh-CN", "active": true, "phases": []int{2}, "samples": 29000},
				{"name": "iCALL", "active": false, "phases": []int{3}},
				{"name": "LATIC", "active": false, "phases": []int{3}},
			},
			"speakers":     460,
			"total_samples": 167234,
			"val_samples":   16723,
		},
	}

	configJSON, _ := json.Marshal(config)
	hub.broadcast(configJSON)

	// Simulation state
	loss := 3.5 + rng.Float64()*0.5 // start loss ~3.5-4.0
	gpuTemp := 45.0
	vramUsed := 5.8 + rng.Float64()*0.4

	// Loss moving average buffer
	lossMA := make([]float64, 0, 5)

	// Emit step events at ~100 step intervals, ~300ms real time
	tickInterval := 300 * time.Millisecond
	stepsPerTick := 100
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	globalStep := 0

	for epoch := 1; epoch <= totalEpochs; epoch++ {
		for step := stepsPerTick; step <= stepsPerEpoch; step += stepsPerTick {
			select {
			case <-stop:
				return
			case <-ticker.C:
			}

			globalStep++

			// Learning rate schedule: linear warmup then cosine decay
			totalGlobalSteps := totalEpochs * stepsPerEpoch / stepsPerTick
			currentGlobalStep := (epoch-1)*(stepsPerEpoch/stepsPerTick) + step/stepsPerTick
			warmupTicks := warmupSteps / stepsPerTick
			lr := baseLR
			if currentGlobalStep < warmupTicks {
				lr = baseLR * float64(currentGlobalStep) / float64(warmupTicks)
			} else {
				progress := float64(currentGlobalStep-warmupTicks) / float64(totalGlobalSteps-warmupTicks)
				lr = baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
			}

			// Loss: decays over epochs with noise
			epochProgress := float64(epoch-1) / float64(totalEpochs)
			targetLoss := 3.5*math.Exp(-2.5*epochProgress) + 0.35
			loss = loss + (targetLoss-loss)*0.02 + (rng.Float64()-0.5)*0.08

			lossMA = append(lossMA, loss)
			if len(lossMA) > 5 {
				lossMA = lossMA[len(lossMA)-5:]
			}
			var ma5 float64
			for _, v := range lossMA {
				ma5 += v
			}
			ma5 /= float64(len(lossMA))

			// Speed: ~8-11 batch/s with small noise
			speed := 9.0 + rng.Float64()*2.0 - 0.5

			// Grad norm: noisy, correlates loosely with loss
			gradNorm := 0.3 + loss*0.2 + (rng.Float64()-0.5)*0.15

			// ETA in seconds — speed is batch/s, each tick = stepsPerTick steps
			remainStepsEpoch := stepsPerEpoch - step
			etaEpoch := float64(remainStepsEpoch) / speed
			remainStepsGlobal := (totalEpochs-epoch)*stepsPerEpoch + remainStepsEpoch
			etaGlobal := float64(remainStepsGlobal) / speed

			// GPU: small fluctuations
			gpuUtil := 88 + rng.Intn(10)
			gpuTemp = gpuTemp + (72.0-gpuTemp)*0.05 + (rng.Float64()-0.5)*0.5
			gpuTemp = math.Max(45, math.Min(85, gpuTemp))
			vramUsed = vramUsed + (6.4-vramUsed)*0.01 + (rng.Float64()-0.5)*0.02
			gpuPower := 165 + rng.Intn(30)

			stepEvt := map[string]any{
				"type":            "step",
				"run_id":          runID,
				"epoch":           epoch,
				"step":            step,
				"steps_per_epoch": stepsPerEpoch,
				"loss":            math.Round(loss*10000) / 10000,
				"loss_ma5":        math.Round(ma5*10000) / 10000,
				"speed_bps":       math.Round(speed*10) / 10,
				"grad_norm":       math.Round(gradNorm*1000) / 1000,
				"lr":              lr,
				"eta_epoch_s":     math.Round(etaEpoch),
				"eta_global_s":    math.Round(etaGlobal),
				"gpu": map[string]any{
					"util_pct":     gpuUtil,
					"temp_c":       math.Round(gpuTemp),
					"vram_used_gb": math.Round(vramUsed*10) / 10,
					"vram_total_gb": 8.0,
					"power_w":      gpuPower,
				},
			}

			msg, _ := json.Marshal(stepEvt)
			hub.broadcast(msg)
		}

		// Epoch end
		select {
		case <-stop:
			return
		default:
		}

		epochProgress := float64(epoch) / float64(totalEpochs)

		// Training metrics improve over epochs
		lossTrain := 3.5*math.Exp(-2.5*epochProgress) + 0.35 + (rng.Float64()-0.5)*0.05
		lossVal := lossTrain + 0.02 + rng.Float64()*0.06 // val slightly worse
		accTrain := 20 + 65*epochProgress + (rng.Float64()-0.5)*3
		accVal := accTrain - 2 - rng.Float64()*3

		// Tone accuracies: T4 learns fastest, T3/T5 slowest
		toneAccs := map[string]any{
			"T1": math.Min(99, 15+55*epochProgress+(rng.Float64()-0.5)*4),
			"T2": math.Min(99, 10+50*epochProgress+(rng.Float64()-0.5)*4),
			"T3": math.Min(99, 2+35*epochProgress+(rng.Float64()-0.5)*3),
			"T4": math.Min(99, 40+50*epochProgress+(rng.Float64()-0.5)*3),
			"T5": math.Min(99, 1+20*epochProgress+(rng.Float64()-0.5)*2),
		}

		isBest := epoch == 1 || accVal > accTrain-5+float64(epoch)*2

		epochEvt := map[string]any{
			"type":      "epoch_end",
			"run_id":    runID,
			"epoch":     epoch,
			"loss_train": math.Round(lossTrain*1000) / 1000,
			"loss_val":   math.Round(lossVal*1000) / 1000,
			"acc_train":  math.Round(accTrain*10) / 10,
			"acc_val":    math.Round(accVal*10) / 10,
			"tone_accs":  toneAccs,
			"is_best":    isBest,
		}

		msg, _ := json.Marshal(epochEvt)
		hub.broadcast(msg)

		if isBest {
			ckpt := map[string]any{
				"type":    "checkpoint",
				"run_id":  runID,
				"epoch":   epoch,
				"path":    fmt.Sprintf("/checkpoints/wavlm-scorer-ph2-ep%d-acc%.1f.pt", epoch, accVal),
				"val_acc": math.Round(accVal*10) / 10,
				"is_best": true,
			}
			ckptMsg, _ := json.Marshal(ckpt)
			hub.broadcast(ckptMsg)
		}

		log.Printf("sim: epoch %d done — loss_val=%.3f acc_val=%.1f", epoch, lossVal, accVal)
	}

	// Training done
	hub.clearSim()
	hub.broadcast([]byte(`{"type":"trainer_disconnected"}`))
	log.Println("simulation completed (all epochs)")
}

// ── Main ────────────────────────────────────────────────────

func main() {
	http.HandleFunc("/health", handleHealth)
	http.HandleFunc("/api/health", handleAPIHealth)
	http.HandleFunc("/ws/training", handleTrainingWS)
	http.HandleFunc("/api/sim", handleSimAPI)

	// Legacy /ws stub
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"stub"}`)
	})

	// Auto-start simulation if SIM=1
	if os.Getenv("SIM") == "1" {
		log.Println("SIM=1 detected — starting simulation on boot")
		startSimulation()
	}

	log.Println("api listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal("fatal:", err)
	}
}
