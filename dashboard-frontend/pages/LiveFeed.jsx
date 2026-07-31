import React, { useState, useEffect } from "react";
import { AlertCircle, ShieldAlert, Cpu, Video, CheckCircle2, IndianRupee, Eye } from "lucide-react";
import MetricsCard from "../components/MetricsCard";

export default function LiveFeed({ challans, backendStatus }) {
  // State for metrics
  const [metrics, setMetrics] = useState({
    violationsToday: 142,
    finesIssued: 214500,
    activeCameras: 4,
    confidenceAvg: 94.2
  });

  // Local ticker for live alerts simulation
  const [liveDetections, setLiveDetections] = useState([
    {
      id: "det-1",
      plate: "RJ 14 AB 1234",
      violation: "No Helmet",
      time: "21:51:12",
      vehicle: "Motorcycle",
      confidence: 0.94,
    },
    {
      id: "det-2",
      plate: "MH 12 CD 5678",
      violation: "Overspeeding",
      time: "21:49:05",
      vehicle: "Car",
      confidence: 0.89,
    },
    {
      id: "det-3",
      plate: "DL 3C A 5555",
      violation: "Zebra Obstruction",
      time: "21:45:22",
      vehicle: "SUV",
      confidence: 0.92,
    }
  ]);

  const [streamStats, setStreamStats] = useState({
    fps: 29.8,
    trackedVehicles: 12,
    activeViolationsCount: 1
  });

  // Draw simulated tracking boxes on a canvas to make the UI look premium and alive
  useEffect(() => {
    const canvas = document.getElementById("yolo-feed-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationId;
    
    // Setup initial box coordinates
    let car1 = { x: 100, y: 150, w: 120, h: 80, speedX: 2.2, label: "Car #2492 (45km/h)" };
    let car2 = { x: 300, y: 280, w: 140, h: 90, speedX: -1.8, label: "Car #4834 (35km/h)" };
    let bike = { x: 550, y: 220, w: 60, h: 80, speedX: 3.5, label: "Motorcycle #6051 (Violation: No Helmet)" };
    
    const draw = () => {
      // Clear canvas
      ctx.fillStyle = "#111827";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid lanes/boundaries
      ctx.strokeStyle = "rgba(59, 130, 246, 0.2)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 200);
      ctx.lineTo(canvas.width, 200);
      ctx.moveTo(0, 350);
      ctx.lineTo(canvas.width, 350);
      ctx.stroke();

      // Speed Trap boundary lines
      ctx.strokeStyle = "rgba(239, 68, 68, 0.4)"; // Line B
      ctx.beginPath();
      ctx.moveTo(0, 310);
      ctx.lineTo(canvas.width, 310);
      ctx.stroke();
      ctx.fillStyle = "rgba(239, 68, 68, 0.7)";
      ctx.font = "10px Inter";
      ctx.fillText("SPEED TRAP BOUNDARY (LINE B)", 10, 305);

      ctx.strokeStyle = "rgba(59, 130, 246, 0.4)"; // Line A
      ctx.beginPath();
      ctx.moveTo(0, 220);
      ctx.lineTo(canvas.width, 220);
      ctx.stroke();
      ctx.fillStyle = "rgba(59, 130, 246, 0.7)";
      ctx.fillText("SPEED TRAP BOUNDARY (LINE A)", 10, 215);

      // Move and draw box 1
      car1.x += car1.speedX;
      if (car1.x > canvas.width) car1.x = -car1.w;
      ctx.strokeStyle = "#10B981";
      ctx.lineWidth = 2;
      ctx.strokeRect(car1.x, car1.y, car1.w, car1.h);
      ctx.fillStyle = "#10B981";
      ctx.fillRect(car1.x, car1.y - 18, car1.w, 18);
      ctx.fillStyle = "#000";
      ctx.font = "bold 10px Inter";
      ctx.fillText(car1.label, car1.x + 5, car1.y - 5);

      // Move and draw box 2
      car2.x += car2.speedX;
      if (car2.x < -car2.w) car2.x = canvas.width;
      ctx.strokeStyle = "#10B981";
      ctx.lineWidth = 2;
      ctx.strokeRect(car2.x, car2.y, car2.w, car2.h);
      ctx.fillStyle = "#10B981";
      ctx.fillRect(car2.x, car2.y - 18, car2.w, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(car2.label, car2.x + 5, car2.y - 5);

      // Move and draw box 3 (Violator)
      bike.x += bike.speedX;
      if (bike.x > canvas.width) {
        bike.x = -bike.w;
      }
      ctx.strokeStyle = "#EF4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(bike.x, bike.y, bike.w, bike.h);
      ctx.fillStyle = "#EF4444";
      ctx.fillRect(bike.x, bike.y - 18, bike.w, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(bike.label, bike.x + 5, bike.y - 5);

      animationId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, []);

  // Simulate incoming live violation notifications
  useEffect(() => {
    const interval = setInterval(() => {
      // Pick random details
      const plates = ["KA 03 MM 9999", "RJ 14 AB 1234", "MH 12 CD 5678", "DL 3C A 5555", "UP 16 FG 4321"];
      const violations = ["No Helmet", "Overspeeding", "Wrong Way", "Zebra Obstruction"];
      const vehicles = ["Motorcycle", "Car", "SUV", "Truck", "Motorcycle"];
      const fines = [1000, 2000, 2000, 500];

      const randPlate = plates[Math.floor(Math.random() * plates.length)];
      const vIndex = Math.floor(Math.random() * violations.length);
      const randViol = violations[vIndex];
      const randFine = fines[vIndex];
      const randVeh = vehicles[Math.floor(Math.random() * vehicles.length)];

      const newAlert = {
        id: `det-${Date.now()}`,
        plate: randPlate,
        violation: randViol,
        time: new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        vehicle: randVeh,
        confidence: Number((0.85 + Math.random() * 0.14).toFixed(2))
      };

      setLiveDetections((prev) => [newAlert, ...prev.slice(0, 4)]);
      setStreamStats((prev) => ({
        ...prev,
        trackedVehicles: 8 + Math.floor(Math.random() * 8),
        activeViolationsCount: Math.random() > 0.5 ? 1 : 0
      }));
      setMetrics((prev) => ({
        ...prev,
        violationsToday: prev.violationsToday + 1,
        finesIssued: prev.finesIssued + randFine
      }));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 space-y-8 overflow-y-auto max-h-[calc(100vh-4rem)]">
      {/* Top row metrics cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricsCard
          title="Total Violations Today"
          value={metrics.violationsToday}
          changePercentage="+12.4%"
          trend="up"
          icon={ShieldAlert}
          accentColor="red"
        />
        <MetricsCard
          title="Consolidated Fines Issued"
          value={`₹${metrics.finesIssued.toLocaleString("en-IN")}`}
          changePercentage="+8.2%"
          trend="up"
          icon={IndianRupee}
          accentColor="amber"
        />
        <MetricsCard
          title="Active Monitoring Cameras"
          value={`${metrics.activeCameras} / 4`}
          changePercentage="100% Online"
          trend="up"
          icon={Video}
          accentColor="blue"
        />
        <MetricsCard
          title="Avg Detection Confidence"
          value={`${metrics.confidenceAvg}%`}
          changePercentage="+1.1% accuracy"
          trend="up"
          icon={Cpu}
          accentColor="emerald"
        />
      </div>

      {/* Main feed layout grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Live video canvas feed container */}
        <div className="lg:col-span-2 space-y-4">
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2.5">
                <span className="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse"></span>
                <h3 className="font-bold text-white tracking-wide">Live Feed Camera 01 (Main Highway)</h3>
              </div>
              <div className="flex space-x-2">
                <span className="text-xs bg-slate-950/60 px-3 py-1 rounded-md text-gray-400 font-medium">RTSP://192.168.1.100</span>
                <span className="text-xs bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-md font-semibold uppercase">Active</span>
              </div>
            </div>
            
            {/* Interactive YOLO simulator Canvas */}
            <div className="relative aspect-video rounded-xl overflow-hidden border border-gray-800 bg-slate-950">
              <canvas
                id="yolo-feed-canvas"
                width="960"
                height="540"
                className="w-full h-full object-cover"
              />
              {/* Overlay warning banners */}
              {streamStats.activeViolationsCount > 0 && (
                <div className="absolute top-4 right-4 bg-red-600/90 text-white font-bold text-xs uppercase px-3 py-1.5 rounded-lg border border-red-400/30 flex items-center space-x-1.5 shadow-lg shadow-red-600/20 animate-pulse">
                  <AlertCircle className="h-4 w-4" />
                  <span>Violation Flagged</span>
                </div>
              )}
            </div>

            {/* Stream telemetry info bar */}
            <div className="mt-4 grid grid-cols-3 gap-4 border-t border-gray-800/60 pt-4 text-center text-sm text-gray-400 font-semibold select-none">
              <div>
                <span className="block text-xs text-gray-500 uppercase">Processing Rate</span>
                <span className="text-white font-mono mt-1 block">{streamStats.fps} FPS</span>
              </div>
              <div className="border-x border-gray-800/60">
                <span className="block text-xs text-gray-500 uppercase">Tracked In Frame</span>
                <span className="text-white font-mono mt-1 block">{streamStats.trackedVehicles} vehicles</span>
              </div>
              <div>
                <span className="block text-xs text-gray-500 uppercase">Frame Status</span>
                <span className={`font-mono mt-1 block ${streamStats.activeViolationsCount > 0 ? "text-red-400" : "text-emerald-400"}`}>
                  {streamStats.activeViolationsCount > 0 ? "ALERT TRIGGER" : "CLEAR ZONE"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Real-time incident alert side ticker */}
        <div className="space-y-4 flex flex-col h-full">
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md flex flex-col flex-1 max-h-[500px] lg:max-h-none overflow-hidden">
            <h3 className="font-bold text-white tracking-wide mb-4 flex items-center space-x-2">
              <ShieldAlert className="h-5 w-5 text-red-500" />
              <span>Real-Time Incident Ticker</span>
            </h3>

            {/* Scrollable list container */}
            <div className="space-y-4 overflow-y-auto flex-1 pr-1">
              {liveDetections.map((item) => (
                <div
                  key={item.id}
                  className="p-4 rounded-xl bg-slate-950/40 border border-gray-800/80 hover:border-red-500/30 hover:bg-slate-950/60 transition-all duration-200 flex flex-col"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <span className="text-xs font-bold uppercase tracking-wider text-red-400 px-2 py-0.5 rounded bg-red-500/10">
                        {item.violation}
                      </span>
                      <h4 className="font-bold text-white mt-2 text-sm font-mono">{item.plate}</h4>
                    </div>
                    <span className="text-xs font-mono text-gray-500">{item.time}</span>
                  </div>

                  <div className="mt-3 flex items-center justify-between text-xs text-gray-400 border-t border-gray-800/40 pt-2 font-semibold">
                    <span>Type: <strong className="text-gray-200">{item.vehicle}</strong></span>
                    <span>Confidence: <strong className="text-emerald-400">{(item.confidence * 100).toFixed(0)}%</strong></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
