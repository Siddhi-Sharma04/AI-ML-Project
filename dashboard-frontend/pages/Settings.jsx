import React, { useState } from "react";
import { Save, Shield, Video, Network, CheckCircle2 } from "lucide-react";

export default function Settings({ backendUrl, setBackendUrl }) {
  // Input fields state
  const [tempBackendUrl, setTempBackendUrl] = useState(backendUrl);

  const [cameraConfig, setCameraConfig] = useState([
    { id: 1, label: "Highway speed Trap - Camera 01", url: "rtsp://192.168.1.100" },
    { id: 2, label: "Zebra Crossing Crossroad - Camera 02", url: "rtsp://192.168.1.101" },
    { id: 3, label: "Wrong Way City Main - Camera 03", url: "rtsp://192.168.1.102" },
  ]);

  const [rulesConfig, setRulesConfig] = useState({
    speedLimit: 55,
    cooldownWindow: 300,
    fineNoHelmet: 1000,
    fineOverspeeding: 2000,
    fineWrongWay: 2000,
    fineZebraObstruction: 500,
  });

  const [savedStatus, setSavedStatus] = useState(false);

  const handleSaveConfigs = (e) => {
    e.preventDefault();
    setBackendUrl(tempBackendUrl);
    setSavedStatus(true);
    setTimeout(() => setSavedStatus(false), 2500);
  };

  const handleCameraChange = (id, field, value) => {
    setCameraConfig((prev) =>
      prev.map((c) => (c.id === id ? { ...c, [field]: value } : c))
    );
  };

  const handleRuleChange = (field, value) => {
    setRulesConfig((prev) => ({ ...prev, [field]: Number(value) }));
  };

  return (
    <div className="p-8 space-y-6 overflow-y-auto max-h-[calc(100vh-4rem)] max-w-4xl select-none">
      {/* Configuration Form wrapper */}
      <form onSubmit={handleSaveConfigs} className="space-y-6">
        
        {/* API Backend Settings Card */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md space-y-4">
          <h3 className="font-bold text-white tracking-wide text-lg flex items-center space-x-2">
            <Network className="h-5 w-5 text-blue-500" />
            <span>API & Network Configuration</span>
          </h3>
          <div className="grid grid-cols-1 gap-4">
            <div className="flex flex-col">
              <label className="text-xs text-gray-500 font-bold uppercase mb-2">Backend Connection API URL</label>
              <input
                type="text"
                value={tempBackendUrl}
                onChange={(e) => setTempBackendUrl(e.target.value)}
                className="bg-slate-950/40 border border-gray-800 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>
        </div>

        {/* Camera stream configuration list */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md space-y-4">
          <h3 className="font-bold text-white tracking-wide text-lg flex items-center space-x-2">
            <Video className="h-5 w-5 text-blue-500" />
            <span>Camera Stream Configurations</span>
          </h3>
          
          <div className="space-y-4">
            {cameraConfig.map((camera) => (
              <div key={camera.id} className="grid grid-cols-1 md:grid-cols-2 gap-4 border-b border-gray-800/40 pb-4 last:border-b-0 last:pb-0">
                <div className="flex flex-col">
                  <label className="text-xs text-gray-500 font-bold uppercase mb-2">Camera Location Label</label>
                  <input
                    type="text"
                    value={camera.label}
                    onChange={(e) => handleCameraChange(camera.id, "label", e.target.value)}
                    className="bg-slate-950/40 border border-gray-800 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div className="flex flex-col">
                  <label className="text-xs text-gray-500 font-bold uppercase mb-2">RTSP Stream Address</label>
                  <input
                    type="text"
                    value={camera.url}
                    onChange={(e) => handleCameraChange(camera.id, "url", e.target.value)}
                    className="bg-slate-950/40 border border-gray-800 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Rules & Fines configurations */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md space-y-4">
          <h3 className="font-bold text-white tracking-wide text-lg flex items-center space-x-2">
            <Shield className="h-5 w-5 text-blue-500" />
            <span>System Enforcement Penal Rules</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Speed limit */}
            <div className="flex flex-col">
              <label className="text-xs text-gray-500 font-bold uppercase mb-2">Speed Trap Limit (KM/H)</label>
              <input
                type="number"
                value={rulesConfig.speedLimit}
                onChange={(e) => handleRuleChange("speedLimit", e.target.value)}
                className="bg-slate-950/40 border border-gray-800 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
              />
            </div>
            {/* Cooldown */}
            <div className="flex flex-col">
              <label className="text-xs text-gray-500 font-bold uppercase mb-2">Deduplication Cooldown window (Sec)</label>
              <input
                type="number"
                value={rulesConfig.cooldownWindow}
                onChange={(e) => handleRuleChange("cooldownWindow", e.target.value)}
                className="bg-slate-950/40 border border-gray-800 rounded-lg px-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
              />
            </div>
          </div>

          <div className="border-t border-gray-800/60 my-4 pt-4 space-y-2">
            <span className="text-xs text-gray-500 font-bold uppercase tracking-wider block">Fines & Penalties Setup (INR)</span>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="flex flex-col">
                <label className="text-[10px] text-gray-500 font-bold uppercase mb-1.5">No Helmet</label>
                <input
                  type="number"
                  value={rulesConfig.fineNoHelmet}
                  onChange={(e) => handleRuleChange("fineNoHelmet", e.target.value)}
                  className="bg-slate-950/40 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
                />
              </div>
              <div className="flex flex-col">
                <label className="text-[10px] text-gray-500 font-bold uppercase mb-1.5">Overspeeding</label>
                <input
                  type="number"
                  value={rulesConfig.fineOverspeeding}
                  onChange={(e) => handleRuleChange("fineOverspeeding", e.target.value)}
                  className="bg-slate-950/40 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
                />
              </div>
              <div className="flex flex-col">
                <label className="text-[10px] text-gray-500 font-bold uppercase mb-1.5">Wrong Way</label>
                <input
                  type="number"
                  value={rulesConfig.fineWrongWay}
                  onChange={(e) => handleRuleChange("fineWrongWay", e.target.value)}
                  className="bg-slate-950/40 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
                />
              </div>
              <div className="flex flex-col">
                <label className="text-[10px] text-gray-500 font-bold uppercase mb-1.5">Zebra Crossing</label>
                <input
                  type="number"
                  value={rulesConfig.fineZebraObstruction}
                  onChange={(e) => handleRuleChange("fineZebraObstruction", e.target.value)}
                  className="bg-slate-950/40 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500/50 font-mono"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center space-x-4">
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold text-sm px-6 py-3 rounded-xl flex items-center space-x-2 transition-colors shadow-lg shadow-blue-600/10 cursor-pointer"
          >
            <Save className="h-4.5 w-4.5" />
            <span>Save System Configurations</span>
          </button>
          
          {savedStatus && (
            <div className="text-emerald-400 text-sm font-semibold flex items-center space-x-1.5 animate-bounce">
              <CheckCircle2 className="h-5 w-5" />
              <span>Configurations saved successfully!</span>
            </div>
          )}
        </div>

      </form>
    </div>
  );
}
