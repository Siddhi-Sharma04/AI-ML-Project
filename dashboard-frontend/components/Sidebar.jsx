import React from "react";
import { Camera, FileText, Settings as SettingsIcon, Shield, Film } from "lucide-react";

export default function Sidebar({ activePage, setActivePage, backendStatus }) {
  const menuItems = [
    { id: "LiveFeed", label: "Live Feed", icon: Camera },
    { id: "ViolationsLog", label: "Violations Log", icon: FileText },
    { id: "VideoAnalysis", label: "Video Analysis", icon: Film },
    { id: "Settings", label: "System Settings", icon: SettingsIcon },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-gray-800 flex flex-col h-screen select-none">
      {/* Brand Header */}
      <div className="p-6 border-b border-gray-800 flex items-center space-x-3">
        <div className="bg-blue-600 p-2 rounded-lg text-white">
          <Shield className="h-6 w-6" />
        </div>
        <div>
          <h1 className="font-bold text-lg leading-tight text-white">Traffic AI</h1>
          <span className="text-xs text-gray-400 font-medium">Control Dashboard</span>
        </div>
      </div>

      {/* Nav Menu */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                  : "text-gray-400 hover:bg-gray-800 hover:text-white"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Connection & Status Panel */}
      <div className="p-4 border-t border-gray-800 bg-slate-950/50">
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">AI Inference Engine</span>
            <div className="flex items-center space-x-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-emerald-400 font-semibold uppercase">Online</span>
            </div>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">Database & Backend</span>
            <div className="flex items-center space-x-1.5">
              <span className={`h-2.5 w-2.5 rounded-full ${backendStatus ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`}></span>
              <span className={`font-semibold uppercase ${backendStatus ? "text-emerald-400" : "text-rose-400"}`}>
                {backendStatus ? "Connected" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
