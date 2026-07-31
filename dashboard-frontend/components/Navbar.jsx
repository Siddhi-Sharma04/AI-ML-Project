import React, { useState, useEffect } from "react";
import { Search, Bell, User, Clock, ChevronDown } from "lucide-react";

export default function Navbar({ activePage }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const getPageTitle = () => {
    switch (activePage) {
      case "LiveFeed":
        return "Live Traffic Monitor";
      case "ViolationsLog":
        return "Violations & Challans Logs";
      case "VideoAnalysis":
        return "Video Upload & Clip Analysis";
      case "Settings":
        return "System Settings & Rules";
      default:
        return "Traffic Intelligence System";
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
  };

  const formatDate = (date) => {
    return date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  };

  return (
    <header className="h-16 bg-slate-900/80 border-b border-gray-800 flex items-center justify-between px-8 backdrop-blur-md sticky top-0 z-30 select-none">
      {/* Page Title Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-wide">{getPageTitle()}</h2>
      </div>

      {/* Navigation Right Actions */}
      <div className="flex items-center space-x-6">
        {/* Real-time clock widget */}
        <div className="flex items-center space-x-2 bg-slate-950/40 px-4 py-1.5 rounded-lg border border-gray-800 text-sm text-gray-300 font-medium">
          <Clock className="h-4 w-4 text-blue-500" />
          <span>{formatDate(time)}</span>
          <span className="text-gray-600">|</span>
          <span className="font-mono">{formatTime(time)}</span>
        </div>

        {/* Search Input bar */}
        <div className="relative w-64">
          <Search className="absolute left-3.5 top-2.5 h-4 w-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search license plate..."
            className="w-full bg-slate-950/40 border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50 transition-colors"
          />
        </div>

        {/* Notification Bell Icon */}
        <button className="relative bg-slate-950/40 p-2.5 rounded-lg border border-gray-800 text-gray-400 hover:text-white transition-colors">
          <Bell className="h-4.5 w-4.5" />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-red-500 ring-2 ring-slate-900"></span>
        </button>

        {/* User Profile menu dropdown */}
        <div className="flex items-center space-x-3 cursor-pointer group pl-2">
          <div className="bg-gradient-to-tr from-blue-500 to-indigo-600 p-2 rounded-full text-white shadow-md shadow-blue-500/20">
            <User className="h-4 w-4" />
          </div>
          <div className="hidden md:block">
            <p className="text-xs text-gray-500 font-semibold uppercase leading-none">Administrator</p>
            <p className="text-sm font-medium text-gray-200 mt-1 flex items-center space-x-1">
              <span>Traffic Inspector</span>
              <ChevronDown className="h-3.5 w-3.5 text-gray-500 group-hover:text-white transition-colors" />
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}
