import React from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

export default function MetricsCard({ title, value, changePercentage, icon: Icon, trend, accentColor = "blue" }) {
  // Color mapping configurations
  const colorMap = {
    blue: {
      text: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "hover:border-blue-500/30",
      shadow: "hover:shadow-blue-500/5",
    },
    red: {
      text: "text-red-400",
      bg: "bg-red-500/10",
      border: "hover:border-red-500/30",
      shadow: "hover:shadow-red-500/5",
    },
    emerald: {
      text: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "hover:border-emerald-500/30",
      shadow: "hover:shadow-emerald-500/5",
    },
    amber: {
      text: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "hover:border-amber-500/30",
      shadow: "hover:shadow-amber-500/5",
    },
  };

  const scheme = colorMap[accentColor] || colorMap.blue;

  return (
    <div className={`p-6 rounded-2xl bg-slate-900/60 border border-gray-800/80 backdrop-blur-md transition-all duration-300 ${scheme.border} ${scheme.shadow} hover:shadow-lg hover:-translate-y-0.5 group`}>
      <div className="flex items-center justify-between">
        {/* Metric Label */}
        <span className="text-sm font-semibold text-gray-400">{title}</span>
        
        {/* Icon container */}
        <div className={`p-3 rounded-xl ${scheme.bg} ${scheme.text} group-hover:scale-105 transition-transform duration-300`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-4 flex items-end justify-between">
        {/* Metric Value */}
        <h3 className="text-2xl md:text-3xl font-bold text-white tracking-tight leading-none">
          {value}
        </h3>

        {/* Metric Trend indicator badge */}
        {changePercentage && (
          <div className={`flex items-center space-x-0.5 text-xs font-semibold px-2 py-1 rounded-lg ${
            trend === "up" 
              ? "text-emerald-400 bg-emerald-500/10" 
              : "text-red-400 bg-red-500/10"
          }`}>
            {trend === "up" ? (
              <ArrowUpRight className="h-3.5 w-3.5" />
            ) : (
              <ArrowDownRight className="h-3.5 w-3.5" />
            )}
            <span>{changePercentage}</span>
          </div>
        )}
      </div>
    </div>
  );
}
