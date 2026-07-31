import React, { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";
import LiveFeed from "../pages/LiveFeed";
import ViolationsLog from "../pages/ViolationsLog";
import Settings from "../pages/Settings";
import VideoAnalysis from "../pages/VideoAnalysis";
import axios from "axios";

export default function App() {
  const [activePage, setActivePage] = useState("LiveFeed");
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000/api/v1");
  const [backendStatus, setBackendStatus] = useState(false);

  // Ping backend database health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        // Strip trailing slash
        const cleanUrl = backendUrl.replace(/\/$/, "");
        const response = await axios.get(cleanUrl.substring(0, cleanUrl.lastIndexOf("/api/v1")), { timeout: 2500 });
        if (response.status === 200) {
          setBackendStatus(true);
        } else {
          setBackendStatus(false);
        }
      } catch (err) {
        setBackendStatus(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [backendUrl]);

  const renderPage = () => {
    switch (activePage) {
      case "LiveFeed":
        return <LiveFeed backendStatus={backendStatus} />;
      case "ViolationsLog":
        return <ViolationsLog backendUrl={backendUrl} />;
      case "VideoAnalysis":
        return <VideoAnalysis backendUrl={backendUrl} />;
      case "Settings":
        return <Settings backendUrl={backendUrl} setBackendUrl={setBackendUrl} />;
      default:
        return <LiveFeed backendStatus={backendStatus} />;
    }
  };

  return (
    <div className="flex h-screen bg-[#0B0F19] text-gray-100 font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar 
        activePage={activePage} 
        setActivePage={setActivePage} 
        backendStatus={backendStatus} 
      />

      {/* Main Content Pane */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Navbar Actions Header */}
        <Navbar activePage={activePage} />

        {/* Dynamic page container */}
        <main className="flex-1 overflow-hidden relative">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
