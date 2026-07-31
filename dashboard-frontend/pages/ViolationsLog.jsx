import React, { useState, useEffect, useMemo } from "react";
import { Search, Filter, Eye, Download, CheckCircle, Smartphone, X, Image as ImageIcon, ExternalLink, Calendar, CreditCard, AlertCircle } from "lucide-react";
import axios from "axios";

export default function ViolationsLog({ backendUrl }) {
  // Database challans state
  const [challans, setChallans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Search & Filter state
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [typeFilter, setTypeFilter] = useState("All");

  // Selected challan for inspection modal
  const [selectedChallan, setSelectedChallan] = useState(null);

  // Alert/Action statuses
  const [smsSentId, setSmsSentId] = useState(null);
  const [downloadingId, setDownloadingId] = useState(null);

  // Mock fallback data if the backend is offline
  const mockChallans = [
    {
      id: 1,
      challan_number: "CH-20260730-228998",
      timestamp: "2026-07-30T14:57:56",
      owner_name: "Niyati Kumawat",
      owner_phone: "+91 98765 43210",
      license_plate: "RJ14OC0398",
      violation_types: ["ZEBRA OBSTRUCTION"],
      fine_amount: 500,
      status: "Pending",
      evidence_image_path: "/media/evidence_CH-20260730-228998.jpg",
      pdf_path: "/media/CHALLAN_CH-20260730-228998.pdf"
    },
    {
      id: 2,
      challan_number: "CH-20260730-05429F",
      timestamp: "2026-07-30T14:58:33",
      owner_name: "Ramesh Kumar",
      owner_phone: "+91 99999 88888",
      license_plate: "RJ14AB1234",
      violation_types: ["NO HELMET"],
      fine_amount: 1000,
      status: "Paid",
      evidence_image_path: "/media/evidence_CH-20260730-05429F.jpg",
      pdf_path: "/media/CHALLAN_CH-20260730-05429F.pdf"
    },
    {
      id: 3,
      challan_number: "CH-20260730-1B2F89",
      timestamp: "2026-07-30T14:32:10",
      owner_name: "Vikram Singh",
      owner_phone: "+91 98111 22222",
      license_plate: "DL3CA5555",
      violation_types: ["WRONG WAY", "NO HELMET"],
      fine_amount: 3000,
      status: "Pending",
      evidence_image_path: "/media/evidence_CH-20260730-1B2F89.jpg",
      pdf_path: "/media/CHALLAN_CH-20260730-1B2F89.pdf"
    },
    {
      id: 4,
      challan_number: "CH-20260730-9K8L77",
      timestamp: "2026-07-30T13:10:45",
      owner_name: "Anil Kumble",
      owner_phone: "+91 94444 55555",
      license_plate: "KA03MM9999",
      violation_types: ["OVERSPEEDING"],
      fine_amount: 2000,
      status: "Disputed",
      evidence_image_path: "/media/evidence_CH-20260730-9K8L77.jpg",
      pdf_path: "/media/CHALLAN_CH-20260730-9K8L77.pdf"
    }
  ];

  // Fetch challans from backend with mock fallback
  const fetchChallans = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${backendUrl}/challans/`, { timeout: 3000 });
      setChallans(response.data);
      setError(null);
    } catch (err) {
      console.warn("Backend API offline. Using interactive mock data.");
      setChallans(mockChallans);
      setError("Unable to connect to local API service. Displaying fallback records.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChallans();
  }, [backendUrl]);

  // Handle Mark as Paid
  const handleMarkAsPaid = async (challanId) => {
    try {
      // Async backend update
      await axios.patch(`${backendUrl}/challans/${challanId}/pay`);
      
      // Update local state
      setChallans((prev) =>
        prev.map((c) => (c.id === challanId ? { ...c, status: "Paid" } : c))
      );
      if (selectedChallan && selectedChallan.id === challanId) {
        setSelectedChallan((prev) => ({ ...prev, status: "Paid" }));
      }
    } catch (err) {
      console.warn("Backend update failed. Updating mock state locally.");
      setChallans((prev) =>
        prev.map((c) => (c.id === challanId ? { ...c, status: "Paid" } : c))
      );
      if (selectedChallan && selectedChallan.id === challanId) {
        setSelectedChallan((prev) => ({ ...prev, status: "Paid" }));
      }
    }
  };

  // Mock SMS trigger
  const handleSendSMS = (challan) => {
    setSmsSentId(challan.id);
    alert(`[MOCK SMS SERVICE]\nSMS Alert dispatched to ${challan.owner_name} (${challan.owner_phone}) for Challan ID: ${challan.challan_number}.\nPenalty: INR ${challan.fine_amount}`);
    setTimeout(() => setSmsSentId(null), 2000);
  };

  // Mock PDF downloader
  const handleDownloadPDF = (challan) => {
    setDownloadingId(challan.id);
    
    // In real system, this opens/downloads the file:
    // window.open(`${backendUrl}/media/${challan.pdf_path}`)
    
    setTimeout(() => {
      setDownloadingId(null);
      alert(`Downloading Challan PDF document: CHALLAN_${challan.challan_number}.pdf`);
    }, 1500);
  };

  // Filter and search logic
  const filteredChallans = useMemo(() => {
    return challans.filter((item) => {
      const matchSearch =
        item.license_plate.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.challan_number.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.owner_name.toLowerCase().includes(searchTerm.toLowerCase());

      const matchStatus = statusFilter === "All" || item.status === statusFilter;
      
      const matchType =
        typeFilter === "All" ||
        item.violation_types.some((v) => v.toUpperCase().includes(typeFilter.toUpperCase()));

      return matchSearch && matchStatus && matchType;
    });
  }, [challans, searchTerm, statusFilter, typeFilter]);

  const formatDate = (isoStr) => {
    const dt = new Date(isoStr);
    return dt.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "Paid":
        return <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider">Paid</span>;
      case "Pending":
        return <span className="bg-red-500/10 text-red-400 border border-red-500/20 px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider animate-pulse">Pending</span>;
      case "Disputed":
        return <span className="bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider">Disputed</span>;
      default:
        return null;
    }
  };

  return (
    <div className="p-8 space-y-6 overflow-y-auto max-h-[calc(100vh-4rem)]">
      {/* Fallback API warning banner */}
      {error && (
        <div className="bg-amber-500/10 border border-amber-500/25 p-4 rounded-xl text-amber-400 text-sm flex items-center space-x-3 select-none">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Filter panel */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-gray-800 backdrop-blur-md flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap gap-4 items-center flex-1">
          {/* Search by plate or owner */}
          <div className="relative w-72">
            <Search className="absolute left-3.5 top-2.5 h-4 w-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search plate, owner, or challan ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-950/40 border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50"
            />
          </div>

          {/* Payment Status filter */}
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-gray-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-950/40 border border-gray-800 rounded-lg text-sm text-gray-200 px-3 py-2 focus:outline-none focus:border-blue-500/50"
            >
              <option value="All">All Statuses</option>
              <option value="Pending">Pending</option>
              <option value="Paid">Paid</option>
              <option value="Disputed">Disputed</option>
            </select>
          </div>

          {/* Violation Type filter */}
          <div className="flex items-center space-x-2">
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-slate-950/40 border border-gray-800 rounded-lg text-sm text-gray-200 px-3 py-2 focus:outline-none focus:border-blue-500/50"
            >
              <option value="All">All Violation Types</option>
              <option value="NO HELMET">No Helmet</option>
              <option value="WRONG WAY">Wrong Way</option>
              <option value="OVERSPEEDING">Overspeeding</option>
              <option value="ZEBRA OBSTRUCTION">Zebra Crossing</option>
            </select>
          </div>
        </div>

        <button 
          onClick={fetchChallans}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm px-4 py-2 rounded-lg transition-colors"
        >
          Refresh Log
        </button>
      </div>

      {/* Main Table Grid */}
      <div className="rounded-2xl bg-slate-900/60 border border-gray-800 overflow-hidden backdrop-blur-md select-none">
        {loading ? (
          <div className="p-20 text-center text-gray-400">
            <span className="animate-spin inline-block h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mb-3"></span>
            <p className="font-semibold">Loading E-Challan records...</p>
          </div>
        ) : filteredChallans.length === 0 ? (
          <div className="p-20 text-center text-gray-400 font-semibold">
            <AlertCircle className="h-10 w-10 text-gray-600 mx-auto mb-3" />
            <p>No violation logs found matching the filter criteria.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-950/50 border-b border-gray-800 text-xs font-bold uppercase tracking-wider text-gray-400">
                <tr>
                  <th className="px-6 py-4">Challan ID</th>
                  <th className="px-6 py-4">Vehicle No</th>
                  <th className="px-6 py-4">Owner Name</th>
                  <th className="px-6 py-4">Contact</th>
                  <th className="px-6 py-4">Violation Type</th>
                  <th className="px-6 py-4">Fine Amount</th>
                  <th className="px-6 py-4">Date & Time</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 font-medium">
                {filteredChallans.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-950/20 transition-colors">
                    <td className="px-6 py-4 text-blue-400 font-semibold font-mono">{item.challan_number}</td>
                    <td className="px-6 py-4 font-mono text-white font-bold">{item.license_plate}</td>
                    <td className="px-6 py-4 text-gray-200">{item.owner_name}</td>
                    <td className="px-6 py-4 text-gray-400 font-mono">{item.owner_phone}</td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-1">
                        {item.violation_types.map((v) => (
                          <span key={v} className="bg-red-500/10 text-red-400 text-xs px-2 py-0.5 rounded border border-red-500/10 font-bold uppercase">
                            {v}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-white font-bold">₹{item.fine_amount}</td>
                    <td className="px-6 py-4 text-gray-400 font-mono text-xs">{formatDate(item.timestamp)}</td>
                    <td className="px-6 py-4">{getStatusBadge(item.status)}</td>
                    <td className="px-6 py-4 text-center">
                      <button
                        onClick={() => setSelectedChallan(item)}
                        className="bg-slate-950/60 border border-gray-800 hover:border-blue-500/50 hover:bg-slate-950 p-2 rounded-lg text-gray-400 hover:text-white transition-colors"
                      >
                        <Eye className="h-4.5 w-4.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Interactive Modal Overlay (Evidence Inspector) */}
      {selectedChallan && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in select-none">
          <div className="w-full max-w-4xl bg-slate-900 border border-gray-800 rounded-3xl overflow-hidden shadow-2xl flex flex-col md:flex-row relative">
            {/* Close Button */}
            <button
              onClick={() => setSelectedChallan(null)}
              className="absolute top-4 right-4 bg-slate-950/60 hover:bg-slate-950 text-gray-400 hover:text-white p-2 rounded-full border border-gray-800/80 transition-colors z-10"
            >
              <X className="h-5 w-5" />
            </button>

            {/* Left Column: Image Snapshot */}
            <div className="w-full md:w-3/5 bg-slate-950 flex flex-col justify-center border-b md:border-b-0 md:border-r border-gray-800">
              <div className="relative aspect-video w-full h-full flex items-center justify-center p-4">
                {/* Simulated Snapshot Bounding Box image */}
                <div className="w-full aspect-video bg-slate-900 border border-gray-800/80 rounded-xl overflow-hidden flex flex-col items-center justify-center relative">
                  <ImageIcon className="h-12 w-12 text-gray-600 mb-3" />
                  <p className="text-gray-400 text-xs font-semibold">SNAPSHOT EVIDENCE IMAGE</p>
                  <p className="text-[10px] text-gray-600 font-mono mt-1">{selectedChallan.evidence_image_path}</p>
                  
                  {/* Glowing bounding box simulation */}
                  <div className="absolute top-1/4 left-1/4 right-1/4 bottom-1/4 border-4 border-red-500/80 rounded shadow-[0_0_15px_rgba(239,68,68,0.5)]">
                    <span className="absolute top-1 left-1 bg-red-600 text-white font-bold text-[8px] px-1.5 py-0.5 rounded">
                      VIOLATOR DETECTED
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Detailed Info Form */}
            <div className="w-full md:w-2/5 p-8 flex flex-col justify-between">
              <div className="space-y-6">
                <div>
                  <span className="text-xs text-gray-500 font-bold uppercase tracking-wider">Challan Summary</span>
                  <h3 className="text-2xl font-bold text-white font-mono mt-1">{selectedChallan.challan_number}</h3>
                </div>

                {/* Owner details card */}
                <div className="p-4 rounded-xl bg-slate-950/50 border border-gray-800/80 space-y-3">
                  <h4 className="text-xs font-bold text-blue-400 uppercase tracking-wider flex items-center space-x-1">
                    <CreditCard className="h-4 w-4" />
                    <span>Registered Profile</span>
                  </h4>
                  <div className="grid grid-cols-2 gap-y-2.5 gap-x-2 text-xs font-semibold">
                    <span className="text-gray-500">Plate Number:</span>
                    <span className="text-white font-mono text-sm uppercase">{selectedChallan.license_plate}</span>
                    <span className="text-gray-500">Owner Name:</span>
                    <span className="text-gray-200">{selectedChallan.owner_name}</span>
                    <span className="text-gray-500">Phone Contact:</span>
                    <span className="text-gray-400 font-mono">{selectedChallan.owner_phone}</span>
                  </div>
                </div>

                {/* Violation list details */}
                <div className="space-y-2">
                  <span className="text-xs text-gray-500 font-bold uppercase tracking-wider block">Violations Committed</span>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedChallan.violation_types.map((v) => (
                      <span key={v} className="bg-red-500/10 text-red-400 text-xs px-2.5 py-1 rounded-md border border-red-500/20 font-bold uppercase tracking-wide">
                        {v}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Fine Summary */}
                <div className="flex items-center justify-between border-t border-gray-800/60 pt-4 select-none">
                  <span className="text-sm font-semibold text-gray-400">Total Penalty Fine:</span>
                  <span className="text-2xl font-bold text-red-500 font-mono">₹{selectedChallan.fine_amount}</span>
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Payment Status:</span>
                  <span>{getStatusBadge(selectedChallan.status)}</span>
                </div>
              </div>

              {/* Action buttons footer */}
              <div className="mt-8 space-y-2.5">
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => handleDownloadPDF(selectedChallan)}
                    className="w-full bg-slate-950 hover:bg-slate-950/80 border border-gray-800 hover:border-gray-700 text-white font-semibold text-xs py-3 rounded-xl flex items-center justify-center space-x-1.5 transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    <span>{downloadingId ? "Downloading..." : "PDF Challan"}</span>
                  </button>
                  <button
                    onClick={() => handleSendSMS(selectedChallan)}
                    className="w-full bg-slate-950 hover:bg-slate-950/80 border border-gray-800 hover:border-gray-700 text-white font-semibold text-xs py-3 rounded-xl flex items-center justify-center space-x-1.5 transition-colors"
                  >
                    <Smartphone className="h-4 w-4" />
                    <span>{smsSentId ? "Sending..." : "Send SMS"}</span>
                  </button>
                </div>
                {selectedChallan.status !== "Paid" && (
                  <button
                    onClick={() => handleMarkAsPaid(selectedChallan.id)}
                    className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs py-3.5 rounded-xl flex items-center justify-center space-x-1.5 shadow-lg shadow-emerald-600/10 transition-colors"
                  >
                    <CheckCircle className="h-4.5 w-4.5" />
                    <span>Mark as Paid</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
