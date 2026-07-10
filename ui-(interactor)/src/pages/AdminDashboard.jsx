import React, { useContext, useState, useEffect } from 'react';
import { AuthContext } from '../context/AuthContext';
import AuditReportModal from '../components/AuditReportModal';

export default function AdminDashboard() {
    const { logout, token } = useContext(AuthContext);
    const [datasets, setDatasets] = useState([]);
    const [selectedAuditId, setSelectedAuditId] = useState(null);

    const fetchDatasets = async () => {
        try {
            const res = await fetch('/api/datasets', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                setDatasets(data);
            }
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        fetchDatasets();
        const interval = setInterval(fetchDatasets, 5000); // Polling
        return () => clearInterval(interval);
    }, [token]);

    return (
        <div className="p-8 max-w-6xl mx-auto">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-800">Admin Dashboard</h1>
                    <p className="text-gray-500">Monitor all student uploads and system audits.</p>
                </div>
                <button onClick={logout} className="px-4 py-2 bg-gray-200 text-gray-700 hover:bg-gray-300 rounded font-medium">Logout</button>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                <h2 className="text-xl font-semibold mb-4 text-gray-700">All Platform Datasets</h2>
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b">
                                <th className="p-3 text-sm font-semibold text-gray-600">ID</th>
                                <th className="p-3 text-sm font-semibold text-gray-600">Uploader</th>
                                <th className="p-3 text-sm font-semibold text-gray-600">Filename</th>
                                <th className="p-3 text-sm font-semibold text-gray-600">Status</th>
                                <th className="p-3 text-sm font-semibold text-gray-600">Date</th>
                                <th className="p-3 text-sm font-semibold text-gray-600 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {datasets.map(ds => (
                                <tr key={ds.id} className="border-b hover:bg-gray-50">
                                    <td className="p-3 text-sm">{ds.id}</td>
                                    <td className="p-3 text-sm text-gray-600">{ds.user?.email}</td>
                                    <td className="p-3 text-sm font-medium">{ds.filename}</td>
                                    <td className="p-3 text-sm">
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                                            ds.status === 'COMPLETED' ? 'bg-green-100 text-green-700' : 
                                            ds.status === 'FAILED' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
                                        }`}>
                                            {ds.status}
                                        </span>
                                    </td>
                                    <td className="p-3 text-sm text-gray-500">{new Date(ds.uploadedAt).toLocaleString()}</td>
                                    <td className="p-3 text-sm text-right">
                                        {(ds.status === 'COMPLETED' || ds.status === 'FAILED') && (
                                            <button 
                                                onClick={() => setSelectedAuditId(ds.id)}
                                                className="text-blue-600 hover:text-blue-800 font-medium text-sm">
                                                View Report
                                            </button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {datasets.length === 0 && (
                                <tr>
                                    <td colSpan="6" className="p-4 text-center text-gray-500">No datasets uploaded yet.</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {selectedAuditId && (
                <AuditReportModal 
                    datasetId={selectedAuditId} 
                    onClose={() => setSelectedAuditId(null)} 
                    token={token}
                />
            )}
        </div>
    );
}
