import { useContext, useState, useEffect, useCallback } from 'react';
import { AuthContext } from '../context/AuthContext';
import AuditReportModal from '../components/AuditReportModal';

export default function StudentDashboard() {
    const { logout, token } = useContext(AuthContext);
    const [file, setFile] = useState(null);
    const [datasets, setDatasets] = useState([]);
    const [selectedAuditId, setSelectedAuditId] = useState(null);
    const [huggingFaceId, setHuggingFaceId] = useState('');

    const fetchDatasets = useCallback(async () => {
        try {
            const res = await fetch('/api/datasets', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                setDatasets(data);
            } else if (res.status === 401 || res.status === 403) {
                logout(); // Token expired or invalid
            }
        } catch (err) {
            console.error(err);
        }
    }, [token, logout]);

    useEffect(() => {
        setTimeout(fetchDatasets, 0); // Defer execution to satisfy strict React Compiler linters
        const interval = setInterval(fetchDatasets, 5000); // Polling for status updates
        return () => clearInterval(interval);
    }, [fetchDatasets]);

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/datasets/upload', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });
            if (res.ok) {
                setFile(null);
                e.target.reset(); // Clear file input
                fetchDatasets();
            } else {
                alert('Upload failed');
            }
        } catch (err) {
            console.error(err);
        }
    };

    const handleHuggingFaceImport = async (e) => {
        e.preventDefault();
        if (!huggingFaceId) return;

        try {
            const res = await fetch('/api/datasets/huggingface', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ huggingFaceId })
            });
            if (res.ok) {
                setHuggingFaceId('');
                fetchDatasets();
            } else {
                alert('Import failed');
            }
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="p-8 max-w-5xl mx-auto">
            <div className="flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold text-gray-800">Student Dashboard</h1>
                <button onClick={logout} className="px-4 py-2 bg-gray-200 text-gray-700 hover:bg-gray-300 rounded font-medium">Logout</button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                    <h2 className="text-xl font-semibold mb-4 text-gray-700">Upload Dataset</h2>
                    <form onSubmit={handleUpload} className="flex flex-col gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-600 mb-2">Select JSON or CSV dataset file</label>
                            <input type="file" onChange={e => setFile(e.target.files[0])} className="block w-full text-sm text-gray-500
                              file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold
                              file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" required />
                        </div>
                        <button type="submit" className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition w-full">
                            Upload & Process
                        </button>
                    </form>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                    <h2 className="text-xl font-semibold mb-4 text-gray-700">Import from HuggingFace</h2>
                    <form onSubmit={handleHuggingFaceImport} className="flex flex-col gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-600 mb-2">HuggingFace Dataset ID</label>
                            <input type="text" 
                                value={huggingFaceId}
                                onChange={e => setHuggingFaceId(e.target.value)}
                                placeholder="e.g. tatsu-lab/alpaca" 
                                className="block w-full p-2 border rounded-md text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500" 
                                required />
                        </div>
                        <button type="submit" className="px-6 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 transition w-full">
                            Import & Process
                        </button>
                    </form>
                </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                <h2 className="text-xl font-semibold mb-4 text-gray-700">My Datasets</h2>
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="border-b">
                                <th className="p-3 text-sm font-semibold text-gray-600">ID</th>
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
                                    <td colSpan="5" className="p-4 text-center text-gray-500">No datasets uploaded yet.</td>
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
