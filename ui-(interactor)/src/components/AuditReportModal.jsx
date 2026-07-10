import React, { useEffect, useState } from 'react';

export default function AuditReportModal({ datasetId, onClose, token }) {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                const res = await fetch(`/api/datasets/${datasetId}/audit`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (res.ok) {
                    const text = await res.text();
                    try {
                        setReport(JSON.parse(text));
                    } catch (e) {
                        setReport({ error: "Failed to parse report JSON", raw: text });
                    }
                }
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, [datasetId, token]);

    if (loading) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white p-6 rounded shadow-lg">Loading...</div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
                <div className="p-6 border-b flex justify-between items-center bg-gray-50 rounded-t-lg">
                    <h2 className="text-2xl font-bold text-gray-800">Dataset Audit Report</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800 text-2xl font-bold">&times;</button>
                </div>
                
                <div className="p-6 overflow-y-auto">
                    {!report ? (
                        <p className="text-gray-500">No report available.</p>
                    ) : report.error ? (
                        <div className="bg-red-50 text-red-700 p-4 rounded border border-red-200">
                            <strong>Error Processing Dataset:</strong> {report.error}
                        </div>
                    ) : (
                        <div className="space-y-6">
                            
                            {/* Header / Score */}
                            <div className="flex items-center gap-6 p-6 bg-blue-50 rounded-lg border border-blue-100">
                                <div className="flex flex-col items-center justify-center bg-white rounded-full w-24 h-24 shadow-sm border-4 border-blue-200">
                                    <span className="text-3xl font-black text-blue-600">{report.health_score || 0}</span>
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold text-gray-800 mb-1">Health Score: {report.grade}</h3>
                                    <p className="text-sm text-gray-600">Based on schema validation, missing values, duplicates, and data privacy.</p>
                                </div>
                            </div>
                            
                            {/* Recommendations */}
                            {report.recommendations && report.recommendations.length > 0 && (
                                <div>
                                    <h4 className="text-lg font-bold text-gray-800 mb-3">Key Findings & Recommendations</h4>
                                    <ul className="space-y-2">
                                        {report.recommendations.map((rec, i) => (
                                            <li key={i} className={`p-3 rounded border text-sm font-medium ${
                                                rec.startsWith('CRITICAL') ? 'bg-red-50 border-red-200 text-red-800' :
                                                rec.startsWith('WARNING') ? 'bg-yellow-50 border-yellow-200 text-yellow-800' :
                                                'bg-blue-50 border-blue-200 text-blue-800'
                                            }`}>
                                                {rec}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Metrics Grid */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-gray-50 p-4 rounded border">
                                    <h5 className="font-bold text-gray-700 mb-2 border-b pb-2">Structure</h5>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Format:</span> <span className="font-medium">{report.schema_type || 'Unknown'}</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Total Records:</span> <span className="font-medium">{report.total_records || 0}</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Invalid Rows:</span> <span className="font-medium">{report.inconsistent_rows || 0}</span></p>
                                </div>
                                <div className="bg-gray-50 p-4 rounded border">
                                    <h5 className="font-bold text-gray-700 mb-2 border-b pb-2">Quality</h5>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Missing Values:</span> <span className="font-medium">{report.missing_percentage || 0}%</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Exact Duplicates:</span> <span className="font-medium">{report.duplicate_percentage || 0}%</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Est. Tokens:</span> <span className="font-medium">{report.estimated_total_tokens || 0}</span></p>
                                </div>
                                <div className="bg-gray-50 p-4 rounded border">
                                    <h5 className="font-bold text-gray-700 mb-2 border-b pb-2">Statistics</h5>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Mean Length:</span> <span className="font-medium">{report.char_length_mean || 0} chars</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">P95 Length:</span> <span className="font-medium">{report.char_length_p95 || 0} chars</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Avg Turns/Dialog:</span> <span className="font-medium">{report.avg_turns_per_dialogue || 0}</span></p>
                                </div>
                                <div className="bg-gray-50 p-4 rounded border">
                                    <h5 className="font-bold text-gray-700 mb-2 border-b pb-2">Privacy & Compliance</h5>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Emails Found:</span> <span className="font-medium">{report.pii_emails_detected || 0}</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">IPs Found:</span> <span className="font-medium">{report.pii_ips_detected || 0}</span></p>
                                    <p className="text-sm flex justify-between"><span className="text-gray-500">Special Char Ratio:</span> <span className="font-medium">{report.special_char_ratio || 0}</span></p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                
                <div className="p-4 border-t bg-gray-50 rounded-b-lg flex justify-end">
                    <button onClick={onClose} className="px-6 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-900 transition">
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}
