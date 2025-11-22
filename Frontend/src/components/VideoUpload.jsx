import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const VideoUpload = ({ onClose }) => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type.startsWith('video/')) {
            setFile(selectedFile);
            setError(null);
        } else {
            setError('Please select a valid video file');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type.startsWith('video/')) {
            setFile(droppedFile);
            setError(null);
        } else {
            setError('Please drop a valid video file');
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message || 'Failed to analyze video');
        } finally {
            setUploading(false);
        }
    };

    const getVerdictColor = (label) => {
        if (label.includes('AUTHENTIC')) return 'text-green-500';
        if (label.includes('MANIPULATED')) return 'text-red-500';
        return 'text-yellow-500';
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={onClose}
        >
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="relative w-full max-w-2xl glass-panel rounded-2xl p-8"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                    <X size={24} className="text-gray-400" />
                </button>

                <h2 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple">
                    Upload Video for Analysis
                </h2>

                {!result ? (
                    <>
                        {/* Upload Area */}
                        <div
                            onDrop={handleDrop}
                            onDragOver={(e) => e.preventDefault()}
                            onClick={() => fileInputRef.current?.click()}
                            className="border-2 border-dashed border-neon-blue/30 rounded-xl p-12 text-center cursor-pointer hover:border-neon-blue/60 transition-colors mb-6"
                        >
                            <Upload size={48} className="mx-auto mb-4 text-neon-blue" />
                            <p className="text-lg text-gray-300 mb-2">
                                {file ? file.name : 'Click to upload or drag and drop'}
                            </p>
                            <p className="text-sm text-gray-500">
                                MP4, AVI, MOV supported
                            </p>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="video/*"
                                onChange={handleFileSelect}
                                className="hidden"
                            />
                        </div>

                        {error && (
                            <div className="flex items-center gap-2 p-4 bg-red-500/10 border border-red-500/30 rounded-lg mb-6">
                                <AlertCircle size={20} className="text-red-500" />
                                <p className="text-red-500">{error}</p>
                            </div>
                        )}

                        {/* Upload Button */}
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleUpload}
                            disabled={!file || uploading}
                            className="w-full px-8 py-4 bg-neon-blue text-black font-bold rounded-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {uploading ? (
                                <>
                                    <Loader2 size={20} className="animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Upload size={20} />
                                    Analyze Video
                                </>
                            )}
                        </motion.button>
                    </>
                ) : (
                    /* Results */
                    <div className="space-y-6">
                        {/* Verdict */}
                        <div className="text-center p-6 glass-panel rounded-xl">
                            <div className={`text-4xl font-bold mb-2 ${getVerdictColor(result.verdict.final_label)}`}>
                                {result.verdict.final_label.replace(/_/g, ' ')}
                            </div>
                            <div className="text-2xl text-neon-blue">
                                {(result.verdict.final_score * 100).toFixed(1)}% Confidence
                            </div>
                        </div>

                        {/* Metrics */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className="glass-panel p-4 rounded-xl">
                                <div className="text-sm text-gray-400">CNN Score</div>
                                <div className="text-xl font-bold text-neon-blue">
                                    {(result.aggregation.max_score * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="glass-panel p-4 rounded-xl">
                                <div className="text-sm text-gray-400">Frequency</div>
                                <div className="text-xl font-bold text-neon-purple">
                                    {(result.aggregation.frequency_score * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="glass-panel p-4 rounded-xl">
                                <div className="text-sm text-gray-400">Temporal</div>
                                <div className="text-xl font-bold text-neon-blue">
                                    {(result.aggregation.temporal_max * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="glass-panel p-4 rounded-xl">
                                <div className="text-sm text-gray-400">Lip-Sync</div>
                                <div className="text-xl font-bold text-neon-purple">
                                    {result.aggregation.lip_sync_score ?
                                        (result.aggregation.lip_sync_score * 100).toFixed(1) : 'N/A'}%
                                </div>
                            </div>
                        </div>

                        {/* Details */}
                        <div className="glass-panel p-4 rounded-xl space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Frames Analyzed:</span>
                                <span className="text-white">{result.frames}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Faces Detected:</span>
                                <span className="text-white">{result.faces}</span>
                            </div>
                        </div>

                        {/* Test Another Button */}
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => {
                                setFile(null);
                                setResult(null);
                                setError(null);
                            }}
                            className="w-full px-8 py-4 border border-neon-blue text-neon-blue font-bold rounded-lg"
                        >
                            Test Another Video
                        </motion.button>
                    </div>
                )}
            </motion.div>
        </motion.div>
    );
};

export default VideoUpload;
