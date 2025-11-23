import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Upload, FileVideo, X, AlertCircle } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import Navbar from '../components/Navbar';
import Background from '../components/Background';
import Footer from '../components/Footer';
import CustomCursor from '../components/CustomCursor';

const UploadPage = () => {
    const navigate = useNavigate();
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        if (file.type.startsWith('video/')) {
            setFile(file);
        } else {
            alert("Please upload a video file.");
        }
    };

    const removeFile = () => {
        setFile(null);
    };

    const handleAnalyze = () => {
        if (file) {
            navigate('/processing', { state: { videoFile: file } });
        }
    };

    return (
        <>
            <CustomCursor />
            <div className="relative min-h-screen w-full overflow-hidden text-white">
                <Background />
                <Navbar />

                <main className="relative z-10 container mx-auto px-4 pt-32 pb-20 flex flex-col items-center justify-center min-h-screen">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="text-center mb-12"
                    >
                        <h1 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight">
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400">
                                Upload Video for Analysis
                            </span>
                        </h1>
                        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
                            Our advanced AI will scan every frame for temporal inconsistencies and GAN fingerprints.
                        </p>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className="w-full max-w-2xl"
                    >
                        <Card className="bg-black/40 border-white/10 backdrop-blur-xl overflow-hidden">
                            <CardContent className="p-8">
                                {!file ? (
                                    <div
                                        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${dragActive ? "border-neon-blue bg-neon-blue/5" : "border-white/10 hover:border-white/20"
                                            }`}
                                        onDragEnter={handleDrag}
                                        onDragLeave={handleDrag}
                                        onDragOver={handleDrag}
                                        onDrop={handleDrop}
                                    >
                                        <input
                                            type="file"
                                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                            onChange={handleChange}
                                            accept="video/*"
                                        />
                                        <div className="flex flex-col items-center gap-4">
                                            <div className="p-4 rounded-full bg-white/5">
                                                <Upload className="w-8 h-8 text-neon-blue" />
                                            </div>
                                            <div>
                                                <p className="text-lg font-medium mb-1">Drag & drop your video here</p>
                                                <p className="text-sm text-gray-400">or click to browse files</p>
                                            </div>
                                            <p className="text-xs text-gray-500 mt-2">MP4, MOV, AVI up to 500MB</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="relative rounded-xl border border-white/10 bg-white/5 p-6">
                                        <button
                                            onClick={removeFile}
                                            className="absolute top-4 right-4 p-1 rounded-full hover:bg-white/10 transition-colors"
                                        >
                                            <X className="w-5 h-5 text-gray-400" />
                                        </button>
                                        <div className="flex items-center gap-4">
                                            <div className="p-3 rounded-lg bg-neon-blue/10">
                                                <FileVideo className="w-8 h-8 text-neon-blue" />
                                            </div>
                                            <div>
                                                <p className="font-medium truncate max-w-[200px] sm:max-w-xs">{file.name}</p>
                                                <p className="text-sm text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div className="mt-8 flex justify-center">
                                    <Button
                                        size="lg"
                                        className="w-full sm:w-auto min-w-[200px] bg-white text-black hover:bg-gray-200 font-bold text-lg h-12 shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                                        disabled={!file}
                                        onClick={handleAnalyze}
                                    >
                                        {file ? "Analyze Video" : "Select Video"}
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>
                </main>
                <Footer />
            </div>
        </>
    );
};

export default UploadPage;
