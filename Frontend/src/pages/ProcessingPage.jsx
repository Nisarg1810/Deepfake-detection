import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Progress } from '../components/ui/progress';
import Background from '../components/Background';
import CustomCursor from '../components/CustomCursor';
import { analyzeVideo } from '../services/api';

const processingSteps = [
    "Initializing neural networks...",
    "Extracting frames from video...",
    "Detecting faces in frames...",
    "Running CNN detection...",
    "Analyzing frequency patterns...",
    "Performing lip-sync analysis...",
    "Running temporal detection...",
    "Aggregating scores...",
    "Generating final report..."
];

const ProcessingPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [progress, setProgress] = useState(0);
    const [stepIndex, setStepIndex] = useState(0);
    const [error, setError] = useState(null);
    const videoFile = location.state?.videoFile;

    useEffect(() => {
        // Redirect if no video file
        if (!videoFile) {
            navigate('/upload');
            return;
        }

        // Start the analysis
        const performAnalysis = async () => {
            try {
                // Simulate progress updates while waiting for backend
                const progressInterval = setInterval(() => {
                    setProgress((prev) => {
                        if (prev >= 95) {
                            clearInterval(progressInterval);
                            return 95; // Stop at 95% until we get results
                        }
                        return prev + 1;
                    });
                }, 100); // Slower progress for realistic feel

                // Step updates
                const stepInterval = setInterval(() => {
                    setStepIndex((prev) => (prev < processingSteps.length - 1 ? prev + 1 : prev));
                }, 1200);

                // Call backend API
                const result = await analyzeVideo(videoFile);

                // Clear intervals
                clearInterval(progressInterval);
                clearInterval(stepInterval);

                // Complete progress
                setProgress(100);
                setStepIndex(processingSteps.length - 1);

                // Navigate to results with the actual data
                setTimeout(() => {
                    navigate('/results', { state: { result } });
                }, 500);

            } catch (err) {
                console.error('Analysis error:', err);
                setError(err.message || 'Analysis failed. Please try again.');
                setProgress(0);
            }
        };

        performAnalysis();
    }, [videoFile, navigate]);

    if (error) {
        return (
            <>
                <CustomCursor />
                <div className="relative min-h-screen w-full overflow-hidden text-white flex flex-col items-center justify-center">
                    <Background />
                    <div className="relative z-10 w-full max-w-md px-6 text-center">
                        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-8 mb-4">
                            <h2 className="text-2xl font-bold text-red-500 mb-4">Analysis Failed</h2>
                            <p className="text-gray-300 mb-6">{error}</p>
                            <button
                                onClick={() => navigate('/upload')}
                                className="px-6 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition-colors"
                            >
                                Try Again
                            </button>
                        </div>
                        <p className="text-xs text-gray-500">
                            Make sure the backend server is running on port 5000
                        </p>
                    </div>
                </div>
            </>
        );
    }

    return (
        <>
            <CustomCursor />
            <div className="relative min-h-screen w-full overflow-hidden text-white flex flex-col items-center justify-center">
                <Background />

                <div className="relative z-10 w-full max-w-md px-6 text-center">
                    {/* Planet Animation */}
                    <div className="relative w-48 h-48 mx-auto mb-12">
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                            className="absolute inset-0 rounded-full border-2 border-dashed border-neon-blue/30"
                        />
                        <motion.div
                            animate={{ rotate: -360 }}
                            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                            className="absolute inset-4 rounded-full border-2 border-dashed border-neon-purple/30"
                        />
                        <div className="absolute inset-0 rounded-full bg-gradient-radial from-neon-blue/20 to-transparent blur-2xl animate-pulse" />

                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-4xl font-bold font-mono">{Math.round(progress)}%</span>
                        </div>
                    </div>

                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        key={stepIndex}
                        className="h-8 mb-4"
                    >
                        <p className="text-neon-blue font-mono text-sm sm:text-base typing-effect">
                            {processingSteps[stepIndex]}
                        </p>
                    </motion.div>

                    <Progress value={progress} className="h-2 bg-white/10" />

                    <p className="mt-4 text-xs text-gray-500 uppercase tracking-widest">
                        Do not close this window
                    </p>
                </div>
            </div>
        </>
    );
};

export default ProcessingPage;
