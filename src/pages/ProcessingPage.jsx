import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Progress } from '../components/ui/progress';
import Background from '../components/Background';
import CustomCursor from '../components/CustomCursor';

const processingSteps = [
    "Initializing neural networks...",
    "Scanning temporal inconsistencies...",
    "Detecting facial anomalies...",
    "Evaluating GAN fingerprints...",
    "Analyzing frame-by-frame coherence...",
    "Finalizing confidence score..."
];

const ProcessingPage = () => {
    const navigate = useNavigate();
    const [progress, setProgress] = useState(0);
    const [stepIndex, setStepIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    setTimeout(() => navigate('/results'), 500);
                    return 100;
                }
                return prev + 1;
            });
        }, 50); // 5 seconds total

        return () => clearInterval(interval);
    }, [navigate]);

    useEffect(() => {
        const stepInterval = setInterval(() => {
            setStepIndex((prev) => (prev < processingSteps.length - 1 ? prev + 1 : prev));
        }, 800);

        return () => clearInterval(stepInterval);
    }, []);

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
                            <span className="text-4xl font-bold font-mono">{progress}%</span>
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
