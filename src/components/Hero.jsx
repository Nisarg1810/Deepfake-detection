import React from 'react';
import { motion } from 'framer-motion';
import { Play, Upload } from 'lucide-react';

const Hero = () => {
    return (
        <section className="relative min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 pt-20">
            <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">

                {/* Text Content */}
                <motion.div
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="text-center lg:text-left z-10"
                >
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="inline-block mb-4 px-4 py-1 rounded-full border border-neon-blue/30 bg-neon-blue/10 text-neon-blue text-sm font-semibold tracking-wider"
                    >
                        NEXT GEN AI DETECTION
                    </motion.div>

                    <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-gray-400">
                            DeepSearch
                        </span>
                        <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple text-glow">
                            AI Video Detection
                        </span>
                    </h1>

                    <p className="text-gray-400 text-lg sm:text-xl mb-8 max-w-2xl mx-auto lg:mx-0">
                        Detect AI-generated videos with advanced deep learning intelligence.
                        Ensure content integrity in real-time with 99.9% accuracy.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                        <motion.button
                            whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(0, 243, 255, 0.5)" }}
                            whileTap={{ scale: 0.95 }}
                            className="px-8 py-4 bg-neon-blue text-black font-bold rounded-lg flex items-center justify-center gap-2 transition-all"
                        >
                            <Upload size={20} />
                            Upload Video
                        </motion.button>

                        <motion.button
                            whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.1)" }}
                            whileTap={{ scale: 0.95 }}
                            className="px-8 py-4 border border-white/20 text-white font-bold rounded-lg flex items-center justify-center gap-2 backdrop-blur-sm hover:border-neon-purple transition-all"
                        >
                            <Play size={20} />
                            Try Demo
                        </motion.button>
                    </div>
                </motion.div>

                {/* Visual Content */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, delay: 0.2 }}
                    className="relative z-10 hidden lg:block"
                >
                    <div className="relative w-full aspect-square max-w-lg mx-auto">
                        {/* Abstract Globe/Brain Representation */}
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                            className="absolute inset-0 rounded-full border border-neon-blue/20 border-dashed"
                        />
                        <motion.div
                            animate={{ rotate: -360 }}
                            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                            className="absolute inset-8 rounded-full border border-neon-purple/20 border-dashed"
                        />

                        {/* Floating Elements */}
                        <motion.div
                            animate={{ y: [-20, 20, -20] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                            className="absolute top-1/4 right-0 p-4 glass-panel rounded-xl border-l-4 border-neon-blue"
                        >
                            <div className="text-xs text-gray-400">Confidence Score</div>
                            <div className="text-2xl font-bold text-neon-blue">99.9%</div>
                        </motion.div>

                        <motion.div
                            animate={{ y: [20, -20, 20] }}
                            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                            className="absolute bottom-1/4 left-0 p-4 glass-panel rounded-xl border-l-4 border-neon-purple"
                        >
                            <div className="text-xs text-gray-400">Deepfake Detected</div>
                            <div className="text-xl font-bold text-red-500">0.00%</div>
                        </motion.div>

                        {/* Central Glow */}
                        <div className="absolute inset-0 bg-gradient-radial from-neon-blue/20 to-transparent opacity-50 blur-3xl" />
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default Hero;
