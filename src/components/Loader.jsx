import React from 'react';
import { motion } from 'framer-motion';

const Loader = () => {
    return (
        <motion.div
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-space-black"
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
        >
            <div className="relative w-24 h-24">
                <motion.div
                    className="absolute inset-0 border-4 border-t-neon-blue border-r-transparent border-b-neon-purple border-l-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                />
                <motion.div
                    className="absolute inset-2 border-4 border-t-transparent border-r-neon-cyan border-b-transparent border-l-neon-blue rounded-full"
                    animate={{ rotate: -360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                        className="w-4 h-4 bg-white rounded-full shadow-[0_0_15px_rgba(255,255,255,0.8)]"
                        animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                    />
                </div>
            </div>
            <motion.h2
                className="mt-8 text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple tracking-widest"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                INITIALIZING SYSTEM...
            </motion.h2>
            <motion.div
                className="w-64 h-1 mt-4 bg-white/10 rounded-full overflow-hidden"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
            >
                <motion.div
                    className="h-full bg-gradient-to-r from-neon-blue to-neon-purple"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                />
            </motion.div>
        </motion.div>
    );
};

export default Loader;
