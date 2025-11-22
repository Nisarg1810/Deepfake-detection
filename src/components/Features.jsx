import React from 'react';
import { motion } from 'framer-motion';
import { Scan, Activity, ShieldCheck, Zap } from 'lucide-react';

const features = [
    {
        icon: <Scan size={32} />,
        title: "Frame-by-Frame Analysis",
        description: "Advanced algorithms analyze every single frame for inconsistencies and artifacts invisible to the human eye.",
        color: "text-neon-blue",
        border: "hover:border-neon-blue"
    },
    {
        icon: <Activity size={32} />,
        title: "Temporal Consistency",
        description: "Detects irregularities in motion and lighting across time, identifying deepfake jitter and warping.",
        color: "text-neon-purple",
        border: "hover:border-neon-purple"
    },
    {
        icon: <ShieldCheck size={32} />,
        title: "GAN Artifact Detection",
        description: "Specialized models trained to spot specific fingerprints left by Generative Adversarial Networks.",
        color: "text-neon-cyan",
        border: "hover:border-neon-cyan"
    },
    {
        icon: <Zap size={32} />,
        title: "Real-time Scoring",
        description: "Get instant probability scores and detailed reports within seconds of uploading your video.",
        color: "text-yellow-400",
        border: "hover:border-yellow-400"
    }
];

const Features = () => {
    return (
        <section className="py-20 px-4 sm:px-6 lg:px-8 relative z-10">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                        Advanced <span className="text-neon-blue">Detection Capabilities</span>
                    </h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Powered by state-of-the-art neural networks designed to expose digital manipulation.
                    </p>
                </motion.div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                    {features.map((feature, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            whileHover={{ y: -10 }}
                            className={`glass-panel p-8 rounded-xl border border-white/5 transition-colors duration-300 ${feature.border} group`}
                        >
                            <div className={`mb-6 ${feature.color} p-3 bg-white/5 rounded-lg inline-block group-hover:scale-110 transition-transform`}>
                                {feature.icon}
                            </div>
                            <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                            <p className="text-gray-400 text-sm leading-relaxed">
                                {feature.description}
                            </p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Features;
