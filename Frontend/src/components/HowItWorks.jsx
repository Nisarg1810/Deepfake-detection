import React from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Cpu, FileSearch, CheckCircle } from 'lucide-react';

const steps = [
    {
        icon: <UploadCloud size={40} />,
        title: "Upload Video",
        desc: "Securely upload your video file for analysis."
    },
    {
        icon: <Cpu size={40} />,
        title: "Frame Extraction",
        desc: "System breaks down video into individual frames."
    },
    {
        icon: <FileSearch size={40} />,
        title: "Deep Analysis",
        desc: "Multi-model AI scans for manipulation traces."
    },
    {
        icon: <CheckCircle size={40} />,
        title: "Result Output",
        desc: "Receive detailed integrity report and score."
    }
];

const HowItWorks = () => {
    return (
        <section className="py-20 px-4 sm:px-6 lg:px-8 relative z-10 bg-black/20">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                        How It <span className="text-neon-purple">Works</span>
                    </h2>
                </motion.div>

                <div className="relative">
                    {/* Connecting Line */}
                    <div className="absolute top-1/2 left-0 w-full h-0.5 bg-white/10 -translate-y-1/2 hidden lg:block" />

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 relative">
                        {steps.map((step, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, scale: 0.8 }}
                                whileInView={{ opacity: 1, scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.2 }}
                                className="flex flex-col items-center text-center relative z-10"
                            >
                                <div className="w-20 h-20 rounded-full bg-deep-blue border-2 border-neon-blue flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(0,243,255,0.2)]">
                                    <div className="text-neon-blue">
                                        {step.icon}
                                    </div>
                                </div>
                                <h3 className="text-xl font-bold mb-2">{step.title}</h3>
                                <p className="text-gray-400 text-sm">{step.desc}</p>

                                {/* Mobile Connector */}
                                {index < steps.length - 1 && (
                                    <div className="lg:hidden absolute bottom-[-24px] left-1/2 -translate-x-1/2 w-0.5 h-8 bg-white/20" />
                                )}
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
};

export default HowItWorks;
