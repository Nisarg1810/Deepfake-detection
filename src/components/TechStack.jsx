import React from 'react';
import { motion } from 'framer-motion';

const techs = [
    "React", "Tailwind CSS", "Python", "TensorFlow", "PyTorch", "OpenCV", "Transformers", "Framer Motion"
];

const TechStack = () => {
    return (
        <section className="py-20 px-4 sm:px-6 lg:px-8 relative z-10">
            <div className="max-w-7xl mx-auto text-center">
                <motion.h2
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-3xl sm:text-4xl font-bold mb-12"
                >
                    Powered by <span className="text-neon-cyan">Advanced Tech</span>
                </motion.h2>

                <div className="flex flex-wrap justify-center gap-4">
                    {techs.map((tech, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, scale: 0.8 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.05 }}
                            whileHover={{ scale: 1.1, boxShadow: "0 0 15px rgba(10, 255, 240, 0.4)" }}
                            className="px-6 py-3 rounded-full bg-white/5 border border-white/10 text-neon-cyan font-semibold cursor-default backdrop-blur-sm"
                        >
                            {tech}
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default TechStack;
