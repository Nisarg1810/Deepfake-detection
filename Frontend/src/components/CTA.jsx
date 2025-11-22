import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

const CTA = () => {
    return (
        <section className="py-20 px-4 sm:px-6 lg:px-8 relative z-10">
            <div className="max-w-5xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="glass-panel rounded-2xl p-12 text-center border border-neon-blue/30 relative overflow-hidden"
                >
                    {/* Background Glow */}
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full bg-gradient-radial from-neon-blue/10 to-transparent opacity-50 pointer-events-none" />

                    <h2 className="text-3xl sm:text-5xl font-bold mb-6 relative z-10">
                        Ready to Detect <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple">AI Videos?</span>
                    </h2>

                    <p className="text-gray-300 text-lg mb-8 max-w-2xl mx-auto relative z-10">
                        Join thousands of users ensuring content integrity with DeepSearch AI.
                        Start your free trial today.
                    </p>

                    <motion.button
                        whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(188, 19, 254, 0.5)" }}
                        whileTap={{ scale: 0.95 }}
                        className="relative z-10 px-10 py-4 bg-gradient-to-r from-neon-blue to-neon-purple text-white font-bold rounded-lg inline-flex items-center gap-2"
                    >
                        Start Detection
                        <ArrowRight size={20} />
                    </motion.button>
                </motion.div>
            </div>
        </section>
    );
};

export default CTA;
