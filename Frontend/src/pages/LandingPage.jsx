import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Hero from '../components/Hero';
import Features from '../components/Features';
import HowItWorks from '../components/HowItWorks';
import TechStack from '../components/TechStack';
import CTA from '../components/CTA';
import Footer from '../components/Footer';
import Background from '../components/Background';
import CustomCursor from '../components/CustomCursor';
import Loader from '../components/Loader';
import Navbar from '../components/Navbar';

function LandingPage() {
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate loading time
        const timer = setTimeout(() => {
            setLoading(false);
        }, 2500);
        return () => clearTimeout(timer);
    }, []);

    return (
        <>
            <CustomCursor />
            <AnimatePresence mode="wait">
                {loading ? (
                    <Loader key="loader" />
                ) : (
                    <motion.div
                        key="content"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                        className="relative w-full min-h-screen overflow-hidden"
                    >
                        <Background />
                        <Navbar />
                        <main className="relative z-10">
                            <Hero />
                            <Features />
                            <HowItWorks />
                            <TechStack />
                            <CTA />
                        </main>
                        <Footer />
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}

export default LandingPage;
