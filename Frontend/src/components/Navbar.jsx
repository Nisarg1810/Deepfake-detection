import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Navbar = () => {
    return (
        <motion.nav
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5 }}
            className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 backdrop-blur-md bg-black/20 border-b border-white/10"
        >
            <Link to="/" className="text-2xl font-bold tracking-tighter">
                <span className="text-white">Deep</span>
                <span className="text-neon-blue">Search</span>
            </Link>

            <div className="hidden md:flex items-center space-x-8">
                <Link to="/" className="text-sm font-medium text-gray-300 hover:text-white transition-colors relative group">
                    Home
                    <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-neon-blue transition-all group-hover:w-full" />
                </Link>
                <Link to="/upload" className="text-sm font-medium text-gray-300 hover:text-white transition-colors relative group">
                    Detect
                    <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-neon-blue transition-all group-hover:w-full" />
                </Link>
                <a href="/#how-it-works" className="text-sm font-medium text-gray-300 hover:text-white transition-colors relative group">
                    How it Works
                    <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-neon-blue transition-all group-hover:w-full" />
                </a>
            </div>

            <Link to="/upload">
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-4 py-2 text-sm font-medium text-black bg-white rounded-full hover:bg-gray-200 transition-colors"
                >
                    Start Detection
                </motion.button>
            </Link>
        </motion.nav>
    );
};

export default Navbar;
