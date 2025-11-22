import React from 'react';

const Footer = () => {
    return (
        <footer className="py-8 px-4 border-t border-white/10 relative z-10 bg-black/40 backdrop-blur-md">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
                <div className="text-gray-400 text-sm">
                    © 2025 DeepSearch AI — Future of Video Integrity.
                </div>

                <div className="flex gap-6">
                    <a href="#" className="text-gray-400 hover:text-neon-blue transition-colors text-sm">GitHub</a>
                    <a href="#" className="text-gray-400 hover:text-neon-blue transition-colors text-sm">Docs</a>
                    <a href="#" className="text-gray-400 hover:text-neon-blue transition-colors text-sm">Contact</a>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
