/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'space-black': '#050505',
                'deep-blue': '#0a0f1e',
                'neon-blue': '#00f3ff',
                'neon-purple': '#bc13fe',
                'neon-cyan': '#0afff0',
                'glass': 'rgba(255, 255, 255, 0.05)',
            },
            backgroundImage: {
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
                'hero-glow': 'conic-gradient(from 180deg at 50% 50%, #0a0f1e 0deg, #00f3ff 180deg, #bc13fe 360deg)',
            },
            animation: {
                'spin-slow': 'spin 20s linear infinite',
                'pulse-glow': 'pulse-glow 3s infinite',
                'float': 'float 6s ease-in-out infinite',
            },
            keyframes: {
                'pulse-glow': {
                    '0%, 100%': { opacity: 1, boxShadow: '0 0 20px rgba(0, 243, 255, 0.5)' },
                    '50%': { opacity: 0.8, boxShadow: '0 0 40px rgba(0, 243, 255, 0.8)' },
                },
                'float': {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-20px)' },
                }
            }
        },
    },
    plugins: [],
}
