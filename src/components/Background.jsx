import React, { useEffect, useRef } from 'react';

const Background = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let width = window.innerWidth;
        let height = window.innerHeight;
        let stars = [];

        const resize = () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
            initStars();
        };

        const initStars = () => {
            stars = [];
            const numStars = Math.floor((width * height) / 3000);
            for (let i = 0; i < numStars; i++) {
                stars.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    radius: Math.random() * 1.5,
                    vx: (Math.random() - 0.5) * 0.2,
                    vy: (Math.random() - 0.5) * 0.2,
                    alpha: Math.random(),
                    targetAlpha: Math.random(),
                });
            }
        };

        const draw = () => {
            ctx.clearRect(0, 0, width, height);

            // Draw Nebula/Gradient Background
            const gradient = ctx.createRadialGradient(width / 2, height / 2, 0, width / 2, height / 2, width);
            gradient.addColorStop(0, '#0a0f1e');
            gradient.addColorStop(1, '#050505');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);

            // Draw Stars
            stars.forEach((star) => {
                ctx.beginPath();
                ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha})`;
                ctx.fill();

                // Update position
                star.x += star.vx;
                star.y += star.vy;

                // Wrap around
                if (star.x < 0) star.x = width;
                if (star.x > width) star.x = 0;
                if (star.y < 0) star.y = height;
                if (star.y > height) star.y = 0;

                // Twinkle effect
                if (Math.abs(star.alpha - star.targetAlpha) < 0.01) {
                    star.targetAlpha = Math.random();
                } else {
                    star.alpha += (star.targetAlpha - star.alpha) * 0.02;
                }
            });

            requestAnimationFrame(draw);
        };

        resize();
        window.addEventListener('resize', resize);
        const animationId = requestAnimationFrame(draw);

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed top-0 left-0 w-full h-full z-0 pointer-events-none"
        />
    );
};

export default Background;
