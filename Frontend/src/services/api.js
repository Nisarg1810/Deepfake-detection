/**
 * API service for communicating with the Flask backend
 */

const API_BASE_URL = 'http://localhost:5000';

/**
 * Upload and analyze a video file
 * @param {File} videoFile - The video file to analyze
 * @param {Function} onProgress - Optional callback for upload progress
 * @returns {Promise<Object>} Analysis result from backend
 */
export const analyzeVideo = async (videoFile, onProgress = null) => {
    const formData = new FormData();
    formData.append('video', videoFile);

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Analysis failed');
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error analyzing video:', error);
        throw error;
    }
};

/**
 * Check backend health status
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('Backend is not responding');
        }
        return await response.json();
    } catch (error) {
        console.error('Health check failed:', error);
        throw error;
    }
};
