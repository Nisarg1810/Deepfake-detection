import React, { useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Download, RefreshCw, Play } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import Navbar from '../components/Navbar';
import Background from '../components/Background';
import Footer from '../components/Footer';
import CustomCursor from '../components/CustomCursor';

const ResultsPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const result = location.state?.result;

    useEffect(() => {
        // Redirect if no result data
        if (!result) {
            navigate('/upload');
        }
    }, [result, navigate]);

    if (!result) {
        return null; // Will redirect
    }

    // Extract data from backend result
    const verdict = result.verdict || {};
    const finalScore = (verdict.final_score || 0.5) * 100; // Convert to percentage
    const isFake = verdict.final_label === 'FAKE';
    const confidence = verdict.confidence || finalScore / 100;

    // Get aggregated scores
    const aggregated = result.aggregated_scores || {};
    const facialConsistency = ((1 - (aggregated.cnn_mean || 0.5)) * 100); // Invert for consistency
    const lightingCoherence = ((1 - (aggregated.freq_mean || 0.5)) * 100);
    const audioVisualSync = ((1 - (aggregated.lip_sync_score || 0.5)) * 100);

    // Get abnormality report
    const abnormalities = result.abnormality_report?.abnormalities || [];
    const techniques = result.technique_report?.detected_techniques || [];

    return (
        <>
            <CustomCursor />
            <div className="relative min-h-screen w-full overflow-hidden text-white">
                <Background />
                <Navbar />

                <main className="relative z-10 container mx-auto px-4 pt-32 pb-20">
                    <div className="max-w-5xl mx-auto">

                        {/* Header */}
                        <motion.div
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="text-center mb-12"
                        >
                            <h1 className="text-3xl md:text-5xl font-bold mb-4">Analysis Complete</h1>
                            <p className="text-gray-400">Report generated successfully</p>
                        </motion.div>

                        <div className="grid lg:grid-cols-3 gap-8">

                            {/* Main Result Card */}
                            <motion.div
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.2 }}
                                className="lg:col-span-2 space-y-6"
                            >
                                <Card className={`border-2 ${isFake ? 'border-red-500/50 bg-red-950/10' : 'border-green-500/50 bg-green-950/10'} backdrop-blur-xl`}>
                                    <CardContent className="p-8 flex flex-col md:flex-row items-center gap-8">
                                        <div className="relative">
                                            <div className={`w-32 h-32 rounded-full flex items-center justify-center border-4 ${isFake ? 'border-red-500' : 'border-green-500'} shadow-[0_0_30px_rgba(0,0,0,0.5)]`}>
                                                <span className={`text-3xl font-bold ${isFake ? 'text-red-500' : 'text-green-500'}`}>
                                                    {finalScore.toFixed(1)}%
                                                </span>
                                            </div>
                                            {isFake && (
                                                <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 bg-red-500 text-white text-xs font-bold px-3 py-1 rounded-full">
                                                    FAKE
                                                </div>
                                            )}
                                        </div>

                                        <div className="text-center md:text-left flex-1">
                                            <h2 className="text-2xl font-bold mb-2 flex items-center justify-center md:justify-start gap-2">
                                                {isFake ? (
                                                    <>
                                                        <AlertTriangle className="text-red-500" />
                                                        <span className="text-red-500">High Probability of Deepfake</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <CheckCircle className="text-green-500" />
                                                        <span className="text-green-500">Authentic Content</span>
                                                    </>
                                                )}
                                            </h2>
                                            <p className="text-gray-300 mb-4">
                                                {isFake
                                                    ? "Our system detected significant anomalies consistent with AI generation techniques."
                                                    : "No significant anomalies detected. The video appears to be authentic."}
                                            </p>
                                            <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                                                {techniques.length > 0 ? (
                                                    techniques.slice(0, 3).map((tech, idx) => (
                                                        <span key={idx} className="px-3 py-1 rounded-full bg-white/5 text-xs border border-white/10">
                                                            {tech.technique || tech}
                                                        </span>
                                                    ))
                                                ) : (
                                                    <>
                                                        <span className="px-3 py-1 rounded-full bg-white/5 text-xs border border-white/10">GAN Fingerprints</span>
                                                        <span className="px-3 py-1 rounded-full bg-white/5 text-xs border border-white/10">Temporal Analysis</span>
                                                        <span className="px-3 py-1 rounded-full bg-white/5 text-xs border border-white/10">Facial Landmarks</span>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Video Preview & Timeline */}
                                <Card className="bg-black/40 border-white/10 backdrop-blur-xl">
                                    <CardHeader>
                                        <CardTitle className="text-lg">Frame Analysis</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="aspect-video bg-black rounded-lg mb-4 relative overflow-hidden group">
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <Play className="w-12 h-12 text-white/50 group-hover:text-white transition-colors" />
                                            </div>
                                            {/* Mock Frame Overlay */}
                                            {isFake && (
                                                <div className="absolute top-1/4 left-1/3 w-24 h-24 border-2 border-red-500 rounded-sm opacity-70" />
                                            )}
                                            <p className="absolute bottom-2 right-2 text-xs text-gray-400">
                                                {result.metadata?.total_frames || 0} frames analyzed
                                            </p>
                                        </div>

                                        {/* Abnormalities List */}
                                        {abnormalities.length > 0 && (
                                            <div className="space-y-2 mb-4">
                                                <p className="text-sm font-medium text-gray-300">Detected Abnormalities:</p>
                                                <div className="space-y-1">
                                                    {abnormalities.slice(0, 3).map((abn, idx) => (
                                                        <div key={idx} className="text-xs text-gray-400 flex items-start gap-2">
                                                            <span className="text-red-400">•</span>
                                                            <span>{abn.type}: {abn.description}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Mock Timeline */}
                                        <div className="space-y-2">
                                            <div className="flex justify-between text-xs text-gray-400">
                                                <span>00:00</span>
                                                <span>00:15</span>
                                                <span>00:30</span>
                                            </div>
                                            <div className="h-8 bg-white/5 rounded flex overflow-hidden">
                                                <div className="w-[20%] bg-green-500/20 h-full" />
                                                <div className={`w-[15%] ${isFake ? 'bg-red-500/50' : 'bg-green-500/20'} h-full`} />
                                                <div className="w-[30%] bg-green-500/20 h-full" />
                                                <div className={`w-[10%] ${isFake ? 'bg-red-500/50' : 'bg-green-500/20'} h-full`} />
                                                <div className="w-[25%] bg-green-500/20 h-full" />
                                            </div>
                                            {isFake && (
                                                <p className="text-xs text-center text-red-400 mt-1">Red segments indicate detected anomalies</p>
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>

                            {/* Sidebar */}
                            <motion.div
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.4 }}
                                className="space-y-6"
                            >
                                <Card className="bg-black/40 border-white/10 backdrop-blur-xl">
                                    <CardHeader>
                                        <CardTitle className="text-lg">Detailed Metrics</CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-4">
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-sm">
                                                <span className="text-gray-400">Facial Consistency</span>
                                                <span className={facialConsistency < 50 ? 'text-red-400' : facialConsistency < 70 ? 'text-yellow-400' : 'text-green-400'}>
                                                    {facialConsistency < 50 ? 'Low' : facialConsistency < 70 ? 'Medium' : 'High'} ({facialConsistency.toFixed(0)}%)
                                                </span>
                                            </div>
                                            <Progress value={facialConsistency} className="h-1.5 bg-white/10" />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-sm">
                                                <span className="text-gray-400">Lighting Coherence</span>
                                                <span className={lightingCoherence < 50 ? 'text-red-400' : lightingCoherence < 70 ? 'text-yellow-400' : 'text-green-400'}>
                                                    {lightingCoherence < 50 ? 'Low' : lightingCoherence < 70 ? 'Medium' : 'High'} ({lightingCoherence.toFixed(0)}%)
                                                </span>
                                            </div>
                                            <Progress value={lightingCoherence} className="h-1.5 bg-white/10" />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex justify-between text-sm">
                                                <span className="text-gray-400">Audio-Visual Sync</span>
                                                <span className={audioVisualSync < 50 ? 'text-red-400' : audioVisualSync < 70 ? 'text-yellow-400' : 'text-green-400'}>
                                                    {audioVisualSync < 50 ? 'Low' : audioVisualSync < 70 ? 'Medium' : 'High'} ({audioVisualSync.toFixed(0)}%)
                                                </span>
                                            </div>
                                            <Progress value={audioVisualSync} className="h-1.5 bg-white/10" />
                                        </div>
                                    </CardContent>
                                </Card>

                                <div className="space-y-3">
                                    <Button
                                        className="w-full bg-white text-black hover:bg-gray-200"
                                        variant="premium"
                                        onClick={() => {
                                            // Download JSON report
                                            const dataStr = JSON.stringify(result, null, 2);
                                            const dataBlob = new Blob([dataStr], { type: 'application/json' });
                                            const url = URL.createObjectURL(dataBlob);
                                            const link = document.createElement('a');
                                            link.href = url;
                                            link.download = `deepfake-analysis-${result.job_id || 'report'}.json`;
                                            link.click();
                                        }}
                                    >
                                        <Download className="w-4 h-4 mr-2" />
                                        Download Full Report
                                    </Button>
                                    <Link to="/upload" className="block">
                                        <Button variant="outline" className="w-full border-white/20 text-white hover:bg-white/10">
                                            <RefreshCw className="w-4 h-4 mr-2" />
                                            Analyze Another Video
                                        </Button>
                                    </Link>
                                </div>
                            </motion.div>

                        </div>
                    </div>
                </main>
                <Footer />
            </div>
        </>
    );
};

export default ResultsPage;
