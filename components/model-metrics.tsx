"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Target, Activity, BarChart3, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useState, useEffect } from "react";

interface ModelMetricsProps {
  metrics: {
    rmse?: number;
    mae?: number;
    r2_score?: number;
    mse?: number;
  };
}

export default function ModelMetrics({ metrics }: ModelMetricsProps) {
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  // Update timestamp whenever metrics change
  useEffect(() => {
    setLastUpdated(new Date());
  }, [metrics]);
  const getR2Color = (r2: number) => {
    if (r2 >= 0.8) return "text-green-600";
    if (r2 >= 0.6) return "text-blue-600";
    if (r2 >= 0.4) return "text-amber-600";
    return "text-red-600";
  };

  const getR2Label = (r2: number) => {
    if (r2 >= 0.8) return "Excellent";
    if (r2 >= 0.6) return "Good";
    if (r2 >= 0.4) return "Fair";
    return "Poor";
  };

  // Calculate accuracy percentage (inverse of normalized RMSE)
  const calculateAccuracy = () => {
    if (metrics.rmse !== undefined) {
      // Assuming grades are 0-20, normalize RMSE
      const normalizedError = metrics.rmse / 20;
      const accuracy = Math.max(0, (1 - normalizedError) * 100);
      return accuracy;
    }
    return null;
  };

  const accuracy = calculateAccuracy();

  return (
    <Card className="sticky top-4">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Model Performance
            </CardTitle>
            <CardDescription>
              Evaluated on 20% test data ({metrics.rmse ? '~130 students' : 'Not trained yet'})
            </CardDescription>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
            <RefreshCw className="h-3 w-3 animate-spin" style={{ animationDuration: '3s' }} />
            <span>Auto-updating</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Accuracy */}
        {accuracy !== null && (
          <div className="rounded-lg border-2 border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-950/20">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-blue-600" />
                <span className="font-semibold text-blue-900 dark:text-blue-100">Model Accuracy</span>
              </div>
              <Badge className="bg-blue-600 text-white">
                {accuracy >= 80 ? 'Excellent' : accuracy >= 70 ? 'Good' : 'Fair'}
              </Badge>
            </div>
            <div className="mb-2 text-4xl font-bold text-blue-600">
              {accuracy.toFixed(1)}%
            </div>
            <Progress value={accuracy} className="mb-2 h-2" />
            <p className="text-xs text-blue-800 dark:text-blue-200">
              Prediction accuracy based on test data
            </p>
          </div>
        )}

        {/* R² Score */}
        {metrics.r2_score !== undefined && (
          <div className="rounded-lg border p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-zinc-600" />
                <span className="text-sm font-medium">R² Score</span>
              </div>
              <Badge variant="outline" className={getR2Color(metrics.r2_score)}>
                {getR2Label(metrics.r2_score)}
              </Badge>
            </div>
            <div className={`text-3xl font-bold ${getR2Color(metrics.r2_score)}`}>
              {metrics.r2_score.toFixed(4)}
            </div>
            <Progress value={metrics.r2_score * 100} className="my-2 h-2" />
            <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
              {(metrics.r2_score * 100).toFixed(1)}% of grade variance explained
            </p>
          </div>
        )}

        {/* RMSE */}
        {metrics.rmse !== undefined && (
          <div className="rounded-lg border p-4">
            <div className="mb-2 flex items-center gap-2">
              <Target className="h-4 w-4 text-zinc-600" />
              <span className="text-sm font-medium">RMSE</span>
            </div>
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              ±{metrics.rmse.toFixed(2)} points
            </div>
            <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
              Average prediction error (penalizes large errors)
            </p>
          </div>
        )}

        {/* MAE */}
        {metrics.mae !== undefined && (
          <div className="rounded-lg border p-4">
            <div className="mb-2 flex items-center gap-2">
              <Activity className="h-4 w-4 text-zinc-600" />
              <span className="text-sm font-medium">MAE</span>
            </div>
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              ±{metrics.mae.toFixed(2)} points
            </div>
            <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
              Average absolute prediction error
            </p>
          </div>
        )}

        {/* MSE */}
        {metrics.mse !== undefined && (
          <div className="rounded-lg border p-4">
            <div className="mb-2 flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-zinc-600" />
              <span className="text-sm font-medium">MSE</span>
            </div>
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              {metrics.mse.toFixed(4)}
            </div>
            <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
              Mean Squared Error
            </p>
          </div>
        )}

        <div className="space-y-2 rounded-lg bg-zinc-50 p-4 dark:bg-zinc-900/50">
          <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            📊 How to Read These Metrics:
          </p>
          <ul className="space-y-1 text-xs text-zinc-700 dark:text-zinc-300">
            <li className="flex gap-2">
              <span className="font-semibold">Accuracy:</span>
              <span>Overall prediction correctness</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">R²:</span>
              <span>How well model explains grade variations (higher = better)</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">RMSE/MAE:</span>
              <span>Average error in grade points (lower = better)</span>
            </li>
          </ul>
          <p className="mt-2 text-xs italic text-zinc-600 dark:text-zinc-400">
            💡 Example: RMSE of ±2.0 means predictions are typically within 2 points of actual grade
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
