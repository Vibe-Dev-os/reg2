"use client";

import { useState, useEffect } from "react";
import { GraduationCap, Brain, TrendingUp, AlertCircle, BarChart3, Database, Award, Target, Layers, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getModelStatus, getDatasetInfo, trainModel, type ModelStatus, type DatasetInfo } from "@/lib/api";
import PredictionForm from "@/components/prediction-form";
import ModelMetrics from "@/components/model-metrics";
import DemographicsCharts from "@/components/demographics-charts";
import { ThemeToggle } from "@/components/theme-toggle";

export default function Home() {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"predict" | "demographics">("predict");
  const [mounted, setMounted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setMounted(true);
    // Restore active tab from localStorage
    const savedTab = localStorage.getItem("activeTab") as "predict" | "demographics" | null;
    if (savedTab) {
      setActiveTab(savedTab);
    }
    loadStatus();
  }, []);

  // Save active tab to localStorage whenever it changes
  useEffect(() => {
    if (mounted) {
      localStorage.setItem("activeTab", activeTab);
    }
  }, [activeTab, mounted]);

  const handleDownloadDataset = () => {
    // Create a link element and trigger download
    const link = document.createElement('a');
    link.href = '/student-por.csv';
    link.download = 'student-por.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const loadStatus = async () => {
    setIsLoading(true);
    try {
      const [status, info] = await Promise.all([
        getModelStatus(),
        getDatasetInfo(),
      ]);
      setModelStatus(status);
      setDatasetInfo(info);
    } catch (error) {
      console.error("Failed to load status:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setTrainingError(null);
    try {
      await trainModel();
      await loadStatus();
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Training failed");
    } finally {
      setIsTraining(false);
    }
  };

  if (!mounted) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      {/* Header */}
      <header className="border-b bg-white/50 backdrop-blur-sm dark:bg-zinc-900/50">
        <div className="container mx-auto px-4 py-4 md:py-6">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-3 md:gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-zinc-900 dark:bg-zinc-100 md:h-14 md:w-14">
                <TrendingUp className="h-6 w-6 text-white dark:text-zinc-900 md:h-7 md:w-7" />
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="truncate text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 md:text-2xl">
                  Group 2: Regressors
                </h1>
                <p className="text-sm text-zinc-500 dark:text-zinc-400 md:text-base">
                  Predict final grades
                </p>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2 md:gap-3">
              {modelStatus && (
                <Badge 
                  variant={modelStatus.model_trained ? "default" : "secondary"}
                  className="gap-1 px-2 py-1 text-xs md:gap-1.5 md:px-3"
                >
                  {modelStatus.model_trained ? (
                    <>
                      <Brain className="h-3 w-3 md:h-3.5 md:w-3.5" />
                      <span className="font-semibold">Random Forest</span>
                      <span className="hidden opacity-70 md:inline">•</span>
                      <span className="hidden text-xs md:inline">Ready</span>
                    </>
                  ) : (
                    "Not Trained"
                  )}
                </Badge>
              )}
              <Button 
                variant="outline" 
                size="sm"
                className="gap-2"
                onClick={handleDownloadDataset}
              >
                <Download className="h-4 w-4" />
                <span className="hidden md:inline">Download Dataset</span>
              </Button>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Dataset Info Cards */}
        {isLoading ? (
          <div className="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Card key={i} className="overflow-hidden">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex-1 space-y-2">
                      <div className="h-3 w-20 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                      <div className="h-8 w-16 animate-pulse rounded bg-zinc-300 dark:bg-zinc-700" />
                      <div className="h-2 w-24 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                    </div>
                    <div className="h-10 w-10 animate-pulse rounded-full bg-zinc-200 dark:bg-zinc-800" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : datasetInfo ? (
          <div className="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {/* Total Records Card */}
            <Card className="overflow-hidden border-l-4 border-l-blue-500 bg-gradient-to-br from-blue-50 to-white dark:from-blue-950/20 dark:to-zinc-900">
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Total Records
                    </p>
                    <p className="mt-1 text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {datasetInfo.total_records.toLocaleString()}
                    </p>
                    <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-500">
                      Student entries
                    </p>
                  </div>
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                    <Database className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Average Grade Card */}
            <Card className="overflow-hidden border-l-4 border-l-green-500 bg-gradient-to-br from-green-50 to-white dark:from-green-950/20 dark:to-zinc-900">
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Average Grade
                    </p>
                    <p className="mt-1 text-2xl font-bold text-green-600 dark:text-green-400">
                      {datasetInfo.grade_distribution.mean.toFixed(2)}
                    </p>
                    <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-500">
                      Median: {datasetInfo.grade_distribution.median.toFixed(2)}
                    </p>
                  </div>
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                    <Award className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Grade Range Card */}
            <Card className="overflow-hidden border-l-4 border-l-purple-500 bg-gradient-to-br from-purple-50 to-white dark:from-purple-950/20 dark:to-zinc-900">
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Grade Range
                    </p>
                    <p className="mt-1 text-2xl font-bold text-purple-600 dark:text-purple-400">
                      {datasetInfo.grade_distribution.min} - {datasetInfo.grade_distribution.max}
                    </p>
                    <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-500">
                      Out of 20 points
                    </p>
                  </div>
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                    <Target className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Features Card */}
            <Card className="overflow-hidden border-l-4 border-l-amber-500 bg-gradient-to-br from-amber-50 to-white dark:from-amber-950/20 dark:to-zinc-900">
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Features
                    </p>
                    <p className="mt-1 text-2xl font-bold text-amber-600 dark:text-amber-400">
                      {datasetInfo.features.length}
                    </p>
                    <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-500">
                      Data attributes
                    </p>
                  </div>
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900/30">
                    <Layers className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : null}

        {/* Tab Navigation */}
        {isLoading ? (
          <div className="mb-6 flex gap-2">
            <div className="h-10 flex-1 animate-pulse rounded-md bg-zinc-200 dark:bg-zinc-800" />
            <div className="h-10 flex-1 animate-pulse rounded-md bg-zinc-200 dark:bg-zinc-800" />
          </div>
        ) : (
          <div className="mb-6 flex gap-2">
            <Button
              variant={activeTab === "predict" ? "default" : "outline"}
              onClick={() => setActiveTab("predict")}
              className="flex-1 gap-2"
            >
              <TrendingUp className="h-4 w-4" />
              Predict Grade
            </Button>
            <Button
              variant={activeTab === "demographics" ? "default" : "outline"}
              onClick={() => setActiveTab("demographics")}
              className="flex-1 gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              Demographics
            </Button>
          </div>
        )}

        {/* Content Area */}
        {activeTab === "predict" ? (
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              {isLoading ? (
                <Card>
                  <CardHeader>
                    <div className="h-6 w-48 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                    <div className="mt-2 h-4 w-full animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="h-32 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
                    <div className="h-32 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
                  </CardContent>
                </Card>
              ) : modelStatus?.model_trained ? (
                <PredictionForm />
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-amber-500" />
                      Model Not Trained
                    </CardTitle>
                    <CardDescription>
                      The model needs to be trained. Please run the training script or contact the administrator.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-zinc-600 dark:text-zinc-400">
                      To train the model, run: <code className="rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800">python backend/main.py</code> and use the <code className="rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800">/train</code> endpoint.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
            <div>
              {isLoading ? (
                <Card>
                  <CardHeader>
                    <div className="h-6 w-32 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="h-24 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
                    <div className="h-24 animate-pulse rounded bg-zinc-100 dark:bg-zinc-900" />
                  </CardContent>
                </Card>
              ) : (
                modelStatus?.model_trained && modelStatus.metrics && (
                  <ModelMetrics metrics={modelStatus.metrics} />
                )
              )}
            </div>
          </div>
        ) : (
          <DemographicsCharts />
        )}
      </main>
    </div>
  );
}
