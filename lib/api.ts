const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'https://reg2-boq6.onrender.com').replace(/\/$/, '');

export interface StudentData {
  school: string;
  sex: string;
  age: number;
  address: string;
  famsize: string;
  Pstatus: string;
  Medu: number;
  Fedu: number;
  Mjob: string;
  Fjob: string;
  reason: string;
  guardian: string;
  traveltime: number;
  studytime: number;
  failures: number;
  schoolsup: string;
  famsup: string;
  paid: string;
  activities: string;
  nursery: string;
  higher: string;
  internet: string;
  romantic: string;
  famrel: number;
  freetime: number;
  goout: number;
  Dalc: number;
  Walc: number;
  health: number;
  absences: number;
  G1: number;
  G2: number;
}

export interface PredictionResponse {
  predicted_grade: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  feature_importance: Record<string, number>;
}

export interface TrainingResponse {
  message: string;
  metrics: {
    rmse: number;
    mae: number;
    r2_score: number;
    mse: number;
  };
  feature_importance: Record<string, number>;
}

export interface ModelStatus {
  model_trained: boolean;
  model_loaded: boolean;
  metrics: {
    rmse?: number;
    mae?: number;
    r2_score?: number;
    mse?: number;
  };
}

export interface DatasetInfo {
  total_records: number;
  features: string[];
  grade_distribution: {
    min: number;
    max: number;
    mean: number;
    median: number;
  };
}

export async function trainModel(): Promise<TrainingResponse> {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error('Failed to train model');
  }

  return response.json();
}

export async function predictGrade(studentData: StudentData): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(studentData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to predict grade' }));
    throw new Error(errorData.detail || 'Failed to predict grade');
  }

  return response.json();
}

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await fetch(`${API_BASE_URL}/model/status`);

  if (!response.ok) {
    throw new Error('Failed to get model status');
  }

  return response.json();
}

export async function getDatasetInfo(): Promise<DatasetInfo> {
  const response = await fetch(`${API_BASE_URL}/dataset/info`);

  if (!response.ok) {
    throw new Error('Failed to get dataset info');
  }

  return response.json();
}
