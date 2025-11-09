"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, TrendingUp, TrendingDown, User, Users, BookOpen, Heart, Award, CheckCircle2 } from "lucide-react";
import { predictGrade, type StudentData, type PredictionResponse } from "@/lib/api";
import { toast } from "sonner";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export default function PredictionForm() {
  const [formData, setFormData] = useState<StudentData>({
    school: "GP",
    sex: "F",
    age: 17,
    address: "U",
    famsize: "GT3",
    Pstatus: "T",
    Medu: 2,
    Fedu: 2,
    Mjob: "other",
    Fjob: "other",
    reason: "course",
    guardian: "mother",
    traveltime: 1,
    studytime: 2,
    failures: 0,
    schoolsup: "no",
    famsup: "yes",
    paid: "no",
    activities: "no",
    nursery: "yes",
    higher: "yes",
    internet: "yes",
    romantic: "no",
    famrel: 4,
    freetime: 3,
    goout: 3,
    Dalc: 1,
    Walc: 1,
    health: 5,
    absences: 0,
    G1: 10,
    G2: 10,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const validateForm = (): boolean => {
    // Check all required fields
    const requiredFields: (keyof StudentData)[] = [
      'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
      'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
      'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
      'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
      'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
      'G1', 'G2'
    ];

    for (const field of requiredFields) {
      const value = formData[field];
      if (value === undefined || value === null || value === '') {
        toast.error('Missing Required Field', {
          description: `Please fill in the ${field} field.`,
        });
        return false;
      }
    }

    // Validate numeric ranges
    if (formData.age < 15 || formData.age > 22) {
      toast.error('Invalid Age', {
        description: 'Age must be between 15 and 22.',
      });
      return false;
    }

    if (formData.Medu < 0 || formData.Medu > 4) {
      toast.error('Invalid Mother Education', {
        description: 'Mother education must be between 0 and 4.',
      });
      return false;
    }

    if (formData.Fedu < 0 || formData.Fedu > 4) {
      toast.error('Invalid Father Education', {
        description: 'Father education must be between 0 and 4.',
      });
      return false;
    }

    if (formData.G1 < 0 || formData.G1 > 20) {
      toast.error('Invalid G1 Grade', {
        description: 'G1 must be between 0 and 20.',
      });
      return false;
    }

    if (formData.G2 < 0 || formData.G2 > 20) {
      toast.error('Invalid G2 Grade', {
        description: 'G2 must be between 0 and 20.',
      });
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await predictGrade(formData);
      setPrediction(result);
      setIsDialogOpen(true);
      toast.success('Prediction Complete!', {
        description: `Predicted grade: ${result.predicted_grade.toFixed(1)}`,
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Prediction failed";
      setError(errorMessage);
      toast.error('Prediction Failed', {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: keyof StudentData, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Student Information</CardTitle>
          <CardDescription>
            Enter student details to predict their final grade (G3)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <Accordion type="multiple" defaultValue={["demographics", "family", "study", "social", "grades"]} className="space-y-4">
              {/* Demographics */}
              <AccordionItem value="demographics" className="rounded-lg border bg-zinc-50 px-4 dark:bg-zinc-900/50">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                      <User className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    </div>
                    <span className="font-semibold">Student Demographics</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">School</label>
                      <Select value={formData.school} onValueChange={(value) => handleInputChange("school", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select school" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="GP">Gabriel Pereira</SelectItem>
                          <SelectItem value="MS">Mousinho da Silveira</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Sex</label>
                      <Select value={formData.sex} onValueChange={(value) => handleInputChange("sex", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select sex" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="F">Female</SelectItem>
                          <SelectItem value="M">Male</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Age</label>
                      <Input
                        type="number"
                        min="15"
                        max="22"
                        value={formData.age}
                        onChange={(e) => handleInputChange("age", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Address Type</label>
                      <Select value={formData.address} onValueChange={(value) => handleInputChange("address", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select address type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="U">Urban</SelectItem>
                          <SelectItem value="R">Rural</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Family Background */}
              <AccordionItem value="family" className="rounded-lg border bg-zinc-50 px-4 dark:bg-zinc-900/50">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                      <Users className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <span className="font-semibold">Family Background</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Family Size</label>
                      <Select value={formData.famsize} onValueChange={(value) => handleInputChange("famsize", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select family size" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="LE3">≤ 3</SelectItem>
                          <SelectItem value="GT3">&gt; 3</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Parent Status</label>
                      <Select value={formData.Pstatus} onValueChange={(value) => handleInputChange("Pstatus", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select parent status" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="T">Together</SelectItem>
                          <SelectItem value="A">Apart</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Mother&apos;s Education (0-4)</label>
                      <Input
                        type="number"
                        min="0"
                        max="4"
                        value={formData.Medu}
                        onChange={(e) => handleInputChange("Medu", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Father&apos;s Education (0-4)</label>
                      <Input
                        type="number"
                        min="0"
                        max="4"
                        value={formData.Fedu}
                        onChange={(e) => handleInputChange("Fedu", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Mother&apos;s Job</label>
                      <Select value={formData.Mjob} onValueChange={(value) => handleInputChange("Mjob", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select mother's job" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="teacher">Teacher</SelectItem>
                          <SelectItem value="health">Health</SelectItem>
                          <SelectItem value="services">Services</SelectItem>
                          <SelectItem value="at_home">At Home</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Father&apos;s Job</label>
                      <Select value={formData.Fjob} onValueChange={(value) => handleInputChange("Fjob", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select father's job" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="teacher">Teacher</SelectItem>
                          <SelectItem value="health">Health</SelectItem>
                          <SelectItem value="services">Services</SelectItem>
                          <SelectItem value="at_home">At Home</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Reason for School Choice</label>
                      <Select value={formData.reason} onValueChange={(value) => handleInputChange("reason", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select reason" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="course">Course</SelectItem>
                          <SelectItem value="home">Close to Home</SelectItem>
                          <SelectItem value="reputation">School Reputation</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Guardian</label>
                      <Select value={formData.guardian} onValueChange={(value) => handleInputChange("guardian", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select guardian" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mother">Mother</SelectItem>
                          <SelectItem value="father">Father</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Study Habits */}
              <AccordionItem value="study" className="rounded-lg border bg-zinc-50 px-4 dark:bg-zinc-900/50">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                      <BookOpen className="h-5 w-5 text-green-600 dark:text-green-400" />
                    </div>
                    <span className="font-semibold">Study Habits</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Travel Time (1-4)</label>
                      <Input
                        type="number"
                        min="1"
                        max="4"
                        value={formData.traveltime}
                        onChange={(e) => handleInputChange("traveltime", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Study Time (1-4)</label>
                      <Input
                        type="number"
                        min="1"
                        max="4"
                        value={formData.studytime}
                        onChange={(e) => handleInputChange("studytime", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Past Failures (0-4)</label>
                      <Input
                        type="number"
                        min="0"
                        max="4"
                        value={formData.failures}
                        onChange={(e) => handleInputChange("failures", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">School Support</label>
                      <Select value={formData.schoolsup} onValueChange={(value) => handleInputChange("schoolsup", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Family Support</label>
                      <Select value={formData.famsup} onValueChange={(value) => handleInputChange("famsup", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Extra Paid Classes</label>
                      <Select value={formData.paid} onValueChange={(value) => handleInputChange("paid", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Social & Health */}
              <AccordionItem value="social" className="rounded-lg border bg-zinc-50 px-4 dark:bg-zinc-900/50">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900/30">
                      <Heart className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                    </div>
                    <span className="font-semibold">Social Factors & Health</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Activities</label>
                      <Select value={formData.activities} onValueChange={(value) => handleInputChange("activities", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Attended Nursery School</label>
                      <Select value={formData.nursery} onValueChange={(value) => handleInputChange("nursery", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Wants Higher Education</label>
                      <Select value={formData.higher} onValueChange={(value) => handleInputChange("higher", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Internet at Home</label>
                      <Select value={formData.internet} onValueChange={(value) => handleInputChange("internet", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Romantic Relationship</label>
                      <Select value={formData.romantic} onValueChange={(value) => handleInputChange("romantic", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select option" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Family Relationship (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.famrel}
                        onChange={(e) => handleInputChange("famrel", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Free Time (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.freetime}
                        onChange={(e) => handleInputChange("freetime", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Going Out (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.goout}
                        onChange={(e) => handleInputChange("goout", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Workday Alcohol (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.Dalc}
                        onChange={(e) => handleInputChange("Dalc", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Weekend Alcohol (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.Walc}
                        onChange={(e) => handleInputChange("Walc", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Health Status (1-5)</label>
                      <Input
                        type="number"
                        min="1"
                        max="5"
                        value={formData.health}
                        onChange={(e) => handleInputChange("health", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Absences</label>
                      <Input
                        type="number"
                        min="0"
                        value={formData.absences}
                        onChange={(e) => handleInputChange("absences", parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              {/* Previous Grades */}
              <AccordionItem value="grades" className="rounded-lg border bg-zinc-50 px-4 dark:bg-zinc-900/50">
                <AccordionTrigger className="hover:no-underline">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-rose-100 dark:bg-rose-900/30">
                      <Award className="h-5 w-5 text-rose-600 dark:text-rose-400" />
                    </div>
                    <span className="font-semibold">Previous Grades</span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="space-y-4 pt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">First Period Grade (G1, 0-20)</label>
                      <Input
                        type="number"
                        min="0"
                        max="20"
                        value={formData.G1}
                        onChange={(e) => handleInputChange("G1", parseInt(e.target.value))}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Second Period Grade (G2, 0-20)</label>
                      <Input
                        type="number"
                        min="0"
                        max="20"
                        value={formData.G2}
                        onChange={(e) => handleInputChange("G2", parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            {error && (
              <div className="rounded-lg bg-red-50 p-4 text-red-900 dark:bg-red-950/20 dark:text-red-100">
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            <Button type="submit" disabled={isLoading} className="w-full gap-2" size="lg">
              <Sparkles className="h-5 w-5" />
              {isLoading ? "Predicting..." : "Predict Final Grade"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Prediction Result Modal */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <CheckCircle2 className="h-6 w-6 text-green-600" />
              Prediction Result
            </DialogTitle>
            <DialogDescription>
              Based on the student information provided, here's the predicted final grade.
            </DialogDescription>
          </DialogHeader>
          
          {prediction && (
            <div className="space-y-6 py-4">
              <div className="flex items-center justify-center">
                <div className="text-center">
                  <p className="text-sm text-zinc-600 dark:text-zinc-400">Predicted Final Grade (G3)</p>
                  <div className="mt-2 text-7xl font-bold text-blue-600 dark:text-blue-400">
                    {prediction.predicted_grade.toFixed(1)}
                  </div>
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <Badge variant="outline" className="text-base px-4 py-1">
                      {prediction.predicted_grade >= 10 ? (
                        <span className="flex items-center gap-1.5 text-green-600">
                          <TrendingUp className="h-4 w-4" />
                          Passing Grade
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-red-600">
                          <TrendingDown className="h-4 w-4" />
                          At Risk
                        </span>
                      )}
                    </Badge>
                  </div>
                </div>
              </div>

              <div className="rounded-lg bg-zinc-50 p-4 dark:bg-zinc-900">
                <h4 className="mb-3 font-semibold text-lg">Confidence Interval (95%)</h4>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="font-medium">Lower: {prediction.confidence_interval.lower.toFixed(2)}</span>
                  <span className="font-medium">Upper: {prediction.confidence_interval.upper.toFixed(2)}</span>
                </div>
                <div className="mt-3 h-3 w-full rounded-full bg-zinc-200 dark:bg-zinc-700">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-600"
                    style={{
                      width: `${((prediction.predicted_grade - prediction.confidence_interval.lower) / (prediction.confidence_interval.upper - prediction.confidence_interval.lower)) * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <h4 className="mb-3 font-semibold text-lg">Top Influential Features</h4>
                <div className="space-y-3">
                  {Object.entries(prediction.feature_importance).map(([feature, importance]) => (
                    <div key={feature} className="flex items-center gap-3">
                      <span className="w-28 text-sm font-medium text-zinc-700 dark:text-zinc-300">{feature}</span>
                      <div className="flex-1">
                        <div className="h-3 w-full rounded-full bg-zinc-200 dark:bg-zinc-700">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-blue-600 to-purple-600"
                            style={{ width: `${importance * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-sm font-semibold w-12 text-right">{(importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
