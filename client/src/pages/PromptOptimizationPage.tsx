import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Sparkles,
  Play,
  AlertCircle,
  CheckCircle,
  Clock,
  Loader2,
  History,
  ArrowRight,
  Copy,
  ChevronDown,
  ChevronUp,
  FileText,
  Link,
  Zap,
  Download,
  BookOpen,
} from 'lucide-react';
import { useWorkshopContext } from '@/context/WorkshopContext';
import { useUser, useRoleCheck } from '@/context/UserContext';
import { useWorkshop } from '@/hooks/useWorkshopApi';
import { getModelOptions, getBackendModelName } from '@/utils/modelMapping';
import { toast } from 'sonner';

interface OptimizationRun {
  id: string;
  job_id: string;
  prompt_uri: string;
  original_prompt: string | null;
  optimized_prompt: string | null;
  optimized_version: number | null;
  optimized_uri: string | null;
  optimizer_model: string | null;
  num_iterations: number | null;
  num_candidates: number | null;
  metrics: Record<string, any> | null;
  status: string;
  error: string | null;
  created_at: string | null;
}

export function PromptOptimizationPage() {
  const { workshopId } = useWorkshopContext();
  const { user } = useUser();
  const { isFacilitator } = useRoleCheck();
  const { data: workshop } = useWorkshop(workshopId!);

  // Configuration state
  const [promptInputMode, setPromptInputMode] = useState<'text' | 'uri'>('text');
  const [promptText, setPromptText] = useState('');
  const [promptUri, setPromptUri] = useState('');
  const [promptName, setPromptName] = useState('');
  const [ucCatalog, setUcCatalog] = useState('');
  const [ucSchema, setUcSchema] = useState('');
  const [optimizerModel, setOptimizerModel] = useState('Claude Sonnet 4.5');
  const [numIterations, setNumIterations] = useState(3);
  const [numCandidates, setNumCandidates] = useState(5);
  const [targetEndpoint, setTargetEndpoint] = useState('');

  // Job state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [jobResult, setJobResult] = useState<any>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const logIndexRef = useRef(0);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // History state
  const [history, setHistory] = useState<OptimizationRun[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [selectedRun, setSelectedRun] = useState<OptimizationRun | null>(null);

  // MLflow config check
  const [hasMlflowConfig, setHasMlflowConfig] = useState(false);

  // Generated skills state
  const [generatedSkills, setGeneratedSkills] = useState<any[]>([]);
  const [expandedSkill, setExpandedSkill] = useState<string | null>(null);

  // Skills generation job state
  const [skillsJobId, setSkillsJobId] = useState<string | null>(null);
  const [skillsJobStatus, setSkillsJobStatus] = useState<string | null>(null);
  const [skillsJobLogs, setSkillsJobLogs] = useState<string[]>([]);
  const [isGeneratingSkills, setIsGeneratingSkills] = useState(false);
  const skillsLogIndexRef = useRef(0);

  // Auto-scroll logs to bottom when new logs arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [jobLogs]);

  const modelOptions = getModelOptions(hasMlflowConfig);

  // Check MLflow config
  useEffect(() => {
    if (!workshopId) return;
    fetch(`/workshops/${workshopId}/mlflow-config`)
      .then(r => r.ok ? r.json() : null)
      .then(config => setHasMlflowConfig(!!config))
      .catch(() => setHasMlflowConfig(false));
  }, [workshopId]);

  // Load history
  const loadHistory = useCallback(async () => {
    if (!workshopId) return;
    try {
      const response = await fetch(`/workshops/${workshopId}/prompt-optimization-history`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (e) {
      console.error('Failed to load optimization history:', e);
    }
  }, [workshopId]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Load generated skills
  const loadGeneratedSkills = useCallback(async () => {
    if (!workshopId) return;
    try {
      const response = await fetch(`/workshops/${workshopId}/generated-skills`);
      if (response.ok) {
        const data = await response.json();
        setGeneratedSkills(data);
      }
    } catch (e) {
      console.error('Failed to load generated skills:', e);
    }
  }, [workshopId]);

  useEffect(() => {
    loadGeneratedSkills();
  }, [loadGeneratedSkills]);

  // On mount: recover the latest skills generation job so polling resumes
  // after tab switches or page reloads.
  useEffect(() => {
    if (!workshopId) return;
    if (skillsJobId) return; // already tracking a job

    (async () => {
      try {
        const response = await fetch(`/workshops/${workshopId}/latest-skills-generation-job`);
        if (!response.ok) return;
        const data = await response.json();
        if (!data.job_id || data.status === 'none') return;

        setSkillsJobId(data.job_id);
        setSkillsJobStatus(data.status);
        if (data.logs && data.logs.length > 0) {
          setSkillsJobLogs(data.logs);
          skillsLogIndexRef.current = data.log_count || data.logs.length;
        }
      } catch (e) {
        console.error('Failed to recover skills generation job:', e);
      }
    })();
  }, [workshopId, skillsJobId]);

  // Reload skills when a job completes (optimization or skills generation)
  useEffect(() => {
    if (jobStatus === 'completed' || skillsJobStatus === 'completed') {
      loadGeneratedSkills();
    }
  }, [jobStatus, skillsJobStatus, loadGeneratedSkills]);

  // Poll job status
  useEffect(() => {
    if (!jobId || !workshopId || jobStatus === 'completed' || jobStatus === 'failed') return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `/workshops/${workshopId}/prompt-optimization-job/${jobId}?since_log_index=${logIndexRef.current}`
        );
        if (!response.ok) return;

        const data = await response.json();

        if (data.logs && data.logs.length > 0) {
          setJobLogs(prev => [...prev, ...data.logs]);
          logIndexRef.current = data.log_count;
        }

        setJobStatus(data.status);

        if (data.result) {
          setJobResult(data.result);
        }
        if (data.error) {
          setJobError(data.error);
        }

        if (data.status === 'completed' || data.status === 'failed') {
          loadHistory();
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, workshopId, jobStatus, loadHistory]);

  const handleStartOptimization = async () => {
    if (!workshopId) return;

    const hasPrompt = promptInputMode === 'text' ? promptText.trim() : promptUri.trim();
    if (!hasPrompt) {
      toast.error(promptInputMode === 'text' ? 'Please enter your agent prompt' : 'Please enter a prompt URI');
      return;
    }

    setIsStarting(true);
    setJobLogs([]);
    setJobResult(null);
    setJobError(null);
    logIndexRef.current = 0;

    try {
      const body: Record<string, any> = {
        optimizer_model_name: getBackendModelName(optimizerModel),
        num_iterations: numIterations,
        num_candidates: numCandidates,
        judge_name: workshop?.judge_name || 'workshop_judge',
      };

      if (targetEndpoint.trim()) {
        body.target_endpoint = targetEndpoint.trim();
      }

      if (promptInputMode === 'text') {
        body.prompt_text = promptText;
        if (promptName.trim()) {
          body.prompt_name = promptName.trim();
        }
        if (ucCatalog.trim()) {
          body.uc_catalog = ucCatalog.trim();
        }
        if (ucSchema.trim()) {
          body.uc_schema = ucSchema.trim();
        }
      } else {
        body.prompt_uri = promptUri;
      }

      const response = await fetch(`/workshops/${workshopId}/start-prompt-optimization`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start optimization');
      }

      const data = await response.json();
      setJobId(data.job_id);
      setJobStatus('running');
      toast.success('Prompt optimization started');
    } catch (e: any) {
      toast.error(e.message || 'Failed to start optimization');
      setJobError(e.message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStartSkillsGeneration = async () => {
    if (!workshopId) return;

    setIsGeneratingSkills(true);
    setSkillsJobLogs([]);
    skillsLogIndexRef.current = 0;

    try {
      const body: Record<string, any> = {
        generation_model_name: getBackendModelName(optimizerModel),
        judge_name: workshop?.judge_name || 'workshop_judge',
      };

      // Only include prompt_text if user has explicitly entered one
      // Otherwise, backend will auto-select optimized or current prompt
      if (promptInputMode === 'text' && promptText.trim()) {
        body.prompt_text = promptText.trim();
      }

      const response = await fetch(`/workshops/${workshopId}/start-skills-generation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start skills generation');
      }

      const data = await response.json();
      setSkillsJobId(data.job_id);
      setSkillsJobStatus('running');
      toast.success('Skills generation started');
    } catch (e: any) {
      toast.error(e.message || 'Failed to start skills generation');
      setSkillsJobStatus('failed');
    } finally {
      setIsGeneratingSkills(false);
    }
  };

  // Poll skills generation job
  useEffect(() => {
    if (!skillsJobId || !workshopId) return;
    if (skillsJobStatus === 'completed' || skillsJobStatus === 'failed') return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `/workshops/${workshopId}/skills-generation-job/${skillsJobId}?since_log_index=${skillsLogIndexRef.current}`
        );
        if (!response.ok) return;

        const data = await response.json();
        setSkillsJobStatus(data.status);

        if (data.logs && data.logs.length > 0) {
          setSkillsJobLogs((prev) => [...prev, ...data.logs]);
          skillsLogIndexRef.current = data.log_count;
        }

        if (data.status === 'completed' && data.result?.success) {
          toast.success(`Skills generated: ${data.result.skills_count} skills`);
          // Reload skills list
          loadGeneratedSkills();
        } else if (data.status === 'failed') {
          toast.error(data.error || 'Skills generation failed');
        }
      } catch (e) {
        console.error('Skills poll error:', e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [skillsJobId, workshopId, skillsJobStatus]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  };

  const isRunning = jobStatus === 'running';
  const isSkillsRunning = skillsJobStatus === 'running';

  if (!isFacilitator) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <Sparkles className="h-12 w-12 mx-auto mb-4 text-violet-400" />
            <h3 className="text-lg font-semibold mb-2">Prompt Optimization</h3>
            <p className="text-sm text-gray-500">
              The facilitator is optimizing the agent's system prompt using GEPA.
              Please wait for this phase to complete.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Sparkles className="h-6 w-6 text-violet-600" />
          Prompt Optimization (GEPA)
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Use MLflow's GEPA optimizer to improve the agent's system prompt based on human evaluation feedback.
        </p>
      </div>

      {/* Configuration Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Configuration</CardTitle>
          <CardDescription>
            Configure the GEPA optimization parameters. The aligned judge from the previous phase will be used as the scorer.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Prompt Input Mode Toggle */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">
              Agent System Prompt
            </label>
            <div className="flex gap-1 mb-3 bg-gray-100 rounded-lg p-1 w-fit">
              <button
                onClick={() => setPromptInputMode('text')}
                disabled={isRunning}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  promptInputMode === 'text'
                    ? 'bg-white text-violet-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <FileText className="h-3.5 w-3.5" />
                Enter Prompt
              </button>
              <button
                onClick={() => setPromptInputMode('uri')}
                disabled={isRunning}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  promptInputMode === 'uri'
                    ? 'bg-white text-violet-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Link className="h-3.5 w-3.5" />
                Load from MLflow
              </button>
            </div>

            {promptInputMode === 'text' ? (
              <div className="space-y-3">
                <Textarea
                  placeholder="Paste your agent's system prompt here...&#10;&#10;Example:&#10;You are a helpful customer support agent. Your job is to..."
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  disabled={isRunning}
                  className="font-mono text-sm min-h-[200px] resize-y"
                />
                {/* UC Catalog, Schema, and Prompt Name */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="text-xs font-medium text-gray-500 block mb-1">UC Catalog</label>
                    <Input
                      placeholder="e.g. main"
                      value={ucCatalog}
                      onChange={(e) => setUcCatalog(e.target.value)}
                      disabled={isRunning}
                      className="text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 block mb-1">UC Schema</label>
                    <Input
                      placeholder="e.g. my_schema"
                      value={ucSchema}
                      onChange={(e) => setUcSchema(e.target.value)}
                      disabled={isRunning}
                      className="text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-gray-500 block mb-1">Prompt Name</label>
                    <Input
                      placeholder="e.g. my_agent_prompt"
                      value={promptName}
                      onChange={(e) => setPromptName(e.target.value)}
                      disabled={isRunning}
                      className="text-sm"
                    />
                  </div>
                </div>
                <p className="text-xs text-gray-400">
                  {ucCatalog && ucSchema && promptName
                    ? `Will register as: ${ucCatalog}.${ucSchema}.${promptName}`
                    : ucCatalog && ucSchema
                    ? `Will register as: ${ucCatalog}.${ucSchema}.<auto_generated_name>`
                    : 'Enter UC catalog and schema to register prompts to Unity Catalog (required for Databricks)'}
                  {promptText ? ` | ${promptText.length} characters` : ''}
                </p>
              </div>
            ) : (
              <div>
                <Input
                  placeholder="prompts:/catalog.schema.my_agent_prompt/1"
                  value={promptUri}
                  onChange={(e) => setPromptUri(e.target.value)}
                  disabled={isRunning}
                  className="font-mono text-sm"
                />
                <p className="text-xs text-gray-400 mt-1">
                  The registered MLflow prompt to optimize (e.g., prompts:/main.my_schema.agent_prompt/latest)
                </p>
              </div>
            )}
          </div>

          {/* Target Endpoint */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">
              Target Endpoint <span className="text-xs text-gray-400 font-normal">(optional)</span>
            </label>
            <Input
              placeholder="e.g. my-agent-endpoint or https://host/serving-endpoints/name/invocations"
              value={targetEndpoint}
              onChange={(e) => setTargetEndpoint(e.target.value)}
              disabled={isRunning}
              className="text-sm font-mono"
            />
            <p className="text-xs text-gray-400 mt-1">
              Serving endpoint name or full invocation URL. Leave blank to use the optimizer model.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Optimizer Model */}
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">
                Optimizer Model
              </label>
              <Select
                value={optimizerModel}
                onValueChange={setOptimizerModel}
                disabled={isRunning}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {modelOptions
                    .filter(m => !m.disabled)
                    .map(option => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>

            {/* Iterations */}
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">
                Iterations ({numIterations})
              </label>
              <input
                type="range"
                min={1}
                max={10}
                value={numIterations}
                onChange={(e) => setNumIterations(Number(e.target.value))}
                disabled={isRunning}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>1</span>
                <span>10</span>
              </div>
            </div>

            {/* Candidates */}
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">
                Candidates per iteration ({numCandidates})
              </label>
              <input
                type="range"
                min={2}
                max={20}
                value={numCandidates}
                onChange={(e) => setNumCandidates(Number(e.target.value))}
                disabled={isRunning}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>2</span>
                <span>20</span>
              </div>
            </div>
          </div>

          {/* Start Button */}
          <div className="flex items-center gap-3">
            <Button
              onClick={handleStartOptimization}
              disabled={isRunning || isStarting || !(promptInputMode === 'text' ? promptText.trim() : promptUri.trim())}
              className=""
            >
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Optimizing...
                </>
              ) : isStarting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Optimization
                </>
              )}
            </Button>

            <Button
              onClick={handleStartSkillsGeneration}
              disabled={isSkillsRunning || isGeneratingSkills}
              variant="outline"
              className="border-blue-300 text-blue-700 hover:bg-blue-50"
            >
              {isSkillsRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : isGeneratingSkills ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <BookOpen className="h-4 w-4 mr-2" />
                  Generate Skills
                </>
              )}
            </Button>

            {jobStatus && (
              <Badge
                variant={
                  jobStatus === 'completed' ? 'default' :
                  jobStatus === 'failed' ? 'destructive' :
                  'secondary'
                }
                className="flex items-center gap-1"
              >
                {jobStatus === 'completed' && <CheckCircle className="h-3 w-3" />}
                {jobStatus === 'failed' && <AlertCircle className="h-3 w-3" />}
                {jobStatus === 'running' && <Clock className="h-3 w-3" />}
                {jobStatus}
              </Badge>
            )}

            {skillsJobStatus && (
              <Badge
                variant={
                  skillsJobStatus === 'completed' ? 'default' :
                  skillsJobStatus === 'failed' ? 'destructive' :
                  'secondary'
                }
                className="flex items-center gap-1 bg-blue-100 text-blue-700 border-blue-300"
              >
                {skillsJobStatus === 'completed' && <CheckCircle className="h-3 w-3" />}
                {skillsJobStatus === 'failed' && <AlertCircle className="h-3 w-3" />}
                {skillsJobStatus === 'running' && <Clock className="h-3 w-3" />}
                Skills: {skillsJobStatus}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Progress / Logs Card */}
      {jobLogs.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              {isRunning ? (
                <Loader2 className="h-5 w-5 animate-spin text-violet-600" />
              ) : jobStatus === 'completed' ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : jobStatus === 'failed' ? (
                <AlertCircle className="h-5 w-5 text-red-600" />
              ) : null}
              Optimization Progress
            </CardTitle>
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-gray-500 hover:text-gray-700"
                onClick={() => copyToClipboard(jobLogs.join('\n'))}
              >
                <Copy className="h-3.5 w-3.5 mr-1" />
                <span className="text-xs">Copy</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-gray-500 hover:text-gray-700"
                onClick={() => {
                  const blob = new Blob([jobLogs.join('\n')], { type: 'text/plain' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `gepa-optimization-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <Download className="h-3.5 w-3.5 mr-1" />
                <span className="text-xs">Download</span>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div
              ref={logContainerRef}
              className="bg-gray-900 rounded-lg p-4 max-h-[400px] overflow-y-auto border border-gray-700 shadow-inner"
            >
              <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
                {jobLogs.map((log, i) => (
                  <div
                    key={i}
                    className={`${
                      log.includes('ERROR') ? 'text-red-400 font-semibold' :
                      log.includes('WARNING') ? 'text-yellow-400' :
                      log.includes('━━━') ? 'text-violet-400 font-semibold' :
                      log.includes('Proposed new text') ? 'text-orange-400 font-semibold' :
                      log.includes('Iteration ') && (log.includes('score') || log.includes('Score')) ? 'text-amber-400 font-medium' :
                      log.includes('Iteration ') && log.includes('Best') ? 'text-emerald-400 font-semibold' :
                      log.includes('Iteration ') ? 'text-amber-300' :
                      log.includes('Candidate Prompt #') ? 'text-cyan-400 font-semibold' :
                      log.includes('Candidate #') && log.includes('evaluated') ? 'text-cyan-400 font-medium' :
                      log.includes('[Predict #') && log.includes('Input:') ? 'text-blue-400' :
                      log.includes('[Predict #') && log.includes('Output:') ? 'text-blue-300' :
                      log.includes('[Predict #') ? 'text-blue-400' :
                      log.includes('Score improvement') || log.includes('Best program') || log.includes('Best score') ? 'text-emerald-400 font-semibold' :
                      log.includes('pareto front') ? 'text-amber-300' :
                      log.includes('complete') || log.includes('success') || log.includes('Optimization complete') ? 'text-green-400 font-semibold' :
                      log.includes('GEPA still optimizing') ? 'text-gray-500' :
                      log.startsWith('  ') ? 'text-gray-400' :
                      'text-green-400'
                    }`}
                  >
                    {log}
                  </div>
                ))}
              </pre>
            </div>
            {isRunning && (
              <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>{jobLogs.length} log entries | Polling every 2s</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Skills Generation Progress / Logs Card */}
      {skillsJobLogs.length > 0 && (
        <Card className="border-blue-200">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              {isSkillsRunning ? (
                <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
              ) : skillsJobStatus === 'completed' ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : skillsJobStatus === 'failed' ? (
                <AlertCircle className="h-5 w-5 text-red-600" />
              ) : null}
              Skills Generation Progress
            </CardTitle>
            <div className="flex gap-1">
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-gray-500 hover:text-gray-700"
                onClick={() => copyToClipboard(skillsJobLogs.join('\n'))}
              >
                <Copy className="h-3.5 w-3.5 mr-1" />
                <span className="text-xs">Copy</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-gray-500 hover:text-gray-700"
                onClick={() => {
                  const blob = new Blob([skillsJobLogs.join('\n')], { type: 'text/plain' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `skills-generation-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <Download className="h-3.5 w-3.5 mr-1" />
                <span className="text-xs">Download</span>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="bg-gray-950 p-4 rounded overflow-y-auto max-h-96">
              <pre className="text-sm text-blue-400 font-mono whitespace-pre-wrap">
                {skillsJobLogs.map((log, i) => (
                  <div
                    key={i}
                    className={`${
                      log.includes('ERROR') ? 'text-red-400 font-semibold' :
                      log.includes('WARNING') ? 'text-yellow-400' :
                      log.includes('======') ? 'text-blue-400 font-semibold' :
                      log.includes('Source 1:') || log.includes('Source 2:') || log.includes('Source 3:') ? 'text-cyan-400 font-semibold' :
                      log.includes('Generating skills') || log.includes('Parsing generated skills') ? 'text-amber-400 font-medium' :
                      log.includes('Writing') && log.includes('skills') ? 'text-green-400 font-medium' :
                      log.includes('Skills generation complete') || log.includes('complete') ? 'text-green-400 font-semibold' :
                      log.includes('Validated:') ? 'text-emerald-400' :
                      log.includes('Extracted') || log.includes('Loading') ? 'text-blue-300' :
                      log.startsWith('  ') ? 'text-gray-400' :
                      'text-blue-400'
                    }`}
                  >
                    {log}
                  </div>
                ))}
              </pre>
            </div>
            {isSkillsRunning && (
              <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>{skillsJobLogs.length} log entries | Polling every 2s</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {jobError && jobStatus === 'failed' && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{jobError}</AlertDescription>
        </Alert>
      )}

      {/* Results Card — gradient success banner (matching judge tuning pattern) */}
      {jobResult && jobResult.success && (
        <Card className="border-green-200 overflow-hidden">
          <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 border-l-4 border-green-500">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="h-6 w-6 text-green-600" />
              <span className="font-semibold text-green-800 text-lg">Optimization Successful</span>
              <Badge className="bg-green-100 text-green-700 border-green-300 ml-auto">
                <Zap className="h-3 w-3 mr-1" />
                Prompt Optimized
              </Badge>
            </div>
            <p className="text-sm text-green-700 font-medium">
              GEPA improved the prompt using {jobResult.metrics?.train_data_size || 0} traces
              over {jobResult.metrics?.num_iterations || numIterations} iterations.
              {jobResult.optimized_version && (
                <span className="ml-1">(Saved as version {jobResult.optimized_version})</span>
              )}
            </p>
            {jobResult.optimized_uri && (
              <p className="text-xs text-green-600 mt-1 font-mono">
                {jobResult.optimized_uri}
                {jobResult.optimized_prompt_name && (
                  <span className="text-green-500 ml-2">| alias: champion</span>
                )}
              </p>
            )}

            {/* Metrics row */}
            {jobResult.metrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                <div className="bg-white/80 rounded-lg p-2.5 border border-green-200">
                  <div className="text-xs text-gray-500">Training Data</div>
                  <div className="text-lg font-semibold text-green-800">{jobResult.metrics.train_data_size} traces</div>
                </div>
                <div className="bg-white/80 rounded-lg p-2.5 border border-green-200">
                  <div className="text-xs text-gray-500">Iterations</div>
                  <div className="text-lg font-semibold text-green-800">{jobResult.metrics.num_iterations}</div>
                </div>
                <div className="bg-white/80 rounded-lg p-2.5 border border-green-200">
                  <div className="text-xs text-gray-500">Original Length</div>
                  <div className="text-lg font-semibold text-green-800">{jobResult.metrics.original_length} chars</div>
                </div>
                <div className="bg-white/80 rounded-lg p-2.5 border border-green-200">
                  <div className="text-xs text-gray-500">Optimized Length</div>
                  <div className="text-lg font-semibold text-green-800">{jobResult.metrics.optimized_length} chars</div>
                </div>
              </div>
            )}

            {/* Expandable: Optimized Prompt (primary) */}
            {jobResult.optimized_prompt && (
              <div className="mt-3">
                <details open>
                  <summary className="cursor-pointer text-green-600 hover:text-green-800 text-sm font-medium">
                    View Optimized Prompt
                  </summary>
                  <div className="mt-2 relative">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 h-7 w-7 p-0 text-gray-400 hover:text-gray-700 z-10"
                      onClick={() => copyToClipboard(jobResult.optimized_prompt)}
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </Button>
                    <pre className="p-3 bg-white rounded-lg text-xs overflow-auto max-h-[300px] border border-green-200 whitespace-pre-wrap font-mono text-gray-700">
{jobResult.optimized_prompt}
                    </pre>
                  </div>
                </details>
              </div>
            )}

            {/* Expandable: Original Prompt (secondary, collapsed by default) */}
            {jobResult.original_prompt && (
              <div className="mt-2">
                <details>
                  <summary className="cursor-pointer text-green-600 hover:text-green-800 text-sm">
                    View Original Prompt
                  </summary>
                  <div className="mt-2 relative">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 h-7 w-7 p-0 text-gray-400 hover:text-gray-700 z-10"
                      onClick={() => copyToClipboard(jobResult.original_prompt)}
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </Button>
                    <pre className="p-3 bg-white rounded-lg text-xs overflow-auto max-h-[200px] border whitespace-pre-wrap font-mono text-gray-500">
{jobResult.original_prompt}
                    </pre>
                  </div>
                </details>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Generated Skills */}
      {generatedSkills.length > 0 && (
        <Card className="border-blue-200">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-blue-600" />
              <CardTitle className="text-lg">Generated Agent Skills</CardTitle>
              <Badge className="bg-blue-100 text-blue-700 border-blue-300 ml-auto">
                {generatedSkills.length} {generatedSkills.length === 1 ? 'skill' : 'skills'}
              </Badge>
            </div>
            <CardDescription>
              Skills synthesized from the optimized prompt, aligned judge guidelines, and evaluated traces.
              Each skill is a self-contained capability description that can be loaded on-demand by an agent.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {generatedSkills.map((skill) => (
              <div
                key={skill.id}
                className="border border-blue-100 rounded-lg overflow-hidden"
              >
                <button
                  onClick={() => setExpandedSkill(expandedSkill === skill.name ? null : skill.name)}
                  className="w-full text-left px-4 py-3 hover:bg-blue-50/50 transition-colors flex items-start gap-3"
                >
                  <FileText className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm text-gray-900">{skill.name}</span>
                      <span className="text-xs text-gray-400 font-mono">{skill.filename}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{skill.description}</p>
                  </div>
                  {expandedSkill === skill.name ? (
                    <ChevronUp className="h-4 w-4 text-gray-400 shrink-0 mt-0.5" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-gray-400 shrink-0 mt-0.5" />
                  )}
                </button>
                {expandedSkill === skill.name && (
                  <div className="border-t border-blue-100 bg-blue-50/30 px-4 py-3">
                    <div className="flex justify-end gap-1 mb-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-gray-500 hover:text-gray-700"
                        onClick={() => copyToClipboard(skill.content)}
                      >
                        <Copy className="h-3.5 w-3.5 mr-1" />
                        <span className="text-xs">Copy</span>
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-gray-500 hover:text-gray-700"
                        onClick={() => {
                          const blob = new Blob([skill.content], { type: 'text/markdown' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = skill.filename;
                          a.click();
                          URL.revokeObjectURL(url);
                          toast.success(`Downloaded ${skill.filename}`);
                        }}
                      >
                        <Download className="h-3.5 w-3.5 mr-1" />
                        <span className="text-xs">Download</span>
                      </Button>
                    </div>
                    <pre className="p-3 bg-white rounded-lg text-xs overflow-auto max-h-[400px] border border-blue-200 whitespace-pre-wrap font-mono text-gray-700">
{skill.content}
                    </pre>
                    {skill.generation_model && (
                      <p className="text-xs text-gray-400 mt-2">
                        Generated with {skill.generation_model}
                        {skill.created_at && <span> on {new Date(skill.created_at).toLocaleString()}</span>}
                      </p>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Download all skills */}
            <div className="pt-2 flex justify-end">
              <Button
                variant="outline"
                size="sm"
                className="text-blue-600 hover:text-blue-800 border-blue-200"
                onClick={() => {
                  generatedSkills.forEach((skill) => {
                    const blob = new Blob([skill.content], { type: 'text/markdown' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = skill.filename;
                    a.click();
                    URL.revokeObjectURL(url);
                  });
                  toast.success(`Downloaded ${generatedSkills.length} skill files`);
                }}
              >
                <Download className="h-3.5 w-3.5 mr-1" />
                Download All Skills
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* History */}
      <Card>
        <CardHeader>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="flex items-center justify-between w-full text-left"
          >
            <CardTitle className="text-lg flex items-center gap-2">
              <History className="h-5 w-5 text-gray-500" />
              Optimization History ({history.length})
            </CardTitle>
            {showHistory ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </CardHeader>
        {showHistory && (
          <CardContent>
            {history.length === 0 ? (
              <p className="text-sm text-gray-500 text-center py-4">No optimization runs yet</p>
            ) : (
              <div className="space-y-3">
                {history.map(run => (
                  <div
                    key={run.id}
                    className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                      selectedRun?.id === run.id ? 'border-violet-300 bg-violet-50/30' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setSelectedRun(selectedRun?.id === run.id ? null : run)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            run.status === 'completed' ? 'default' :
                            run.status === 'failed' ? 'destructive' :
                            'secondary'
                          }
                          className="text-xs"
                        >
                          {run.status}
                        </Badge>
                        <span className="text-sm font-mono text-gray-600">{run.prompt_uri}</span>
                      </div>
                      <span className="text-xs text-gray-400">
                        {run.created_at ? new Date(run.created_at).toLocaleString() : ''}
                      </span>
                    </div>
                    {run.optimizer_model && (
                      <div className="text-xs text-gray-400 mt-1">
                        Model: {run.optimizer_model} | Iterations: {run.num_iterations} | Candidates: {run.num_candidates}
                      </div>
                    )}

                    {/* Expanded view */}
                    {selectedRun?.id === run.id && (
                      <div className="mt-3 border-t pt-3 space-y-3">
                        {/* Metrics row */}
                        {run.metrics && (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-[10px] text-gray-500 uppercase">Training Data</div>
                              <div className="text-sm font-semibold">{run.metrics.train_data_size} traces</div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-[10px] text-gray-500 uppercase">Iterations</div>
                              <div className="text-sm font-semibold">{run.metrics.num_iterations}</div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-[10px] text-gray-500 uppercase">Original Length</div>
                              <div className="text-sm font-semibold">{run.metrics.original_length} chars</div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-[10px] text-gray-500 uppercase">Optimized Length</div>
                              <div className="text-sm font-semibold">{run.metrics.optimized_length} chars</div>
                            </div>
                          </div>
                        )}

                        {/* Optimized URI */}
                        {run.optimized_uri && (
                          <div className="flex items-center gap-2 text-xs">
                            <Link className="h-3 w-3 text-green-600" />
                            <span className="font-mono text-green-700">{run.optimized_uri}</span>
                            {run.optimized_version && (
                              <Badge variant="outline" className="text-[10px] text-green-600 border-green-300">
                                v{run.optimized_version} | alias: champion
                              </Badge>
                            )}
                          </div>
                        )}

                        {/* Optimized prompt (primary, open by default) */}
                        {run.optimized_prompt && (
                          <details open>
                            <summary className="cursor-pointer text-sm font-medium text-green-700 hover:text-green-900">
                              Optimized Prompt
                            </summary>
                            <div className="mt-1 relative">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="absolute top-1 right-1 h-6 w-6 p-0 text-gray-400 hover:text-gray-700 z-10"
                                onClick={(e) => { e.stopPropagation(); copyToClipboard(run.optimized_prompt!); }}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                              <pre className="text-xs bg-green-50 p-3 rounded border border-green-200 max-h-60 overflow-y-auto whitespace-pre-wrap font-mono text-gray-700">
{run.optimized_prompt}
                              </pre>
                            </div>
                          </details>
                        )}

                        {/* Original prompt (collapsed by default) */}
                        {run.original_prompt && (
                          <details>
                            <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                              Original Prompt
                            </summary>
                            <div className="mt-1 relative">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="absolute top-1 right-1 h-6 w-6 p-0 text-gray-400 hover:text-gray-700 z-10"
                                onClick={(e) => { e.stopPropagation(); copyToClipboard(run.original_prompt!); }}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                              <pre className="text-xs bg-gray-50 p-3 rounded border max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-gray-500">
{run.original_prompt}
                              </pre>
                            </div>
                          </details>
                        )}

                        {/* No prompt available message */}
                        {!run.optimized_prompt && run.status === 'completed' && (
                          <p className="text-xs text-amber-600">Optimized prompt not saved for this run.</p>
                        )}
                        {!run.optimized_prompt && run.status === 'running' && (
                          <p className="text-xs text-gray-500">Optimization still in progress...</p>
                        )}

                        {/* Error details */}
                        {run.error && (
                          <div className="bg-red-50 border border-red-200 rounded p-2 text-xs text-red-700">
                            {run.error}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  );
}
