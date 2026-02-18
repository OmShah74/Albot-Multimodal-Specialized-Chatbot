'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Search, Square, CheckCircle2, Circle, Loader2,
  Globe, Brain, FileText, ChevronDown, ChevronUp,
  AlertCircle, Microscope, XCircle
} from 'lucide-react';

// ─────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────

interface ResearchProgressEvent {
  event_type: string;
  step_index?: number | null;
  total_steps?: number | null;
  sources_scraped: number;
  total_findings: number;
  current_activity: string;
  thinking?: string | null;
  data?: Record<string, any> | null;
  timestamp?: string;
}

interface ResearchStep {
  index: number;
  title: string;
  status: 'pending' | 'running' | 'completed' | 'error';
}

// ─────────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────────

interface DeepResearchPanelProps {
  sessionId: string;
  query: string;
  onComplete: (report: string, sources: string[], metrics?: any) => void;
  onCancel: () => void;
  apiBaseUrl: string;
}

// ─────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────

export default function DeepResearchPanel({
  sessionId,
  query,
  onComplete,
  onCancel,
  apiBaseUrl,
}: DeepResearchPanelProps) {
  const [status, setStatus] = useState<string>('planning');
  const [steps, setSteps] = useState<ResearchStep[]>([]);
  const [sourceCount, setSourceCount] = useState(0);
  const [findingsCount, setFindingsCount] = useState(0);
  const [currentActivity, setCurrentActivity] = useState('Starting research...');
  const [thinking, setThinking] = useState<string | null>(null);
  const [activityLog, setActivityLog] = useState<{ text: string; icon: string; time: string }[]>([]);
  const [progress, setProgress] = useState(0);
  const [showSteps, setShowSteps] = useState(true);
  const [isStopping, setIsStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activityRef = useRef<HTMLDivElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const hasCompletedRef = useRef(false);

  // Auto-scroll activity feed
  useEffect(() => {
    if (activityRef.current) {
      activityRef.current.scrollTop = activityRef.current.scrollHeight;
    }
  }, [activityLog]);

  // Poll for status updates
  useEffect(() => {
    let lastEventCount = 0;

    const poll = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/research/${sessionId}/status`);
        if (!res.ok) return;
        const data = await res.json();

        setStatus(data.status);
        setSourceCount(data.sources_scraped || 0);
        setFindingsCount(data.total_findings || 0);

        // Process new events
        const events: ResearchProgressEvent[] = data.progress_events || [];
        const newEvents = events.slice(lastEventCount);
        lastEventCount = events.length;

        for (const evt of newEvents) {
          processEvent(evt);
        }

        // Update plan steps
        if (data.plan?.steps) {
          const planSteps: ResearchStep[] = data.plan.steps.map((s: any) => ({
            index: s.step_index,
            title: s.title,
            status: s.status || 'pending',
          }));
          // Merge with current step state
          setSteps(prev => {
            const merged = planSteps.map((ps: ResearchStep) => {
              const existing = prev.find(p => p.index === ps.index);
              return existing ? { ...ps, status: existing.status } : ps;
            });
            return merged;
          });
        }

        // Check completion
        if (['complete', 'cancelled', 'error'].includes(data.status)) {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }

          if (hasCompletedRef.current) return;

          if (data.status === 'error') {
            setError('Research encountered an error.');
            hasCompletedRef.current = true; // Allow the error state to trigger once
          }

          // Fetch the final result
          try {
            const resultRes = await fetch(`${apiBaseUrl}/research/${sessionId}/result`);
            if (resultRes.ok) {
              const resultData = await resultRes.json();
              if (resultData.report && !hasCompletedRef.current) {
                hasCompletedRef.current = true;
                // Extract unique source URLs
                const sources = resultData.sources 
                  ? Array.from(new Set(resultData.sources.map((s: any) => s.url))).filter(Boolean) as string[]
                  : [];
                onComplete(resultData.report, sources, {
                  type: 'deep_research',
                  total_time_ms: resultData.research_time_ms,
                  total_sources: resultData.total_sources,
                  total_findings: resultData.total_findings,
                  session_id: sessionId
                });
              }
            }
          } catch {
            if (!hasCompletedRef.current) {
              hasCompletedRef.current = true;
              onComplete('Research completed but failed to fetch results.', [], { type: 'deep_research', error: true });
            }
          }
        }

      } catch (err) {
        console.error('Polling error:', err);
      }
    };

    // Start polling
    poll(); // Initial fetch
    pollingRef.current = setInterval(poll, 2000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [sessionId, apiBaseUrl]);

  const processEvent = (evt: ResearchProgressEvent) => {
    if (evt.current_activity) {
      setCurrentActivity(evt.current_activity);
    }
    if (evt.thinking) {
      setThinking(evt.thinking);
    }

    const time = evt.timestamp
      ? new Date(evt.timestamp).toLocaleTimeString()
      : new Date().toLocaleTimeString();

    switch (evt.event_type) {
      case 'plan_generated':
        if (evt.data?.steps) {
          setSteps(evt.data.steps.map((s: any) => ({
            index: s.index,
            title: s.title,
            status: 'pending' as const,
          })));
        }
        addActivity('📋 Research plan generated', time);
        break;

      case 'step_started':
        if (evt.step_index !== null && evt.step_index !== undefined) {
          setSteps(prev => prev.map(s =>
            s.index === evt.step_index
              ? { ...s, status: 'running' as const }
              : s.index < (evt.step_index || 0)
                ? { ...s, status: 'completed' as const }
                : s
          ));
          updateProgress(evt);
        }
        addActivity(`🔍 ${evt.current_activity}`, time);
        break;

      case 'search_completed':
        addActivity(`🔎 ${evt.current_activity}`, time);
        break;

      case 'source_scraped':
        setSourceCount(evt.sources_scraped);
        addActivity(`🌐 ${evt.current_activity}`, time);
        break;

      case 'analyzing_source':
        addActivity(`🧠 ${evt.current_activity}`, time);
        break;

      case 'findings_extracted':
        setFindingsCount(evt.total_findings);
        addActivity(`📝 ${evt.current_activity}`, time);
        break;

      case 'step_completed':
        if (evt.step_index !== null && evt.step_index !== undefined) {
          setSteps(prev => prev.map(s =>
            s.index === evt.step_index
              ? { ...s, status: 'completed' as const }
              : s
          ));
          updateProgress(evt);
        }
        addActivity(`✅ ${evt.current_activity}`, time);
        break;

      case 'synthesis_started':
        setStatus('synthesizing');
        addActivity('🧬 Synthesizing final research report...', time);
        break;

      case 'research_complete':
        setProgress(100);
        addActivity('✨ Research complete!', time);
        break;

      case 'error':
        setError(evt.current_activity);
        addActivity(`❌ ${evt.current_activity}`, time);
        break;

      default:
        if (evt.current_activity) {
          addActivity(`ℹ️ ${evt.current_activity}`, time);
        }
    }
  };

  const addActivity = (text: string, time: string) => {
    setActivityLog(prev => [...prev.slice(-50), { text, icon: '', time }]);
  };

  const updateProgress = (evt: ResearchProgressEvent) => {
    if (evt.step_index !== null && evt.step_index !== undefined && evt.total_steps) {
      const pct = Math.round(((evt.step_index + 1) / evt.total_steps) * 90);
      setProgress(pct);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);
    try {
      await fetch(`${apiBaseUrl}/research/${sessionId}/stop`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to stop research:', err);
    }
  };

  // ─────────────────────────────────────────────────
  // Status helpers
  // ─────────────────────────────────────────────────

  const getStatusLabel = () => {
    switch (status) {
      case 'planning': return 'Generating Research Plan';
      case 'researching': return 'Researching';
      case 'synthesizing': return 'Synthesizing Report';
      case 'complete': return 'Research Complete';
      case 'cancelled': return 'Research Stopped';
      case 'error': return 'Error';
      default: return 'Initializing';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'planning': return '#a78bfa';
      case 'researching': return '#60a5fa';
      case 'synthesizing': return '#f59e0b';
      case 'complete': return '#34d399';
      case 'cancelled': return '#fb923c';
      case 'error': return '#f87171';
      default: return '#94a3b8';
    }
  };

  const getStepIcon = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed': return <CheckCircle2 size={16} color="#34d399" />;
      case 'running': return <Loader2 size={16} color="#60a5fa" className="animate-spin" />;
      case 'error': return <XCircle size={16} color="#f87171" />;
      default: return <Circle size={16} color="#4b5563" />;
    }
  };

  const isActive = !['complete', 'cancelled', 'error'].includes(status);

  // ─────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95))',
      border: '1px solid rgba(148, 163, 184, 0.15)',
      borderRadius: '16px',
      padding: '24px',
      margin: '8px 0',
      backdropFilter: 'blur(12px)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <div style={{
          background: `linear-gradient(135deg, ${getStatusColor()}33, ${getStatusColor()}11)`,
          borderRadius: '12px',
          padding: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <Microscope size={24} color={getStatusColor()} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: '16px',
            fontWeight: 600,
            color: '#e2e8f0',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            Deep Research
            <span style={{
              fontSize: '11px',
              fontWeight: 500,
              background: `${getStatusColor()}22`,
              color: getStatusColor(),
              padding: '2px 10px',
              borderRadius: '12px',
              border: `1px solid ${getStatusColor()}44`,
            }}>
              {getStatusLabel()}
            </span>
          </div>
          <div style={{
            fontSize: '13px',
            color: '#94a3b8',
            marginTop: '2px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            maxWidth: '500px',
          }}>
            &quot;{query}&quot;
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{
        background: 'rgba(30, 41, 59, 0.8)',
        borderRadius: '8px',
        height: '6px',
        marginBottom: '16px',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          borderRadius: '8px',
          background: `linear-gradient(90deg, ${getStatusColor()}, ${getStatusColor()}cc)`,
          width: `${progress}%`,
          transition: 'width 0.5s ease',
          boxShadow: `0 0 8px ${getStatusColor()}66`,
        }} />
      </div>

      {/* Stats */}
      <div style={{
        display: 'flex',
        gap: '24px',
        marginBottom: '16px',
        padding: '12px 16px',
        background: 'rgba(30, 41, 59, 0.5)',
        borderRadius: '10px',
        border: '1px solid rgba(148, 163, 184, 0.08)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Globe size={16} color="#60a5fa" />
          <span style={{ fontSize: '13px', color: '#94a3b8' }}>Sources:</span>
          <span style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0' }}>{sourceCount}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={16} color="#34d399" />
          <span style={{ fontSize: '13px', color: '#94a3b8' }}>Findings:</span>
          <span style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0' }}>{findingsCount}</span>
        </div>
        {steps.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search size={16} color="#a78bfa" />
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>Steps:</span>
            <span style={{ fontSize: '15px', fontWeight: 600, color: '#e2e8f0' }}>
              {steps.filter(s => s.status === 'completed').length}/{steps.length}
            </span>
          </div>
        )}
      </div>

      {/* Research Steps */}
      {steps.length > 0 && (
        <div style={{
          marginBottom: '16px',
          background: 'rgba(30, 41, 59, 0.4)',
          borderRadius: '10px',
          border: '1px solid rgba(148, 163, 184, 0.08)',
          overflow: 'hidden',
        }}>
          <button
            onClick={() => setShowSteps(!showSteps)}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 16px',
              background: 'none',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            <span>Research Steps</span>
            {showSteps ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showSteps && (
            <div style={{ padding: '0 16px 12px' }}>
              {steps.map(step => (
                <div
                  key={step.index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '6px 0',
                    fontSize: '13px',
                    color: step.status === 'completed' ? '#34d399'
                      : step.status === 'running' ? '#60a5fa'
                      : '#64748b',
                  }}
                >
                  {getStepIcon(step.status)}
                  <span style={{
                    fontWeight: step.status === 'running' ? 600 : 400,
                  }}>
                    Step {step.index + 1}: {step.title}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Thinking */}
      {thinking && isActive && (
        <div style={{
          marginBottom: '16px',
          padding: '12px 16px',
          background: 'rgba(167, 139, 250, 0.08)',
          borderRadius: '10px',
          border: '1px solid rgba(167, 139, 250, 0.15)',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '6px',
          }}>
            <Brain size={14} color="#a78bfa" />
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#a78bfa' }}>Thinking</span>
          </div>
          <div style={{
            fontSize: '12px',
            color: '#c4b5fd',
            lineHeight: '1.5',
            fontStyle: 'italic',
          }}>
            {thinking.length > 200 ? thinking.slice(0, 200) + '...' : thinking}
          </div>
        </div>
      )}

      {/* Activity Feed */}
      <div
        ref={activityRef}
        style={{
          maxHeight: '180px',
          overflowY: 'auto',
          marginBottom: '16px',
          padding: '12px',
          background: 'rgba(15, 23, 42, 0.6)',
          borderRadius: '10px',
          border: '1px solid rgba(148, 163, 184, 0.08)',
        }}
      >
        {activityLog.length === 0 ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#64748b',
            fontSize: '13px',
          }}>
            <Loader2 size={14} className="animate-spin" />
            Initializing deep research...
          </div>
        ) : (
          activityLog.map((item, i) => (
            <div
              key={i}
              style={{
                padding: '4px 0',
                fontSize: '12px',
                color: '#94a3b8',
                lineHeight: '1.5',
                display: 'flex',
                justifyContent: 'space-between',
                gap: '12px',
              }}
            >
              <span style={{ flex: 1 }}>{item.text}</span>
              <span style={{ fontSize: '11px', color: '#475569', whiteSpace: 'nowrap' }}>{item.time}</span>
            </div>
          ))
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{
          marginBottom: '16px',
          padding: '12px 16px',
          background: 'rgba(248, 113, 113, 0.1)',
          borderRadius: '10px',
          border: '1px solid rgba(248, 113, 113, 0.2)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <AlertCircle size={16} color="#f87171" />
          <span style={{ fontSize: '13px', color: '#fca5a5' }}>{error}</span>
        </div>
      )}

      {/* Stop Button */}
      {isActive && (
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <button
            onClick={handleStop}
            disabled={isStopping}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 24px',
              background: isStopping
                ? 'rgba(148, 163, 184, 0.1)'
                : 'rgba(239, 68, 68, 0.12)',
              border: `1px solid ${isStopping ? 'rgba(148, 163, 184, 0.2)' : 'rgba(239, 68, 68, 0.3)'}`,
              borderRadius: '10px',
              color: isStopping ? '#94a3b8' : '#fca5a5',
              cursor: isStopping ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: 500,
              transition: 'all 0.2s',
            }}
          >
            <Square size={14} />
            {isStopping ? 'Stopping...' : 'Stop Research'}
          </button>
        </div>
      )}
    </div>
  );
}
