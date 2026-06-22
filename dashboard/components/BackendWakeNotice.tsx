"use client";

import { ArrowClockwise, CloudArrowUp, WarningCircle } from "@phosphor-icons/react";
import { getBackendHostLabel, getBackendWakeUrl, isHostedBackend } from "@/lib/backend-notice";

type BackendWakeNoticeProps = {
  /** Request still in flight — show “please wait” copy. */
  slow?: boolean;
  /** Request failed — show retry / wake-server guidance. */
  offline?: boolean;
  onRetry?: () => void;
  /** Smaller padding for hero mini-demo. */
  compact?: boolean;
};

export function BackendWakeNotice({ slow, offline, onRetry, compact = false }: BackendWakeNoticeProps) {
  const hosted = isHostedBackend();
  const wakeUrl = getBackendWakeUrl();
  const host = getBackendHostLabel();

  if (!slow && !offline) return null;

  if (offline) {
    return (
      <div className={`backend-notice backend-notice-offline${compact ? " is-compact" : ""}`}>
        <WarningCircle size={compact ? 20 : 28} weight="duotone" className="backend-notice-icon" />
        <div>
          <p className="backend-notice-title">
            {hosted ? "Inference API is waking up or unreachable" : "Inference API not reachable"}
          </p>
          <p className="backend-notice-body">
            {hosted ? (
              <>
                The live pipeline runs on our Render backend ({host}). Free-tier services sleep after
                inactivity — the first request can take <strong>30–60 seconds</strong>.
              </>
            ) : (
              <>
                The worked example needs the FastAPI service running locally. Start it on port 8000, then
                retry.
              </>
            )}
          </p>
          {hosted ? (
            <p className="backend-notice-body">
              <a href={wakeUrl} target="_blank" rel="noopener noreferrer" className="backend-notice-link">
                Open the API health check
              </a>{" "}
              in a new tab to wake the server, wait for a JSON response, then click Retry here.
            </p>
          ) : (
            <pre className="backend-notice-code mono">
              {`cd api\npython -m uvicorn main:app --port 8000`}
            </pre>
          )}
          {onRetry ? (
            <button type="button" className="btn btn-sm" style={{ marginTop: 10 }} onClick={onRetry}>
              <ArrowClockwise size={14} weight="bold" />
              Retry
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div className={`backend-notice backend-notice-slow${compact ? " is-compact" : ""}`}>
      <CloudArrowUp size={compact ? 18 : 22} weight="duotone" className="backend-notice-icon spin" />
      <div>
        <p className="backend-notice-title">Please wait — loading the pipeline</p>
        <p className="backend-notice-body">
          Running all seven layers through the inference API. This usually takes a few seconds
          {hosted ? " once the server is awake" : ""}.
        </p>
        {hosted ? (
          <p className="backend-notice-body">
            If this takes longer than ~30s, the Render backend may be cold-starting.{" "}
            <a href={wakeUrl} target="_blank" rel="noopener noreferrer" className="backend-notice-link">
              Open {host}/health
            </a>{" "}
            in another tab to wake it, then return here — your request will continue automatically.
          </p>
        ) : null}
      </div>
    </div>
  );
}
