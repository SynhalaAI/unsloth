// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { authFetch } from "@/features/auth";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import { useTrainingConfigStore } from "../stores/training-config-store";

interface ResumeCheckpointInfo {
  display_name: string;
  path: string;
  loss?: number | null;
}

interface ResumeCheckpointRun {
  name: string;
  checkpoints: ResumeCheckpointInfo[];
}

interface CheckpointListResponse {
  models?: ResumeCheckpointRun[];
}

/** Optional resume picker for the Train tab: run + checkpoint, mirroring the
 *  Export page's Fine-tuned tab (same /api/models/checkpoints source). The
 *  fresh-start config stays authoritative; this only fills
 *  `resumeCheckpointPath`, which the start payload carries as
 *  `resume_from_checkpoint`. Choosing a checkpoint arms the Resume
 *  Training switch, and only while that switch is on does the payload
 *  resume, so a chosen-but-unarmed checkpoint still trains from scratch.
 *  The choice is never persisted, so a stale absolute path cannot outlive
 *  the page. */
export function ResumeCheckpointPicker() {
  const t = useT();
  const resumeRun = useTrainingConfigStore((s) => s.resumeCheckpointRun);
  const resumePath = useTrainingConfigStore((s) => s.resumeCheckpointPath);
  const setResumeCheckpoint = useTrainingConfigStore(
    (s) => s.setResumeCheckpoint,
  );
  const resumeEnabled = useTrainingConfigStore(
    (s) => s.resumeTrainingEnabled,
  );
  const setResumeTrainingEnabled = useTrainingConfigStore(
    (s) => s.setResumeTrainingEnabled,
  );
  const [runs, setRuns] = useState<ResumeCheckpointRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    authFetch("/api/models/checkpoints")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return (await response.json()) as CheckpointListResponse;
      })
      .then((data) => {
        if (cancelled) return;
        setRuns(Array.isArray(data.models) ? data.models : []);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(
          cause instanceof Error && cause.message
            ? cause.message
            : t("studio.wizard.resumeLoadFailed"),
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const run = runs.find((candidate) => candidate.name === resumeRun) ?? null;
  const checkpoints = run?.checkpoints ?? [];
  const selected =
    checkpoints.find((candidate) => candidate.path === resumePath) ?? null;

  const handleRunChange = (name: string) => {
    const runCheckpoints =
      runs.find((candidate) => candidate.name === name)?.checkpoints ?? [];
    // Default to the newest saved checkpoint; the second select re-picks any other.
    const latest = runCheckpoints[runCheckpoints.length - 1] ?? null;
    setResumeCheckpoint(name, latest?.display_name ?? null, latest?.path ?? null);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-4 @md/train-section:grid-cols-2">
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium text-muted-foreground">
            {t("studio.wizard.resumeRunLabel")}
          </label>
          <Select
            value={resumeRun ?? ""}
            onValueChange={handleRunChange}
            disabled={loading || runs.length === 0}
          >
            <SelectTrigger className="w-full">
              <SelectValue
                placeholder={t(
                  loading
                    ? "studio.wizard.resumeSelectRun"
                    : runs.length === 0
                      ? "studio.wizard.resumeNoRuns"
                      : "studio.wizard.resumeSelectRun",
                )}
              />
            </SelectTrigger>
            <SelectContent>
              {runs.map((candidate) => (
                <SelectItem key={candidate.name} value={candidate.name}>
                  <span className="flex items-center gap-2">
                    {candidate.name}
                    <span className="text-muted-foreground text-xs">
                      {candidate.checkpoints.length} checkpoint
                      {candidate.checkpoints.length !== 1 ? "s" : ""}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium text-muted-foreground">
            {t("studio.wizard.resumeCheckpointLabel")}
          </label>
          <Select
            value={selected?.display_name ?? ""}
            onValueChange={(displayName) => {
              const next = checkpoints.find(
                (candidate) => candidate.display_name === displayName,
              );
              if (next && resumeRun) {
                setResumeCheckpoint(
                  resumeRun,
                  next.display_name,
                  next.path,
                );
              }
            }}
            disabled={!resumeRun || checkpoints.length === 0}
          >
            <SelectTrigger className="w-full">
              <SelectValue
                placeholder={t(
                  !resumeRun
                    ? "studio.wizard.resumeSelectRunFirst"
                    : checkpoints.length === 0
                      ? "studio.wizard.resumeNoCheckpoints"
                      : "studio.wizard.resumeSelectCheckpoint",
                )}
              />
            </SelectTrigger>
            <SelectContent>
              {checkpoints.map((candidate) => (
                <SelectItem key={candidate.path} value={candidate.display_name}>
                  <span className="flex items-center gap-2">
                    {candidate.display_name}
                    {candidate.loss != null && (
                      <span className="text-muted-foreground text-xs">
                        loss: {candidate.loss.toFixed(4)}
                      </span>
                    )}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}

      {selected && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2 rounded-lg bg-foreground/[0.04] px-3 py-2">
            <span
              className="min-w-0 truncate text-xs text-muted-foreground"
              title={selected.path}
            >
              {resumeRun} / {selected.display_name}
            </span>
            <button
              type="button"
              onClick={() => setResumeCheckpoint(null, null, null)}
              className="shrink-0 text-xs font-medium text-foreground transition-colors hover:underline"
            >
              {t("studio.wizard.resumeClear")}
            </button>
          </div>

          {/* The switch arms resume: the CTA only offers Resume Training
              and the payload only carries the path while this is on. */}
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border/60 px-3 py-2">
            <span className="flex min-w-0 flex-col">
              <span className="text-xs font-medium text-foreground">
                {t("studio.wizard.resumeEnableLabel")}
              </span>
              <span className="text-xs text-muted-foreground">
                {t("studio.wizard.resumeEnableHint")}
              </span>
            </span>
            <Switch
              aria-label={t("studio.wizard.resumeEnableLabel")}
              checked={resumeEnabled}
              onCheckedChange={setResumeTrainingEnabled}
            />
          </div>
        </div>
      )}
    </div>
  );
}
