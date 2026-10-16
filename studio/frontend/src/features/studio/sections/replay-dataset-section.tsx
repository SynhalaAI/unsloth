// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import { DatasetSelector, datasetDisplayName } from "@/features/dataset-picker";
import { hfApiToken, useHfTokenStore, useOnlineStatus } from "@/features/hub";
import {
  HfDatasetSubsetSplitSelectors,
  replayDatasetSupported,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { FileAttachmentIcon, ViewIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { DatasetPreviewDialog } from "./dataset-preview-dialog";
import { ParamsRow, ParamsSliderRow } from "./params-section-controls";

/** Optional replay dataset: rows from a second dataset (Hugging Face Hub or a local device
 * file/folder) interleaved into the main training data to slow catastrophic forgetting across
 * staged fine-tunes. The picker is the same one the main dataset uses, writing to the
 * replay fields instead. */
const noop = (): void => undefined;

export function ReplayDatasetSection(): ReactElement | null {
  const t = useT();
  const deviceType = usePlatformStore((state) => state.deviceType);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      supported: replayDatasetSupported(state),
      enabled: state.replayDatasetEnabled,
      dataset: state.replayDataset,
      source: state.replayDatasetSource,
      subset: state.replayDatasetSubset,
      split: state.replayDatasetSplit,
      probability: state.replayDatasetProbability,
      setEnabled: state.setReplayDatasetEnabled,
      setSubset: state.setReplayDatasetSubset,
      setSplit: state.setReplayDatasetSplit,
      setProbability: state.setReplayDatasetProbability,
      isVisionModel: state.isVisionModel,
    })),
  );
  const [open, setOpen] = useState(false);
  const [replayPreviewOpen, setReplayPreviewOpen] = useState(false);
  const hfToken = useHfTokenStore((state) => state.token);
  const online = useOnlineStatus();

  // The MLX worker has no mixing path, and other run types are refused by the backend.
  if (!store.supported || deviceType === "mac") {
    return null;
  }

  const repoReady = (store.dataset ?? "").trim().length > 0;
  const isHubSource = store.source === "huggingface";

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.dataset.replayLabel")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <div className="flex flex-col gap-4 pt-1.5">
          <ParamsRow
            label={t("studio.dataset.replayEnable")}
            tooltip={t("studio.dataset.replayTooltip")}
          >
            <Switch
              id="replayDatasetEnabled"
              aria-label={t("studio.dataset.replayEnable")}
              checked={store.enabled && repoReady}
              disabled={!repoReady}
              onCheckedChange={(value) => store.setEnabled(!!value)}
            />
          </ParamsRow>
          <div className="flex flex-col gap-2">
            <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
              {t("studio.dataset.replayDatasetId")}
            </span>
            <DatasetSelector
              target="replay"
              label={t("studio.dataset.replayDatasetId")}
            />
          </div>
          {repoReady && (
            <div className="flex flex-col items-stretch gap-3 rounded-lg border bg-muted/40 px-3.5 py-3 @md/train-section:flex-row @md/train-section:items-center">
              <div className="flex min-w-0 items-center gap-3">
                <div className="shrink-0 rounded-md bg-indigo-500/10 p-1.5">
                  <HugeiconsIcon
                    icon={FileAttachmentIcon}
                    className="size-4 text-indigo-500"
                  />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-mono text-sm font-medium">
                    {isHubSource
                      ? store.dataset
                      : datasetDisplayName(store.dataset ?? "")}
                  </p>
                  <p className="text-ui-10 text-muted-foreground">
                    {isHubSource ? (
                      <>
                        {t("studio.dataset.huggingFaceDataset")}
                        {store.subset && ` / ${store.subset}`}
                        {store.split && ` / ${store.split}`}
                      </>
                    ) : (
                      <>{t("studio.dataset.localDataset")}</>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-end gap-1 @md/train-section:ml-auto @md/train-section:flex-nowrap">
                <Button
                  variant="ghost"
                  size="sm"
                  className="shrink-0 text-xs"
                  onClick={() => setReplayPreviewOpen(true)}
                >
                  <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
                  {t("studio.dataset.viewDataset")}
                </Button>
              </div>
            </div>
          )}
          {isHubSource && repoReady && (
            <HfDatasetSubsetSplitSelectors
              variant="studio"
              enabled={true}
              datasetName={store.dataset}
              accessToken={hfApiToken(hfToken)}
              online={online}
              datasetSubset={store.subset}
              setDatasetSubset={store.setSubset}
              datasetSplit={store.split}
              setDatasetSplit={store.setSplit}
              showEvalSplit={false}
              datasetStreaming={false}
              setManualDatasetOptionsValid={noop}
              markManualDatasetOptionsEdited={noop}
            />
          )}
          <ParamsSliderRow
            label={t("studio.dataset.replayProbability")}
            tooltip={t("studio.dataset.replayTooltip")}
            value={store.probability}
            onChange={(next) => {
              if (Number.isFinite(next)) {
                store.setProbability(Math.min(95, Math.max(1, Math.round(next))));
              }
            }}
            min={1}
            max={95}
            step={1}
          />
          <span className="text-xs text-muted-foreground">{t("studio.dataset.replayHint")}</span>
          {replayPreviewOpen && (
            <DatasetPreviewDialog
              open={true}
              onOpenChange={setReplayPreviewOpen}
              datasetName={store.dataset}
              datasetSource={store.source}
              hfToken={hfToken.trim() || null}
              datasetSubset={store.subset}
              datasetSplit={store.split}
              isVlm={store.isVisionModel}
            />
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
