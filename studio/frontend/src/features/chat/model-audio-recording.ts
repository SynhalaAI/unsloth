import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { getAudioSizeError, MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import {
  createAudioRecorder,
  type SegmentRecorder,
} from "@/features/chat/adapters/pcm-recorder";

function recordedAudioName(contentType: string): string {
  if (contentType === "audio/wav") return "recording.wav";
  if (contentType === "audio/ogg") return "recording.ogg";
  if (contentType === "audio/mp4") return "recording.m4a";
  return "recording.webm";
}

function stopMicrophone(stream: MediaStream | null) {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
}

export function useModelAudioRecording(
  onRecorded: (file: File) => Promise<void>,
) {
  const [isRecording, setIsRecording] = useState(false);
  const recorderRef = useRef<SegmentRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cancelledRef = useRef(false);
  const mountedRef = useRef(true);
  const limitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cleanup = useCallback(() => {
    if (limitTimerRef.current) clearTimeout(limitTimerRef.current);
    limitTimerRef.current = null;
    stopMicrophone(streamRef.current);
    streamRef.current = null;
    recorderRef.current = null;
  }, []);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    const recorder = recorderRef.current;
    if (recorder?.state === "recording") recorder.stop();
    cleanup();
    if (mountedRef.current) setIsRecording(false);
  }, [cleanup]);

  const start = useCallback(async () => {
    if (recorderRef.current || isRecording) return;
    cancelledRef.current = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (cancelledRef.current || !mountedRef.current) {
        stopMicrophone(stream);
        return;
      }
      streamRef.current = stream;
      const recorder = createAudioRecorder(stream);
      const chunks: Blob[] = [];
      recorderRef.current = recorder;
      recorder.addEventListener("dataavailable", (event) => {
        if (!cancelledRef.current && event.data.size > 0) {
          chunks.push(event.data);
          if (chunks.reduce((size, chunk) => size + chunk.size, 0) > MAX_AUDIO_SIZE) {
            const sizeError = getAudioSizeError(MAX_AUDIO_SIZE + 1);
            if (sizeError) toast.error(sizeError);
            cancel();
          }
        }
      });
      recorder.addEventListener("stop", () => {
        const wasCancelled = cancelledRef.current;
        const contentType = recorder.mimeType || "audio/wav";
        const clip = new File(chunks, recordedAudioName(contentType), {
          type: contentType,
        });
        cleanup();
        if (mountedRef.current) setIsRecording(false);
        if (wasCancelled) return;
        const sizeError = getAudioSizeError(clip.size);
        if (sizeError) {
          toast.error(sizeError);
          return;
        }
        if (clip.size === 0) {
          toast.error("No audio was recorded");
          return;
        }
        void onRecorded(clip).catch(() => {
          toast.error("Could not attach recorded audio");
        });
      }, { once: true });
      recorder.start(250);
      setIsRecording(true);
      const secondsWithin = (recorder as SegmentRecorder & {
        secondsWithin?: (maxBytes: number) => number;
      }).secondsWithin?.(MAX_AUDIO_SIZE);
      if (secondsWithin) {
        limitTimerRef.current = setTimeout(() => {
          if (recorderRef.current !== recorder) return;
          const sizeError = getAudioSizeError(MAX_AUDIO_SIZE + 1);
          if (sizeError) toast.error(sizeError);
          cancel();
        }, secondsWithin * 1000);
      }
    } catch (error) {
      cleanup();
      if (mountedRef.current) setIsRecording(false);
      if (!cancelledRef.current) {
        toast.error("Could not start audio recording", {
          description: error instanceof Error ? error.message : undefined,
        });
      }
    }
  }, [cancel, cleanup, isRecording, onRecorded]);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder?.state === "recording") recorder.stop();
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cancel();
    };
  }, [cancel]);

  return { isRecording, start, stop, cancel };
}