import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';

/**
 * Tracks background processing jobs for the whole app.
 *
 * Lives above the views rather than inside Library so the reasoning rail keeps
 * showing progress after you switch to Search — the work continues on the
 * server either way, and hiding it behind a tab would make a long upload look
 * like it stalled.
 *
 * Polling runs only while something is unfinished, so an idle session makes no
 * requests at all.
 */
export function useJobs(onJobFinished) {
  const [jobs, setJobs] = useState([]);
  // Held in a ref so the poll effect doesn't tear down and re-subscribe every
  // time the parent re-creates the callback. Written in an effect rather than
  // during render — a render can be discarded, and a ref written by a
  // discarded render would leave a stale callback behind.
  const finishedRef = useRef(onJobFinished);
  useEffect(() => { finishedRef.current = onJobFinished; }, [onJobFinished]);

  const track = useCallback(async (jobId) => {
    try {
      const job = await api.job(jobId);
      setJobs((current) => [job, ...current.filter((j) => j.id !== job.id)]);
    } catch {
      // A job we cannot read yet will be picked up by the next poll.
    }
  }, []);

  // Adopt whatever the server is already working on. Processing continues on
  // the server across a page reload, so without this the rail would go blank
  // while a long upload was still running — making finished work look lost.
  useEffect(() => {
    let cancelled = false;
    api.jobs()
      .then((existing) => { if (!cancelled && existing?.length) setJobs(existing); })
      .catch(() => { /* an unreachable API is already surfaced elsewhere */ });
    return () => { cancelled = true; };
  }, []);

  const pending = jobs.some((j) => j.status === 'queued' || j.status === 'running');

  useEffect(() => {
    if (!pending) return undefined;

    const timer = setInterval(async () => {
      const refreshed = await Promise.all(
        jobs.map(async (job) => {
          if (job.status === 'done' || job.status === 'failed') return job;
          try {
            return await api.job(job.id);
          } catch (e) {
            if (e.message?.includes('404') || e.message?.toLowerCase().includes('not found')) {
              return { ...job, status: 'failed', error: 'Job expired or server restarted' };
            }
            return job;
          }
        }),
      );

      const justFinished = refreshed.some(
        (job, i) =>
          jobs[i].status !== job.status && (job.status === 'done' || job.status === 'failed'),
      );
      setJobs(refreshed);
      if (justFinished) finishedRef.current?.();
    }, 1200);

    return () => clearInterval(timer);
  }, [jobs, pending]);

  return { jobs, track, pending };
}
