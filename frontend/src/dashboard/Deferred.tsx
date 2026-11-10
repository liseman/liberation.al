// Keep deferred code failures local to the page, with a loading state and an explicit recovery action.
import { Center, Loader } from '@mantine/core';
import { Component, type ReactNode, Suspense } from 'react';
import { reportError } from '../shared/telemetry';
import { PageError, PageLoading } from './lib/page';
import { isChunkLoadError, reloadForNewBuild } from './lib/staleChunk';

class LoadBoundary extends Component<
  { name: string; children: ReactNode },
  { failed: boolean; reloading: boolean; chunk: boolean }
> {
  state = { failed: false, reloading: false, chunk: false };

  static getDerivedStateFromError(error: unknown) {
    return { failed: true, chunk: isChunkLoadError(error) };
  }

  componentDidCatch(error: unknown) {
    // A chunk a redeploy removed: reload once for the new build rather than show a failure. That reload is
    // the recovery, not an error, so only a failure the page shows goes into its log.
    if (reloadForNewBuild(error)) this.setState({ reloading: true });
    else reportError('boundary', error, this.props.name);
  }

  render() {
    if (this.state.reloading)
      return (
        <Center p="xl">
          <Loader aria-label={`Loading the new version of ${this.props.name}`} />
        </Center>
      );
    if (this.state.failed)
      return (
        <div style={{ padding: 'var(--mantine-spacing-md)' }}>
          <PageError title={`Couldn't load ${this.props.name}`} retryLabel="Reload" onRetry={() => location.reload()}>
            {this.state.chunk
              ? "This page's code didn't download. Reloading usually fixes it, and nothing was lost."
              : 'This page encountered an unexpected error. Try reloading; if it happens again, report the problem.'}
          </PageError>
        </div>
      );
    return this.props.children;
  }
}

/** A boundary per page/section, so switching tabs cannot hide the shell while a chunk downloads. */
export function Deferred({ name, children }: { name: string; children: ReactNode }) {
  return (
    <LoadBoundary name={name}>
      <Suspense
        fallback={
          <div style={{ padding: 'var(--mantine-spacing-md)' }}>
            <PageLoading label={name} />
          </div>
        }
      >
        {children}
      </Suspense>
    </LoadBoundary>
  );
}
