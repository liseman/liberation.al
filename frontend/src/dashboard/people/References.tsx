import { Button, FileInput, Group, NativeSelect, Progress, Stack, Text, TextInput } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { messageOf, request } from '../../shared/http';
import { SafeImage } from '../../shared/SafeImage';
import { Disclosure } from '../lib/Disclosure';

type FaceResult = 'added' | 'same' | 'none' | 'several';
type Reference = {
  id: string;
  kind: 'image' | 'audio' | 'video';
  source: string;
  person: string | null;
  enrollment?: string;
  face?: { result?: FaceResult };
};

/** What attaching a reference did, from the attach reply (#3254): a photo with exactly one face teaches it. */
export function attachedNote(item: Reference): string {
  if (item.kind !== 'image') return "Stored as a reference. Audio and video don't train recognition.";
  switch (item.face?.result) {
    case 'added':
      return 'Taught their face from this photo. Remove the reference to undo.';
    case 'same':
      return 'Stored. This face was already one of theirs.';
    case 'several':
      return 'Stored as a reference only: it shows several faces, so it trains no one.';
    case 'none':
      return 'Stored as a reference only: no face was found in it.';
    default:
      return 'Stored as a reference.';
  }
}
function Preview({ item }: { item: Reference }) {
  const url = `/people/reference-media/${item.id}`;
  if (item.kind === 'image')
    return <SafeImage src={url} alt="Selected reference preview" style={{ maxWidth: '100%', maxHeight: 240 }} />;
  if (item.kind === 'audio')
    return (
      <audio src={url} controls preload="metadata">
        <track kind="captions" />
      </audio>
    );
  return (
    <video src={url} controls preload="metadata" style={{ maxWidth: '100%', maxHeight: 240 }}>
      <track kind="captions" />
    </video>
  );
}
/** Local reference media. A photo with exactly one face also teaches the person's face; nothing else trains. */
export function References({ name }: { name: string }) {
  const [opened, setOpened] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [items, setItems] = useState<Reference[]>([]);
  const [preview, setPreview] = useState<Reference | null>(null);
  const [target, setTarget] = useState('');
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [cancellable, setCancellable] = useState(false);
  const [progress, setProgress] = useState(0);
  const epoch = useRef(0);
  const xhr = useRef<XMLHttpRequest | null>(null);
  const controller = useRef<AbortController | null>(null);
  const load = async () => {
    const r = await request<{ references: Reference[] }>(`/people/references?person=${encodeURIComponent(name)}`);
    setItems(r.references);
  };
  useEffect(
    () => () => {
      epoch.current++;
      xhr.current?.abort();
      controller.current?.abort();
    },
    [],
  );
  const cancel = () => {
    epoch.current++;
    xhr.current?.abort();
    controller.current?.abort();
    setBusy(false);
    setStatus('Cancelled. No reference attached.');
  };
  const upload = (file?: File) => {
    if (!file || busy) return;
    if (!file.size || file.size > 20 * 1024 * 1024) {
      setError('Choose a nonempty file up to 20 MB.');
      return;
    }
    setCancellable(true);
    const id = ++epoch.current;
    setError('');
    setBusy(true);
    setProgress(0);
    setStatus('Uploading locally…');
    setPreview(null);
    setTarget('');
    const x = new XMLHttpRequest();
    xhr.current = x;
    x.open('POST', '/people/reference-upload');
    x.timeout = 100000;
    x.setRequestHeader('Content-Type', 'application/octet-stream');
    x.upload.onprogress = e => {
      if (id === epoch.current && e.lengthComputable) {
        setProgress((e.loaded / e.total) * 100);
        if (e.loaded === e.total) setStatus('Validating and decoding locally…');
      }
    };
    x.onload = () => {
      if (id !== epoch.current) return;
      try {
        const data = JSON.parse(x.responseText);
        if (x.status !== 200 || data.error) throw new Error(data.error || 'Upload failed');
        setPreview(data);
        setStatus('Preview ready. Choose the target and confirm.');
      } catch (e) {
        setError(messageOf(e));
      } finally {
        setBusy(false);
      }
    };
    x.onerror = x.ontimeout = () => {
      if (id === epoch.current) {
        setBusy(false);
        setError('Upload interrupted. Select the file again to retry.');
      }
    };
    x.send(file);
  };
  const importUrl = async () => {
    setCancellable(true);
    const id = ++epoch.current;
    setBusy(true);
    setError('');
    setPreview(null);
    setTarget('');
    setStatus('Fetching and validating direct media…');
    controller.current = new AbortController();
    try {
      const item = await request<Reference>('/people/reference-url', { url }, controller.current.signal);
      if (id === epoch.current) {
        setPreview(item);
        setStatus('Preview ready. Choose the target and confirm.');
      }
    } catch (e) {
      if (id === epoch.current) setError(messageOf(e));
    } finally {
      if (id === epoch.current) setBusy(false);
    }
  };
  const attach = async () => {
    setCancellable(false);
    if (!preview || target !== name) return;
    setBusy(true);
    setError('');
    try {
      const attached = await request<Reference>('/people/reference-attach', { id: preview.id, person: target });
      setPreview(null);
      setStatus(attachedNote(attached));
      await load();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  };
  const remove = async (id: string) => {
    setCancellable(false);
    setBusy(true);
    setError('');
    try {
      await request('/people/reference-remove', { id });
      if (preview?.id === id) setPreview(null);
      await load();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section aria-label="Reference media">
      <Stack gap="sm">
        <Disclosure
          summary="Reference photos, audio, and video"
          opened={opened}
          keepMounted={false}
          onChange={next => {
            setOpened(next);
            if (next) void load().catch(e => setError(messageOf(e)));
          }}
        >
          <Stack gap="sm" p="sm">
            <Text size="sm">
              A photo with exactly one face also teaches recognition their face; removing it undoes that. Photos with
              several faces, audio, and video are stored as references only, and never clone a voice.
            </Text>
            <Text size="sm">
              Up to 20 MB; audio/video up to 120 seconds. JPEG/PNG, MP3/WAV, MP4/WebM and other supported media. Direct
              public HTTP(S) media links only; no web-page crawling or redirects.
            </Text>
            <Stack
              component="fieldset"
              gap="xs"
              p="sm"
              style={{
                margin: 0,
                border: '1px solid var(--mantine-color-default-border)',
                borderRadius: 'var(--mantine-radius-default)',
              }}
              aria-label="Drop reference media"
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                const selected = e.dataTransfer.files[0];
                if (selected) {
                  setFile(selected);
                  upload(selected);
                }
              }}
            >
              <FileInput
                label="Choose or drop a photo, audio, or video"
                placeholder="Choose a file"
                value={file}
                clearable
                accept="image/*,audio/*,video/*,.mp3"
                disabled={busy}
                fileInputProps={{ 'aria-label': 'Reference file' }}
                onChange={selected => {
                  setFile(selected);
                  if (selected) upload(selected);
                }}
              />
            </Stack>
            <Stack gap="xs">
              <TextInput
                label="Direct media URL"
                value={url}
                disabled={busy}
                onChange={e => setUrl(e.currentTarget.value)}
              />
              <Button
                variant="default"
                style={{ alignSelf: 'flex-start' }}
                disabled={busy || !url.trim()}
                onClick={() => void importUrl()}
              >
                Preview URL
              </Button>
            </Stack>
            {busy && (
              <>
                <Progress value={progress} aria-label="Upload progress" />
                {cancellable && (
                  <Button variant="subtle" onClick={cancel}>
                    Cancel operation
                  </Button>
                )}
              </>
            )}
            {status && <Text role="status">{status}</Text>}
            {error && (
              <Text role="alert" c="red">
                {error}
              </Text>
            )}
            {preview && (
              <div>
                <Preview item={preview} />
                <Text>{preview.source}</Text>
                <NativeSelect
                  label="Attach reference to"
                  value={target}
                  data={[
                    { value: '', label: 'Choose target person' },
                    { value: name, label: name },
                  ]}
                  onChange={e => setTarget(e.currentTarget.value)}
                />
                <Group>
                  <Button disabled={busy || target !== name} onClick={() => void attach()}>
                    Confirm reference for {name}
                  </Button>
                  <Button disabled={busy} variant="default" onClick={() => void remove(preview.id)}>
                    Discard preview
                  </Button>
                </Group>
              </div>
            )}
            {items.map(item => (
              <div key={item.id}>
                <Preview item={item} />
                <Text size="sm">
                  {item.source} ({item.enrollment === 'face' ? 'taught their face' : 'reference only'})
                </Text>
                <Button disabled={busy} variant="subtle" onClick={() => void remove(item.id)}>
                  Remove reference
                </Button>
              </div>
            ))}
          </Stack>
        </Disclosure>
      </Stack>
    </section>
  );
}
