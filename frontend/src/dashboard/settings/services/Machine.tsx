// Settings' This machine (#386, drawn as cards in #787): what it is, a verdict on how it's doing, the models and system
// packages it's missing (with the commands that add them), its processor, memory, and all reported models, plus
// Liberational's own programs stacked over the last half hour (#2245). Polled while shown.
import { Button, Code, ColorSwatch, CopyButton, Progress } from '@mantine/core';
import { useQuery } from '@tanstack/react-query';
import { messageOf } from '../../../shared/http';
import { Disclosure } from '../../lib/Disclosure';
import { momentWords } from '../../lib/format';
import { PageError, PageLoading } from '../../lib/page';
import { getStats, STATS_KEY, type Stats } from '../../sources/api';
import { gb } from '../../sources/format';
import { SettingsRow, SettingsSection } from '../layout';
import { BADGE_POLL_MS, servicesQuery } from './api';
import classes from './machine.module.css';
import { BUSY_CPU, BUSY_SWAP, FULL_MEMORY, modelsWhere, platformLine, size, verdict } from './machineMeters';
import { TONE_COLOR, TONE_TINT } from './model';
import {
  deviceLabel,
  type InventoryGroup,
  type InventoryModel,
  inventoryMemory,
  type ModelInventory,
  modelInventory,
} from './modelInventory';

export const STATS_MS = 2000;

const clamp = (n: number) => Math.max(0, Math.min(100, n));

function Bar({ pct, color, label }: { pct: number; color: string; label: string }) {
  return (
    <Progress.Root role="img" aria-label={label} className={classes.bar} size={8} radius="xl" bg="var(--lib-track)">
      <Progress.Section value={clamp(pct)} color={color} withAria={false} />
    </Progress.Root>
  );
}

/** One thing to install: its name, what it is, and the command, with a button that copies it. */
function Install({ name, kind, command }: { name: string; kind: string; command?: string }) {
  return (
    <SettingsRow component="li" className={classes.install} aria-label={name}>
      <div>
        <div>
          <b>{name}</b> <span className={classes.kind}>{kind}</span>
        </div>
        {command && <Code className={classes.command}>{command}</Code>}
      </div>
      {command && (
        <CopyButton value={command}>
          {({ copied, copy }) => (
            <Button size="xs" variant="default" onClick={copy}>
              {copied ? 'Copied' : 'Copy command'}
            </Button>
          )}
        </CopyButton>
      )}
    </SettingsRow>
  );
}

function NeedsInstalling({ s }: { s: Stats }) {
  const models = s.ollama?.missing ?? [];
  const packages = s.system?.missing ?? [];
  const all = s.system?.commands ?? [];
  const together = packages.some(p => !p.command) && all.length > 0; // an older server: one command for them all
  if (!models.length && !packages.length && !s.system?.error) return null;
  return (
    <SettingsSection
      title="Needs installing"
      headingId="install-h"
      description={
        s.system?.error
          ? `Couldn't check the packages: ${s.system.error}`
          : "Some features won't work until these are on this machine. Run each command in a terminal here."
      }
    >
      <ul className={classes.list}>
        {models.map(m => (
          <Install key={m} name={m} kind="model" command={`ollama pull ${m}`} />
        ))}
        {packages.map(p => (
          <Install
            key={p.name}
            name={p.name}
            kind={`package${p.optional ? ', optional' : ''}. ${p.why}`}
            command={together ? undefined : p.command}
          />
        ))}
        {together && <Install name="The packages" kind="all at once" command={all.join(' && ')} />}
      </ul>
    </SettingsSection>
  );
}

function Tiles({ s, inventory }: { s: Stats; inventory: ModelInventory }) {
  const cpu = s.cpu ?? 0;
  const memPct = s.mem_total ? (100 * (s.mem_used ?? 0)) / s.mem_total : 0;
  const swap = s.swap_used ?? 0;
  const load = s.load ?? [];
  const models = s.ollama?.models ?? [];
  const total = models.reduce((n, m) => n + m.size, 0);
  const cpuTone = cpu >= BUSY_CPU ? 'var(--lib-warn-text)' : undefined;
  const memTone = memPct >= FULL_MEMORY ? 'var(--lib-bad)' : undefined;
  return (
    <div className={classes.tiles}>
      <SettingsSection className={classes.tile} aria-label="Processor">
        <h3 className={classes.tileTitle}>Processor</h3>
        <p className={classes.big} style={{ color: cpuTone }}>{`${Math.round(cpu)}%`}</p>
        <Bar
          pct={cpu}
          color={cpuTone ?? 'var(--mantine-primary-color-filled)'}
          label={`${Math.round(cpu)}% of the processor in use`}
        />
        <p className={classes.small}>
          {`${s.cpu_count ?? '?'} cores.`}
          {load.length >= 3 && ` Load ${load[0]?.toFixed(1)} now, ${load[2]?.toFixed(1)} over the last 15 minutes.`}
        </p>
      </SettingsSection>
      <SettingsSection className={classes.tile} aria-label="Memory">
        <h3 className={classes.tileTitle}>Memory</h3>
        <p className={classes.big} style={{ color: memTone }}>
          {gb(s.mem_used ?? 0)} <span className={classes.of}>{`of ${gb(s.mem_total ?? 0)}`}</span>
        </p>
        <Bar
          pct={memPct}
          color={memTone ?? 'var(--mantine-primary-color-filled)'}
          label={`${Math.round(memPct)}% of memory in use`}
        />
        <p className={classes.small}>
          {`${Math.round(memPct)}% in use.`}
          {swap > 0 && (
            <span style={swap > BUSY_SWAP ? { color: 'var(--lib-warn-text)', fontWeight: 600 } : undefined}>
              {` ${gb(swap)} in swap.`}
            </span>
          )}
        </p>
      </SettingsSection>
      <SettingsSection className={`${classes.tile} ${classes.modelsTile}`} aria-label="Models">
        <h3 className={classes.tileTitle}>Models</h3>
        <p className={classes.big}>
          {inventory.reported
            ? `${inventory.loaded} loaded`
            : s.ollama?.up === false
              ? 'Off'
              : `${models.length} loaded`}
        </p>
        {!inventory.reported && (
          <Progress.Root
            role="img"
            aria-label={`Model sizes: ${models.map(m => gb(m.size)).join(', ')}`}
            className={classes.bar}
            size={8}
            radius="xl"
            bg="var(--lib-track)"
          >
            {models.map((m, i) => (
              <Progress.Section
                key={m.name}
                value={clamp((100 * m.size) / (total || 1))}
                withAria={false}
                style={{ opacity: Math.max(0.35, 1 - i * 0.25) }}
              />
            ))}
          </Progress.Root>
        )}
        <p className={classes.small}>
          {inventory.reported ? inventoryMemory(inventory) || 'None loaded right now.' : modelsLine(s)}
        </p>
        {inventory.reported && inventory.configured.length > 0 && (
          <p className={classes.small}>{`${inventory.configured.length} configured, not currently reported loaded.`}</p>
        )}
      </SettingsSection>
    </div>
  );
}

/** The models in a sentence: their size, where they sit, and what's missing. */
function modelsLine(s: Stats): string {
  const o = s.ollama;
  if (o?.up === false) return o.message || 'Ollama is not running.';
  if (o?.error) return `Ollama: ${o.error}`;
  const models = o?.models ?? [];
  const missing = o?.missing ?? [];
  const lacks = missing.length ? `${missing.length} missing.` : 'Nothing missing.';
  if (!models.length) return `None loaded right now. ${lacks}`;
  return `${gb(models.reduce((n, m) => n + m.size, 0))}, ${modelsWhere(models)}. ${lacks}`;
}

function LoadedSince({ since }: { since: number | null }) {
  const date = since == null ? null : new Date(since * 1000);
  if (!date || !Number.isFinite(date.getTime())) return <span>Load time unavailable</span>;
  return (
    <span>
      {/* "Today, 13:22": the app's 24-hour words, not "Nov 6, 2026, 01:22 PM" (#2695) */}
      Loaded since <time dateTime={date.toISOString()}>{momentWords(since as number)}</time>
    </span>
  );
}

function ModelRow({ model }: { model: InventoryModel }) {
  const details = [
    model.params && `${model.params} parameters`,
    model.ctx && `context ${model.ctx.toLocaleString('en-US')}`,
  ]
    .filter(Boolean)
    .join(', ');
  return (
    <li className={classes.modelRow}>
      <div className={classes.modelName}>
        <b className={classes.mono}>{model.name}</b>
        <span className={classes.modelPurpose}>{model.purpose}</span>
        {details && <span className={classes.modelDetails}>{details}</span>}
      </div>
      <div className={classes.modelState}>
        <span>{deviceLabel(model.device)}</span>
        <LoadedSince since={model.since} />
        {model.allocation != null && <span>{`${gb(model.allocation)} model allocation`}</span>}
      </div>
    </li>
  );
}

/** A process's memory in a few words: "12.0 GB VRAM, 5.8 GB RSS", or Ollama's allocations. */
function groupMemory(group: InventoryGroup): string {
  if (group.ollama)
    return group.gpu == null ? 'GPU allocations unavailable' : `${size(group.gpu)} GPU model allocations`;
  // Unknown stays "unavailable", never a missing number that reads as zero
  return `${group.gpu == null ? 'VRAM unavailable' : `${size(group.gpu)} VRAM`}, ${
    group.rss == null ? 'RSS unavailable' : `${size(group.rss)} RSS`
  }`;
}

/** A process and its models, folded to one line (#2728, Joe's Settings board idea 5): "see-actions · 6 models · 12.0 GB
 *  VRAM, 5.8 GB RSS", opening to each model. The tiles above already answer "is it OK?". */
function ModelGroup({ group }: { group: InventoryGroup }) {
  const n = group.models.length;
  const memory = groupMemory(group);
  return (
    <section aria-label={`${group.name} models`} className={classes.modelGroup}>
      <Disclosure
        buttonClassName={classes.modelGroupToggle}
        summary={
          <span className={classes.modelGroupLine}>
            <b>{group.name}</b>
            <span>{n === 1 ? '1 model' : `${n} models`}</span>
            {memory && <span className={classes.dim}>{memory}</span>}
          </span>
        }
      >
        <p className={classes.modelProcessNote}>
          {group.pid == null ? 'Process not reported' : `PID ${group.pid}`}
          {group.ollama ? '. Process memory unavailable' : ''}
        </p>
        <ul className={classes.list} aria-label={`${group.name} loaded models`}>
          {group.models.map(model => (
            <ModelRow key={`${model.key}:${model.name}:${model.device}`} model={model} />
          ))}
        </ul>
      </Disclosure>
    </section>
  );
}

function Models({ s, inventory }: { s: Stats; inventory: ModelInventory }) {
  const up = s.ollama?.up !== false && !s.ollama?.error;
  const message =
    s.ollama?.up === false
      ? s.ollama.message || 'Ollama is not running.'
      : s.ollama?.error
        ? `Ollama: ${s.ollama.error}`
        : null;
  return (
    <SettingsSection
      title="Models in memory"
      headingId="models-h"
      description="Application models are grouped by their process. Process memory is shown once; Ollama reports model allocations separately."
      trailing={
        <span className={classes.ollama} style={{ color: up ? 'var(--lib-ok-text)' : 'var(--lib-bad)' }}>
          <span aria-hidden className={classes.smallDot} style={{ background: 'currentColor' }} />
          {up ? 'Ollama running' : 'Ollama not running'}
        </span>
      }
    >
      <p className={classes.modelSummary}>{`${inventory.loaded} loaded`}</p>
      {message && <p className={classes.inventoryNote}>{message}</p>}
      {!inventory.reported && (
        <p className={classes.inventoryNote}>Application model reports are unavailable on this server.</p>
      )}
      {inventory.errors.length > 0 && (
        <div role="status" className={classes.inventoryWarning}>
          <p>Some model reports are unavailable.</p>
          <ul>
            {inventory.errors.map(error => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {inventory.groups.map(group => (
        <ModelGroup key={group.key} group={group} />
      ))}
      {inventory.configured.length > 0 && (
        <div className={classes.configured}>
          <Disclosure
            buttonClassName={classes.modelGroupToggle}
            summary={
              <span
                className={classes.modelGroupLine}
              >{`${inventory.configured.length} configured, loaded when needed`}</span>
            }
          >
            <ul className={classes.list} aria-label="Configured models">
              {inventory.configured.map(model => (
                <li key={model.key} className={`${classes.modelRow} ${classes.unloaded}`}>
                  <div className={classes.modelName}>
                    <b className={classes.mono}>{model.name}</b>
                    <span className={classes.modelPurpose}>{model.purpose}</span>
                  </div>
                  <div className={classes.modelState}>
                    <span>{deviceLabel(model.device)}</span>
                    <span>{model.state === 'unknown' ? 'Status unavailable' : 'Loads when needed'}</span>
                  </div>
                </li>
              ))}
            </ul>
          </Disclosure>
        </div>
      )}
    </SettingsSection>
  );
}

export function Machine() {
  const stats = useQuery({ queryKey: STATS_KEY, queryFn: getStats, refetchInterval: STATS_MS });
  const services = useQuery(servicesQuery(BADGE_POLL_MS)); // for the platform: the same poll as the menu badge
  const s = stats.data;
  if (!s)
    return stats.error ? (
      <PageError
        title="Couldn't read this machine's stats"
        detail={messageOf(stats.error)}
        onRetry={() => void stats.refetch()}
        busy={stats.isFetching}
      />
    ) : (
      <PageLoading label="this machine's stats" lines={4} />
    );
  if (s.error)
    return (
      <PageError title="Couldn't read this machine's stats" detail={s.error}>
        It checks again on its own in a few seconds.
      </PageError>
    );
  const v = verdict(s);
  const inventory = modelInventory(s);
  return (
    <div className={classes.stack}>
      <p className={classes.platform}>{platformLine(services.data?.platform, s)}</p>
      <SettingsSection className={classes.verdict} aria-label="Health">
        <ColorSwatch
          component="span"
          aria-hidden
          className={classes.bigDot}
          color={TONE_COLOR[v.tone]}
          size={12}
          withShadow={false}
          style={{ boxShadow: `0 0 0 5px ${TONE_TINT[v.tone]}` }}
        />
        <div>
          <p className={classes.headline}>{v.title}</p>
          <p className={classes.small} style={{ fontSize: 13.5, marginTop: 3 }}>
            {v.text}
          </p>
        </div>
      </SettingsSection>
      <NeedsInstalling s={s} />
      <Tiles s={s} inventory={inventory} />
      <Models s={s} inventory={inventory} />
    </div>
  );
}
