// Settings' menu: the sections in order, each with its title, and optionally its own panel and a badge. A
// setting belongs to a section by labels.toml ("section" on each row); every setting also shows in Advanced. A
// section can draw a panel of its own (Body) above its rows: Devices (S3), Services, and This machine
// (S5), and Voice clones plug in here, and read and change the draft through useDraft (draft.tsx). Cameras and
// microphones also holds the network camera sources, and Imported media the photo libraries and recorders, both
// moved from the Ingest tab.
import { type ComponentType, lazy } from 'react';
import { DEVICE_CARD } from './devices/cards';
import { useDeviceIssues } from './devices/issues';
import { useServicesBadge } from './services/api';
import { CARD_ID } from './sources/model';

const Devices = lazy(() => import('./devices/Devices').then(m => ({ default: m.Devices })));
const Machine = lazy(() => import('./services/Machine').then(m => ({ default: m.Machine })));
const Services = lazy(() => import('./services/ServicesPanel').then(m => ({ default: m.Services })));
const Clones = lazy(() => import('../people/clones/Clones').then(m => ({ default: m.Clones })));
// What reaches outside this machine: Phone's status line and People and privacy's list
const PhoneStatus = lazy(() => import('./outside').then(m => ({ default: m.PhoneStatus })));
const WhatLeaves = lazy(() => import('./outside').then(m => ({ default: m.WhatLeaves })));
const ImportedMedia = lazy(() => import('./sources/Sources').then(m => ({ default: m.ImportedMedia })));

export type SectionId =
  | 'you'
  | 'devices'
  | 'media'
  | 'features'
  | 'people'
  | 'voices'
  | 'phone'
  | 'services'
  | 'machine'
  | 'advanced';

export interface SectionBadge {
  text: string; // "1 issue", "Off"
  tone: 'bad' | 'warn' | 'dim'; // bad: a red pill (a service that failed)
}

export interface SectionDef {
  id: SectionId;
  title: string;
  blurb?: string; // one line under the section's heading
  Body?: ComponentType; // a panel of the section's own, above its rows
  drawn?: readonly string[]; // settings the Body draws itself: not drawn again as rows (search and Advanced list them)
  hidden?: readonly string[]; // settings this section leaves out: a link to one opens Advanced, which has its field
  useBadge?: () => SectionBadge | null; // a hook: the menu badge, from the section's own data
  anchors?: readonly string[]; // ids of cards in its Body: a link to one (#<id>) opens this section and scrolls to it
}

export const SECTIONS: SectionDef[] = [
  { id: 'you', title: 'You and the assistant' },
  {
    id: 'devices',
    title: 'Devices',
    blurb: 'Cameras and microphones. Phones, watches, and tags are in Known › Devices.',
    Body: Devices,
    useBadge: useDeviceIssues,
    drawn: ['vision.camera_names', 'audio.sensitivity', 'audio.auto_gain_target_dbfs'],
    anchors: [CARD_ID.camera, DEVICE_CARD.cameras, DEVICE_CARD.microphones, DEVICE_CARD.bluetooth],
  },
  {
    id: 'media',
    title: 'Imported media',
    blurb: 'Photo libraries and recorders whose photos, video, and audio Liberational brings in.',
    Body: ImportedMedia,
    anchors: [CARD_ID.media],
  },
  { id: 'features', title: 'Features', blurb: 'What the assistant does, each on or off.' },
  {
    id: 'people',
    title: 'People and privacy',
    blurb: 'What leaves this machine, and how other devices reach it.',
    Body: WhatLeaves,
  },
  {
    // Every voice heard, named or not, and a new one recorded: a person's own clone is on their card.
    id: 'voices',
    title: 'Voice clones',
    blurb:
      "A voice can be cloned once Liberational has about 12 seconds of clean speech from it. A person's own is on their card under People.",
    Body: Clones,
  },
  {
    id: 'phone',
    title: 'Phone',
    Body: PhoneStatus,
    blurb: 'Calls go through Twilio. Its account keys stay on this machine.',
  },
  {
    id: 'services',
    title: 'Services',
    blurb: 'The programs that record, recognize, and answer. They run in the background on this machine.',
    Body: Services,
    useBadge: useServicesBadge,
    // Which services run here is the switches and Enable and Disable now; the two lists stay in Advanced and search.
    hidden: ['services.enable', 'services.disable'],
  },
  { id: 'machine', title: 'This machine', Body: Machine },
  {
    id: 'advanced',
    title: 'Advanced',
    blurb: 'Every setting in the config, for fine-tuning. Most never need changing.',
  },
];

export const sectionOf = (id: string | null | undefined): SectionDef =>
  SECTIONS.find(s => s.id === id) ?? (SECTIONS[0] as SectionDef);

/** The section a link opens: "#<section id>", "#<anchor>" for a card in a section's panel, or "#set-<key>" for a
 *  setting, which opens the section showing it. */
export function sectionInHash(hash: string, sectionOfKey: (key: string) => string): SectionId | null {
  let id: string;
  try {
    id = decodeURIComponent(hash.replace(/^#/, ''));
  } catch {
    return null;
  }
  if (id.startsWith('set-')) return homeOf(id.slice(4), sectionOfKey);
  return SECTIONS.find(s => s.id === id || s.anchors?.includes(id))?.id ?? null;
}

/** The section that shows a setting's field: its own, or Advanced for one its section leaves out (or none). */
export function homeOf(key: string, sectionOfKey: (key: string) => string): SectionId {
  const section = sectionOf(sectionOfKey(key) || 'advanced');
  return section.hidden?.includes(key) ? 'advanced' : section.id;
}
