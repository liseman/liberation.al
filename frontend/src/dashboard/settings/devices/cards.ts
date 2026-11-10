/** The Devices section's cards a link can go straight to (/settings#device-links). catalog.ts lists them as the
 *  section's anchors, so a link opens Devices and scrolls to the card. A module of its own, so the catalog can name
 *  them without loading the Devices panel. */
export const DEVICE_CARD = {
  cameras: 'device-cameras',
  microphones: 'device-microphones',
  bluetooth: 'settings-group-bluetooth', // the Bluetooth card's heading (SettingsPage's group heading id, from labels.toml)
  links: 'device-links', // now Known › Devices: Settings redirects a link to it there
} as const;
