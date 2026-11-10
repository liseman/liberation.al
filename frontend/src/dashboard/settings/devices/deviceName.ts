/** A device's name as people read it. Linux names a UVC camera "<card>: <product>", which for most webcams says the
 *  same thing twice ("Elgato Facecam: Elgato Facecam"); show it once. The server keeps the full name, which capture's
 *  manifest matches on. */
export function deviceName(name: string): string {
  const m = /^(.+): (.+)$/.exec(name.trim());
  const [, card, product] = m ?? [];
  return card !== undefined && card === product ? card : name;
}
