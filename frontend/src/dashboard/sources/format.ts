// Byte counts as the old dashboard showed them (static/lib/ui.js).
export const mb = (b: number): string => `${(b / 1048576).toFixed(0)} MB`;
export const gb = (b: number): string => `${(b / 1073741824).toFixed(1)} GB`;
