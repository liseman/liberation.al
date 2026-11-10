/** Native datalist selection, distinct from typing, paste, and IME composition. */
export function datalistPicked(input: HTMLInputElement, event: Event): boolean {
  const { inputType, isComposing } = event as InputEvent;
  if (isComposing || (inputType && inputType !== 'insertReplacementText')) return false;
  // Firefox omits inputType for a suggestion pick. Require an exact option value in that fallback too, so unrelated
  // input events never commit arbitrary text; ordinary typing reports insertText and remains a draft.
  return [...(input.list?.options ?? [])].some(option => option.value === input.value);
}
