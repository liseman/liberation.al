// Client-only Review search. Fold words once, then compare each query word against a shared vocabulary trie,
// not every transcript line. Inverted postings intersect the matching documents before ranking/highlighting.

export interface SearchDocument {
  id: string;
  kind: 'transcript' | 'summary';
  ts: number;
  text: string;
}

export interface SearchHit {
  document: SearchDocument;
  /** Lower is better within one query: edits first, then prefix matches, then the document's timestamp. */
  score: number;
  /** Whole matched words in the original text, as half-open UTF-16 offsets suitable for String.slice. */
  ranges: [number, number][];
}

interface TrieNode {
  children: Map<string, TrieNode>;
  word?: number;
}

interface Word {
  text: string;
  documents: number[];
}

interface Occurrence {
  word: number;
  range: [number, number];
}

export interface SearchIndex {
  readonly documents: readonly SearchDocument[];
  readonly occurrences: readonly Occurrence[][];
  readonly words: readonly Word[];
  readonly trie: TrieNode;
  readonly longestWord: number;
}

/** Accent-insensitive Unicode folding, including expanded ligatures and the common case-fold expansions. */
export function foldSearchText(text: string): string {
  return text.normalize('NFKD').toLowerCase().replace(/\p{M}/gu, '').replace(/ß/g, 'ss').replace(/ς/g, 'σ');
}

const wordsIn = (text: string) => text.matchAll(/[\p{L}\p{N}\p{M}]+/gu);
const node = (): TrieNode => ({ children: new Map() });

/** Build a snapshot: future edits to the caller's documents require a new index. No input object is mutated. */
export function createSearchIndex(documents: readonly SearchDocument[]): SearchIndex {
  const copied = documents.map(document => ({ ...document }));
  const words: Word[] = [];
  const vocabulary = new Map<string, number>();
  const trie = node();
  let longestWord = 0;
  const occurrences = copied.map((document, documentIndex) => {
    const seen = new Set<number>();
    const found: Occurrence[] = [];
    for (const match of wordsIn(document.text)) {
      const text = foldSearchText(match[0]);
      if (!text) continue;
      let id = vocabulary.get(text);
      if (id === undefined) {
        id = words.length;
        words.push({ text, documents: [] });
        vocabulary.set(text, id);
        let at = trie;
        let length = 0;
        for (const character of text) {
          let child = at.children.get(character);
          if (!child) {
            child = node();
            at.children.set(character, child);
          }
          at = child;
          length++;
        }
        at.word = id;
        longestWord = Math.max(longestWord, length);
      }
      if (!seen.has(id)) {
        words[id]?.documents.push(documentIndex);
        seen.add(id);
      }
      found.push({ word: id, range: [match.index, match.index + match[0].length] });
    }
    return found;
  });
  return { documents: copied, occurrences, words, trie, longestWord };
}

/** The typo allowance depends on query code points, not UTF-16 code units. */
export const typoBudget = (length: number): number => (length <= 3 ? 0 : length <= 7 ? 1 : 2);

interface BandRow {
  offset: number;
  values: number[];
  minimum: number;
}

const atColumn = (row: BandRow | undefined, column: number, beyond: number): number =>
  row?.values[column - row.offset] ?? beyond;

function firstRow(length: number, budget: number): BandRow {
  return { offset: 0, values: Array.from({ length: Math.min(length, budget) + 1 }, (_, i) => i), minimum: 0 };
}

function nextRow(
  query: readonly string[],
  character: string,
  previousCharacter: string | undefined,
  previous: BandRow,
  beforePrevious: BandRow | undefined,
  depth: number,
  budget: number,
): BandRow {
  const beyond = budget + 1;
  const first = Math.max(0, depth - budget);
  const last = Math.min(query.length, depth + budget);
  const current: BandRow = { offset: first, values: [], minimum: beyond };
  // Only the diagonal band exists: allocation, recurrence, and minimum tracking all cost O(budget), even for a
  // single unusually long transcript token. Missing columns cannot be part of a path within the edit allowance.
  for (let j = first; j <= last; j++) {
    let cost =
      j === 0
        ? depth
        : Math.min(
            atColumn(previous, j, beyond) + 1,
            atColumn(current, j - 1, beyond) + 1,
            atColumn(previous, j - 1, beyond) + (character === query[j - 1] ? 0 : 1),
          );
    if (j > 1 && character === query[j - 2] && previousCharacter === query[j - 1])
      cost = Math.min(cost, atColumn(beforePrevious, j - 2, beyond) + 1);
    cost = Math.min(cost, beyond);
    current.values.push(cost);
    current.minimum = Math.min(current.minimum, cost);
  }
  return current;
}

/** Bounded optimal-string-alignment distance: adjacent swaps cost one; overlapping swaps are not reused. */
export function boundedDistance(left: string, right: string, budget: number): number {
  const query = Array.from(left);
  const word = Array.from(right);
  const beyond = budget + 1;
  if (Math.abs(query.length - word.length) > budget) return beyond;
  let previous = firstRow(query.length, budget);
  let beforePrevious: BandRow | undefined;
  let previousCharacter: string | undefined;
  for (let i = 0; i < word.length; i++) {
    const character = word[i];
    if (character === undefined) break;
    const current = nextRow(query, character, previousCharacter, previous, beforePrevious, i + 1, budget);
    if (current.minimum > budget) return beyond;
    beforePrevious = previous;
    previous = current;
    previousCharacter = character;
  }
  return Math.min(atColumn(previous, query.length, beyond), beyond);
}

interface Walk {
  node: TrieNode;
  character: string;
  previousCharacter?: string;
  previous: BandRow;
  beforePrevious?: BandRow;
  depth: number;
  prefixCost: number;
}

function matchingWords(index: SearchIndex, query: string[], prefix: boolean, weight: number): Map<number, number> {
  const budget = typoBudget(query.length);
  const beyond = budget + 1;
  const found = new Map<number, number>();
  if (query.length > index.longestWord + budget) return found;
  const initial = firstRow(query.length, budget);
  const stack: Walk[] = [];
  for (const [character, child] of index.trie.children)
    stack.push({ node: child, character, previous: initial, depth: 1, prefixCost: beyond });
  while (stack.length) {
    const state = stack.pop();
    if (!state) break;
    const row = nextRow(
      query,
      state.character,
      state.previousCharacter,
      state.previous,
      state.beforePrevious,
      state.depth,
      budget,
    );
    const fullCost = atColumn(row, query.length, beyond);
    if (state.node.word !== undefined) {
      const full = fullCost * weight;
      const partial = prefix ? state.prefixCost * weight + 1 : beyond * weight;
      const best = Math.min(full, partial);
      if (best < beyond * weight) found.set(state.node.word, best);
    }
    const prefixCost = prefix ? Math.min(state.prefixCost, fullCost) : beyond;
    if (prefix && prefixCost <= budget && state.depth >= query.length + prefixCost) {
      // No longer prefix can improve this distance. Enumerate its terminals without more DP rows.
      const descendants = [...state.node.children.values()];
      while (descendants.length) {
        const child = descendants.pop();
        if (!child) break;
        if (child.word !== undefined) found.set(child.word, prefixCost * weight + 1);
        descendants.push(...child.children.values());
      }
      continue;
    }
    if (row.minimum > budget && prefixCost > budget) continue;
    for (const [character, child] of state.node.children)
      stack.push({
        node: child,
        character,
        previousCharacter: state.character,
        previous: row,
        beforePrevious: state.previous,
        depth: state.depth + 1,
        prefixCost,
      });
  }
  return found;
}

const compareText = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

/** Every query word must match. The last word may match a token prefix while it is being typed. */
export function searchIndex(index: SearchIndex, query: string, limit = 100): SearchHit[] {
  const terms = [...wordsIn(query)].map(match => Array.from(foldSearchText(match[0]))).filter(term => term.length);
  if (!terms.length || limit <= 0 || !Number.isFinite(limit)) return [];
  const grouped = new Map<string, { words: Map<number, number>; count: number }>();
  for (let i = 0; i < terms.length; i++) {
    const term = terms[i];
    if (!term) continue;
    const prefix = i === terms.length - 1;
    const key = `${term.join('')}\0${prefix}`;
    const previous = grouped.get(key);
    if (previous) previous.count++;
    else {
      const words = matchingWords(index, term, prefix, terms.length + 1);
      if (!words.size) return [];
      grouped.set(key, { words, count: 1 });
    }
  }
  const matches = [...grouped.values()];
  let candidates: number[] | undefined;
  let scores: Float64Array | undefined;
  for (const { words, count } of matches) {
    const costs = new Float64Array(index.documents.length).fill(Number.POSITIVE_INFINITY);
    const documents: number[] = [];
    for (const [word, unitCost] of words) {
      const score = unitCost * count;
      for (const document of index.words[word]?.documents ?? []) {
        const previous = costs[document] ?? Number.POSITIVE_INFINITY;
        if (previous === Number.POSITIVE_INFINITY) documents.push(document);
        if (score < previous) costs[document] = score;
      }
    }
    if (candidates === undefined) {
      candidates = documents;
      scores = costs;
    } else {
      let kept = 0;
      for (const document of candidates) {
        const cost = costs[document] ?? Number.POSITIVE_INFINITY;
        if (cost === Number.POSITIVE_INFINITY) continue;
        candidates[kept++] = document;
        if (scores) scores[document] = (scores[document] ?? 0) + cost;
      }
      candidates.length = kept;
    }
    if (!candidates.length) return [];
  }
  const ranked = (candidates ?? []).sort((left, right) => {
    const first = index.documents[left];
    const second = index.documents[right];
    return (
      (scores?.[left] ?? 0) - (scores?.[right] ?? 0) ||
      (first?.ts ?? 0) - (second?.ts ?? 0) ||
      compareText(first?.id ?? '', second?.id ?? '') ||
      left - right
    );
  });
  return ranked.slice(0, Math.floor(limit)).flatMap(documentIndex => {
    const document = index.documents[documentIndex];
    if (!document) return [];
    const ranges = (index.occurrences[documentIndex] ?? [])
      .filter(occurrence => matches.some(match => match.words.has(occurrence.word)))
      .map(occurrence => [...occurrence.range] as [number, number]);
    return [{ document, score: scores?.[documentIndex] ?? 0, ranges }];
  });
}
