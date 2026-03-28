/** Converte score de similaridade (0–1) para string legível. Ex: 0.923 → "92%" */
export function formatSimilarity(value: number): string {
  return `${Math.round(value * 100)}%`
}
