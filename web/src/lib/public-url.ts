/** Resolves a path under `public/` for GitHub Pages (`base` may or may not end with `/`). */
export function publicUrl(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const clean = path.replace(/^\//, "");
  return `${base}/${clean}`;
}
