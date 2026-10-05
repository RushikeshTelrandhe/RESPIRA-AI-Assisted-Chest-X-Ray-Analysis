/** Optional production media. Use same-origin, licensed files under public/media.
 * No generation provider key is ever sent to the browser.
 * The visual fallback runs automatically if a clip is unavailable.
 */
export const INTRO_MEDIA: { video: string | null; voice: string | null; music: string | null } = {
  video: null,
  voice: null,
  music: null,
};
