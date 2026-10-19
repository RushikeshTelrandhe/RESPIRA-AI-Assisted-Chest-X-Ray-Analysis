type Entry = { value?: unknown; promise?: Promise<unknown>; expires: number };
/** Session-scoped, bounded cache. Never written to browser storage. */
export class SessionCache {
  private entries = new Map<string, Entry>();
  constructor(
    private limit = 18,
    private ttl = 180_000,
  ) {}
  peek<T>(key: string): T | undefined {
    const e = this.entries.get(key);
    return e && e.expires > Date.now() ? (e.value as T | undefined) : undefined;
  }
  get<T>(key: string, loader: () => Promise<T>): Promise<T> {
    const existing = this.entries.get(key);
    if (existing?.promise) return existing.promise as Promise<T>;
    if (
      existing &&
      existing.expires > Date.now() &&
      existing.value !== undefined
    )
      return Promise.resolve(existing.value as T);
    const entry: Entry = { expires: Date.now() + this.ttl };
    this.entries.set(key, entry);
    while (this.entries.size > this.limit)
      this.entries.delete(this.entries.keys().next().value!);
    const promise = Promise.resolve()
      .then(loader)
      .then(
        (value) => {
          // Invalidation/logout must prevent an old request repopulating the cache.
          if (this.entries.get(key) === entry) {
            entry.value = value;
            entry.expires = Date.now() + this.ttl;
            entry.promise = undefined;
          }
          return value;
        },
        (error) => {
          if (this.entries.get(key) === entry) this.entries.delete(key);
          throw error;
        },
      );
    entry.promise = promise;
    return promise;
  }
  invalidate(prefix = "") {
    for (const key of this.entries.keys())
      if (key.startsWith(prefix)) this.entries.delete(key);
  }
  clear() {
    this.entries.clear();
  }
}
