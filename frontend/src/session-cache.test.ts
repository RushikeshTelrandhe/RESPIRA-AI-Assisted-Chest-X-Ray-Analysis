import { describe, expect, it, vi } from "vitest";
import { SessionCache } from "./services/SessionCache";
describe("session-scoped request cache", () => {
  it("deduplicates the same in-flight request", async () => {
    const cache = new SessionCache();
    const loader = vi.fn(async () => ({ patient: "one" }));
    const [a, b] = await Promise.all([
      cache.get("analysis:1", loader),
      cache.get("analysis:1", loader),
    ]);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(a).toEqual(b);
  });
  it("keeps explanation method and target keys separate", async () => {
    const cache = new SessionCache();
    await cache.get("explain:a:gradcam:0", async () => 1);
    await cache.get("explain:a:gradcam:1", async () => 2);
    await cache.get("explain:a:vit", async () => 3);
    expect(cache.peek("explain:a:gradcam:0")).toBe(1);
    expect(cache.peek("explain:a:gradcam:1")).toBe(2);
    expect(cache.peek("explain:a:vit")).toBe(3);
  });
  it("does not repopulate after private session data is cleared", async () => {
    const cache = new SessionCache();
    let release: (value: string) => void = () => undefined;
    const pending = cache.get(
      "patient:private",
      () =>
        new Promise<string>((resolve) => {
          release = resolve;
        }),
    );
    await Promise.resolve();
    cache.clear();
    release("late response");
    await pending;
    expect(cache.peek("patient:private")).toBeUndefined();
  });
});
