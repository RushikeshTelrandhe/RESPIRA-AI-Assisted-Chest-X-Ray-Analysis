export type ExplanationResult = {
  image?: string;
  heatmap?: string;
  overlay?: string;
  target_class?: string;
  attention_map?: number[][];
  explanation?: string;
};

/** Accept actual image sources; never substitute one explanation for another. */
export function imageSource(value?: string): string | undefined {
  const src = value?.trim();
  if (!src) return undefined;
  if (/^\/9j\//.test(src)) return `data:image/jpeg;base64,${src}`;
  if (/^(data:image\/(png|jpeg|webp);base64,|blob:|https?:\/\/|\/)/i.test(src)) return src;
  if (/^iVBORw0KGgo/.test(src)) return `data:image/png;base64,${src}`;
  return undefined;
}

export function explanationImages(result: ExplanationResult) {
  return {
    original: imageSource(result.image),
    overlay: imageSource(result.overlay),
    heatmap: imageSource(result.heatmap),
  };
}

export function explanationKey(analysis: string, model: string, method: "gradcam" | "vit", target: number) {
  return `explain-v4:${analysis}:${model}:${method}${method === "gradcam" ? `:${target}` : ""}`;
}

export function fitImage(width: number, height: number, availableWidth: number, availableHeight: number) {
  if (Math.min(width, height, availableWidth, availableHeight) <= 0) return { width: 0, height: 0 };
  const ratio = Math.min(availableWidth / width, availableHeight / height);
  return { width: width * ratio, height: height * ratio };
}
