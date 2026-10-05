import { api, type Patient } from "./api";

const START = "\n\n[RESPIRA_CLINICAL_REVIEWS_V1]\n";
const END = "\n[/RESPIRA_CLINICAL_REVIEWS_V1]";
export type Agreement = "not_reviewed" | "agree" | "partly_agree" | "disagree" | "unsure";
export type ReviewFields = {
  observations: string;
  assessment: string;
  followUp: string;
  agreement: Agreement;
  suggestedFinding: string;
  imageQuality: "not_reviewed" | "adequate" | "limited" | "unusable";
  explanationUseful: "not_reviewed" | "helpful" | "partly_helpful" | "not_helpful";
  feedback: string;
};
export type ClinicalReview = ReviewFields & {
  analysisId: string; doctorId: string; doctorName: string; modelVersion: string;
  createdAt: string; updatedAt: string;
};
export const emptyReview: ReviewFields = {
  observations: "", assessment: "", followUp: "", agreement: "not_reviewed", suggestedFinding: "",
  imageQuality: "not_reviewed", explanationUseful: "not_reviewed", feedback: "",
};
export const agreementLabels: Record<Agreement, string> = {
  not_reviewed: "Not reviewed", agree: "Agree", partly_agree: "Partly agree", disagree: "Disagree", unsure: "Unsure",
};

/** Preserve ordinary patient notes and unrelated reviews when writing a case review. */
export function decodeNotes(raw: string): { text: string; reviews: ClinicalReview[] } {
  const start = raw.indexOf(START);
  if (start < 0) {
    if (raw.includes("[RESPIRA_CLINICAL_REVIEWS_V1]")) throw new Error("The saved review section is incomplete. No changes have been made.");
    return { text: raw, reviews: [] };
  }
  const end = raw.indexOf(END, start);
  if (end < 0 || raw.indexOf(START, start + START.length) >= 0) throw new Error("The saved review section could not be read. No changes have been made.");
  try {
    const data = JSON.parse(raw.slice(start + START.length, end));
    if (data.version !== 1 || !Array.isArray(data.reviews) || !data.reviews.every((r: ClinicalReview) =>
      r && typeof r.analysisId === "string" && typeof r.doctorId === "string" && typeof r.updatedAt === "string" &&
      Object.keys(emptyReview).every(key => typeof (r as unknown as Record<string, unknown>)[key] === "string") && r.agreement in agreementLabels)) throw new Error();
    return { text: raw.slice(0, start) + raw.slice(end + END.length), reviews: data.reviews };
  } catch { throw new Error("The saved review section could not be read. No changes have been made."); }
}

export function encodeNotes(text: string, reviews: ClinicalReview[]) {
  return reviews.length ? text + START + JSON.stringify({ version: 1, reviews }) + END : text;
}
export function findReview(reviews: ClinicalReview[], analysisId: string, doctorId: string) {
  return reviews.find(r => r.analysisId === analysisId && r.doctorId === doctorId);
}
export function mergeReview(raw: string, review: ClinicalReview, expected?: ClinicalReview) {
  const notes = decodeNotes(raw);
  const current = findReview(notes.reviews, review.analysisId, review.doctorId);
  if (JSON.stringify(current) !== JSON.stringify(expected)) throw new Error("This review changed in another window. Reload the saved review before saving again; your text is still here.");
  const others = notes.reviews.filter(r => r.analysisId !== review.analysisId || r.doctorId !== review.doctorId);
  return encodeNotes(notes.text, [...others, review]);
}

export async function saveReview(patientId: string, review: ClinicalReview, expected: ClinicalReview | undefined, token: string | null) {
  const path = `/api/v1/patients/${encodeURIComponent(patientId)}`;
  const latest = await api.get<Patient>(path, token);
  const notes = mergeReview(latest.notes || "", review, expected);
  await api.put<Patient>(path, { notes }, token);
  // Read back the record: a successful HTTP status alone is not proof of persistence.
  const confirmed = await api.get<Patient>(path, token);
  const saved = findReview(decodeNotes(confirmed.notes || "").reviews, review.analysisId, review.doctorId);
  if (JSON.stringify(saved) !== JSON.stringify(review)) throw new Error("The server did not confirm this review. Your changes remain unsaved; please retry.");
  return saved!;
}

export async function savePatientNotes(patientId: string, text: string, expectedText: string, token: string | null) {
  const path = `/api/v1/patients/${encodeURIComponent(patientId)}`;
  const latest = await api.get<Patient>(path, token);
  const current = decodeNotes(latest.notes || "");
  if (current.text !== expectedText) throw new Error("Patient notes changed in another window. Reload before saving to avoid replacing those edits.");
  await api.put<Patient>(path, { notes: encodeNotes(text, current.reviews) }, token);
  const saved = await api.get<Patient>(path, token);
  if (decodeNotes(saved.notes || "").text !== text) throw new Error("The server did not confirm the saved notes. Please retry.");
  return saved;
}
