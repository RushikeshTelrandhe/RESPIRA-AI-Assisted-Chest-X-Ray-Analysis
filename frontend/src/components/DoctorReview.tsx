import { useEffect, useRef, useState, type FormEvent } from "react";
import { useLocation } from "react-router-dom";
import { Check, ClipboardPen, Download, MessageSquareText, Save, Stethoscope } from "lucide-react";
import { api, DISEASES, type AnalysisDetail, type Patient } from "../services/api";
import { agreementLabels, decodeNotes, emptyReview, findReview, saveReview, type Agreement, type ClinicalReview, type ReviewFields } from "../services/clinicalReviews";
import { useAuth } from "../context/AuthContext";
import { useUnsavedChanges, useWorkspace } from "../context/WorkspaceContext";
import { Alert } from "./UI";
import { dateLabel, errorText } from "../utils/display";

export function DoctorReview({ analysis }: { analysis: AnalysisDetail }) {
  const { doctor, token } = useAuth();
  const { cache } = useWorkspace();
  const location = useLocation();
  const [fields, setFields] = useState<ReviewFields>({ ...emptyReview });
  const [saved, setSaved] = useState<ClinicalReview>();
  const [baseline, setBaseline] = useState<ReviewFields>({ ...emptyReview });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const pending = useRef(false);
  const section = useRef<HTMLElement>(null);
  const [revision, setRevision] = useState(0);
  const dirty = JSON.stringify(fields) !== JSON.stringify(baseline);
  useUnsavedChanges(dirty, "Your clinical notes or doctor feedback have not been saved. Leave and discard these changes?");
  useEffect(() => {
    let active = true;
    setLoading(true); setError("");
    void api.get<Patient>(`/api/v1/patients/${encodeURIComponent(analysis.patient.id)}`, token).then(patient => {
      const existing = findReview(decodeNotes(patient.notes || "").reviews, analysis.analysis_id, doctor?.id || "");
      if (!active) return;
      const initial = Object.fromEntries(Object.keys(emptyReview).map(key => [key, existing?.[key as keyof ReviewFields] ?? emptyReview[key as keyof ReviewFields]])) as ReviewFields;
      setFields(initial); setBaseline(initial); setSaved(existing);
    }).catch(e => { if (active) setError(errorText(e)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [analysis.patient.id, analysis.analysis_id, doctor?.id, token, revision]);
  useEffect(() => {
    if (!loading && location.hash === "#doctor-feedback") section.current?.scrollIntoView({ block: "start" });
  }, [loading, location.hash]);
  function update<K extends keyof ReviewFields>(key: K, value: ReviewFields[K]) {
    setFields(current => ({ ...current, [key]: value })); setConfirmed(false);
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!doctor || loading || pending.current || !dirty) return;
    pending.current = true; setSaving(true); setError(""); setConfirmed(false);
    const now = new Date().toISOString();
    const review: ClinicalReview = { ...fields, analysisId: analysis.analysis_id, doctorId: doctor.id, doctorName: doctor.full_name,
      modelVersion: analysis.model.version, createdAt: saved?.createdAt || now, updatedAt: now };
    try {
      const stored = await saveReview(analysis.patient.id, review, saved, token);
      setSaved(stored); setBaseline({ ...fields }); setConfirmed(true);
      cache.invalidate(`patient:${analysis.patient.id}`); cache.invalidate("patients:");
    } catch (e) { setError(errorText(e)); }
    finally { pending.current = false; setSaving(false); }
  }
  function exportNotes() {
    const contents = [`RESPIRA — Doctor review ${dirty ? '(unsaved draft)' : ''}`, `Analysis: ${analysis.analysis_id}`, `Doctor: ${doctor?.full_name || ''}`,
      `Model: ${analysis.model.version}`, `Last saved: ${saved?.updatedAt || 'Not saved'}`, '', 'OBSERVATIONS', fields.observations || '—', '',
      'CLINICAL ASSESSMENT', fields.assessment || '—', '', 'FOLLOW-UP NOTES', fields.followUp || '—', '', 'DOCTOR FEEDBACK',
      `Agreement: ${agreementLabels[fields.agreement]}`, `Suggested finding: ${fields.suggestedFinding || '—'}`,
      `Image quality: ${fields.imageQuality}`, `Explanation: ${fields.explanationUseful}`, fields.feedback || '—'].join('\n');
    const url = URL.createObjectURL(new Blob([contents], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `respira-review-${analysis.analysis_id.slice(0,8)}.txt`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="doctor-review" id="doctor-feedback" ref={section} aria-labelledby="doctor-review-heading">
    <div className="doctor-review-head"><div className="section-heading"><span className="review-symbol"><Stethoscope size={23} /></span><div><p className="eyebrow">Your clinical perspective</p><h2 id="doctor-review-heading">Doctor notes & feedback</h2><p>Record your assessment alongside this study.</p></div></div>
      <span className={`review-save-state${dirty ? ' is-dirty' : ''}`} role="status">{loading ? 'Loading saved review…' : saving ? 'Saving…' : dirty ? 'Unsaved changes' : saved ? `Saved ${dateLabel(saved.updatedAt, true)}` : 'No review saved yet'}</span>
    </div>
    {loading ? <div className="review-loading"><span className="spin" /> Loading your notes…</div> : <form onSubmit={submit}>
      {error && <Alert>{error}{!dirty && <button type="button" className="text-button" onClick={() => setRevision(v => v + 1)}>Reload saved review</button>}</Alert>}
      <fieldset disabled={saving || (Boolean(error) && !dirty)} className="review-fieldset">
        <div className="review-form-grid">
          <div className="review-notes-column"><h3><ClipboardPen size={19} /> Clinical notes</h3><p className="small muted">Your observations remain separate from the AI output.</p>
            <label className="label" htmlFor="review-observations">Observations</label><textarea id="review-observations" className="input" rows={4} maxLength={2500} placeholder="Describe what you observe in this study…" value={fields.observations} onChange={e => update('observations', e.target.value)} />
            <label className="label" htmlFor="review-assessment">Clinical assessment</label><textarea id="review-assessment" className="input" rows={3} maxLength={2500} placeholder="Add your interpretation and relevant clinical context…" value={fields.assessment} onChange={e => update('assessment', e.target.value)} />
            <label className="label" htmlFor="review-followup">Follow-up notes</label><textarea id="review-followup" className="input" rows={3} maxLength={2500} placeholder="Record next steps you want to remember…" value={fields.followUp} onChange={e => update('followUp', e.target.value)} />
          </div>
          <div className="review-feedback-column"><h3><MessageSquareText size={19} /> Doctor feedback</h3><p className="small muted">Help document agreement and the usefulness of the explanation.</p>
            <fieldset className="review-agreement"><legend>Do you agree with the model output?</legend><div>{(['agree','partly_agree','disagree','unsure'] as Agreement[]).map(value => <label className={fields.agreement === value ? 'selected' : ''} key={value}><input type="radio" name="review-agreement" value={value} checked={fields.agreement === value} onChange={() => update('agreement',value)} />{agreementLabels[value]}</label>)}</div></fieldset>
            <label className="label" htmlFor="review-suggestion">Suggested finding <span className="muted">(optional)</span></label><select id="review-suggestion" className="input" value={fields.suggestedFinding} onChange={e => update('suggestedFinding', e.target.value)}><option value="">No alternative selected</option>{DISEASES.map(d => <option key={d}>{d}</option>)}<option>Other — describe below</option></select>
            <div className="review-select-grid"><div><label className="label" htmlFor="review-quality">Image quality</label><select id="review-quality" className="input" value={fields.imageQuality} onChange={e => update('imageQuality', e.target.value as ReviewFields['imageQuality'])}><option value="not_reviewed">Not assessed</option><option value="adequate">Adequate</option><option value="limited">Limited</option><option value="unusable">Unusable</option></select></div>
            <div><label className="label" htmlFor="review-helpful">Visual explanation</label><select id="review-helpful" className="input" value={fields.explanationUseful} onChange={e => update('explanationUseful', e.target.value as ReviewFields['explanationUseful'])}><option value="not_reviewed">Not assessed</option><option value="helpful">Helpful</option><option value="partly_helpful">Partly helpful</option><option value="not_helpful">Not helpful</option></select></div></div>
            <label className="label" htmlFor="review-feedback">Feedback details</label><textarea id="review-feedback" className="input" rows={5} maxLength={2500} placeholder="What matched your assessment? What needs closer review?" value={fields.feedback} onChange={e => update('feedback',e.target.value)} />
            <p className="review-feedback-note">Feedback is recorded with this patient. It does not automatically retrain the model.</p>
          </div>
        </div>
      </fieldset>
      <div className="review-savebar"><div><span className="small muted">Saved with the patient record and linked to this analysis and doctor.</span>{confirmed && <span className="review-confirmed" role="status"><Check size={15} /> Saved and confirmed by the server</span>}</div>
        <div className="page-actions"><button type="button" className="btn-ghost" disabled={saving} onClick={exportNotes}><Download size={16} /> Export notes</button><button className="btn-primary" type="submit" disabled={!dirty || saving || !doctor}><Save size={17} />{saving ? 'Saving…' : 'Save notes & feedback'}</button></div>
      </div>
    </form>}
  </section>;
}
