import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, type Patient } from './services/api';
import { decodeNotes, emptyReview, encodeNotes, mergeReview, saveReview, savePatientNotes, type ClinicalReview } from './services/clinicalReviews';
import { explanationImages, explanationKey, fitImage, imageSource } from './services/explanationImages';
const review: ClinicalReview = { ...emptyReview, observations: 'TEST FIXTURE — no patient', analysisId:'test-study', doctorId:'test-doctor', doctorName:'Test Doctor', modelVersion:'test-model', createdAt:'2026-01-01T00:00:00Z', updatedAt:'2026-01-01T00:00:00Z' };
const patient = (notes: string) => ({ id:'test-patient', notes }) as Patient;
afterEach(() => vi.restoreAllMocks());
describe('Clinical review persistence', () => {
  it('retains existing notes and reviews belonging to other studies and doctors', () => {
    const other = {...review,doctorId:'other-doctor', observations:'Other saved text'};
    const input=encodeNotes('General notes\nKeep all formatting.',[other]);
    const output=decodeNotes(mergeReview(input,review,undefined));
    expect(output.text).toBe('General notes\nKeep all formatting.');
    expect(output.reviews).toEqual([other,review]);
  });
  it('updates a review without creating duplicate entries', () => {
    const changed={...review,assessment:'Updated',updatedAt:'2026-01-02T00:00:00Z'};
    expect(decodeNotes(mergeReview(encodeNotes('',[review]),changed,review)).reviews).toEqual([changed]);
  });
  it('refuses to overwrite a newer saved review', () => {
    const newer={...review,assessment:'A later edit'};
    expect(() => mergeReview(encodeNotes('',[newer]),review,review)).toThrow('changed in another window');
  });
  it('does not damage incomplete or corrupt saved review blocks', () => {
    expect(() => decodeNotes('notes\n\n[RESPIRA_CLINICAL_REVIEWS_V1]\n{')).toThrow();
    expect(() => decodeNotes('notes\n\n[RESPIRA_CLINICAL_REVIEWS_V1]\n{}\n[/RESPIRA_CLINICAL_REVIEWS_V1]')).toThrow();
  });
  it('saves through the patient endpoint and verifies the returned record', async () => {
    const get=vi.spyOn(api,'get').mockResolvedValueOnce(patient('Prior note')).mockResolvedValueOnce(patient(encodeNotes('Prior note',[review])));
    const put=vi.spyOn(api,'put').mockResolvedValue({});
    expect(await saveReview('test-patient',review,undefined,'test-token')).toEqual(review);
    expect(put).toHaveBeenCalledWith('/api/v1/patients/test-patient',{notes:encodeNotes('Prior note',[review])},'test-token');
    expect(get).toHaveBeenCalledTimes(2);
  });
  it('does not show success when a PUT succeeds but the server does not store the review', async () => {
    vi.spyOn(api,'get').mockResolvedValue(patient('Original'));
    vi.spyOn(api,'put').mockResolvedValue({});
    await expect(saveReview('test-patient',review,undefined,'test-token')).rejects.toThrow('did not confirm');
  });
  it('keeps study reviews when editing general patient notes', async () => {
    vi.spyOn(api,'get').mockResolvedValueOnce(patient(encodeNotes('old',[review]))).mockResolvedValueOnce(patient(encodeNotes('new',[review])));
    const put=vi.spyOn(api,'put').mockResolvedValue({});
    await savePatientNotes('test-patient','new','old','test-token');
    expect(decodeNotes((put.mock.calls[0][1] as {notes:string}).notes)).toEqual({text:'new',reviews:[review]});
  });
});
describe('Explanation image regression', () => {
  it('preserves different Grad-CAM and ViT overlays instead of reusing the common original', () => {
    const cam=explanationImages({image:'/original.png',heatmap:'/cam-map.png',overlay:'/cam-overlay.png'});
    const vit=explanationImages({image:'/original.png',heatmap:'/vit-map.png',overlay:'/vit-overlay.png'});
    expect(cam.original).toBe(vit.original); expect(cam.overlay).not.toBe(vit.overlay);
    expect(cam.overlay).toBe('/cam-overlay.png'); expect(vit.overlay).toBe('/vit-overlay.png');
  });
  it('separates cache entries by study, model, method and supported class', () => {
    const key=explanationKey('a','fusion','gradcam',0);
    expect(key).not.toBe(explanationKey('a','fusion','vit',0));
    expect(key).not.toBe(explanationKey('a','fusion','gradcam',1));
    expect(key).not.toBe(explanationKey('b','fusion','gradcam',0));
    expect(key).not.toBe(explanationKey('a','cnn','gradcam',0));
    expect(explanationKey('a','fusion','vit',0)).toBe(explanationKey('a','fusion','vit',1));
  });
  it('normalizes known raw base64 without turning it into a file path', () => {
    expect(imageSource('/9j/AAAA')).toBe('data:image/jpeg;base64,/9j/AAAA');
    expect(imageSource('iVBORw0KGgoAAAA')).toBe('data:image/png;base64,iVBORw0KGgoAAAA');
    expect(imageSource('javascript:alert(1)')).toBeUndefined();
    expect(explanationImages({image:'/original.png'}).overlay).toBeUndefined();
  });
  it.each([[1200,1800,700,500],[1800,1200,700,500],[224,224,370,600]])('fits the entire %sx%s image within a %sx%s viewport', (w,h,vw,vh) => {
    const fit=fitImage(w,h,vw,vh); expect(fit.width).toBeLessThanOrEqual(vw); expect(fit.height).toBeLessThanOrEqual(vh);
    expect(fit.width / fit.height).toBeCloseTo(w/h);
  });
});
