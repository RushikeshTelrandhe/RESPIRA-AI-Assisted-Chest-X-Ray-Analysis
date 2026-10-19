import { ArrowUpRight, Copy, Mail } from "lucide-react";
import { useState } from "react";
export function DeveloperContact() {
  const [copyState, setCopyState] = useState('Copy email');
  async function copy() {
    try { await navigator.clipboard.writeText('respirahelp@gmail.com'); setCopyState('Email copied'); }
    catch { setCopyState('Select the email address to copy it'); }
  }
  return <section className="developer-contact" aria-labelledby="developer-contact-heading"><div className="contact-orbit" aria-hidden="true"><Mail size={27} /></div><div><p className="eyebrow">Here to help</p><h2 id="developer-contact-heading">Contact the RESPIRA developers</h2><p>Report a technical issue, suggest an improvement, or ask about the project.</p><a className="contact-email" href="mailto:respirahelp@gmail.com?subject=RESPIRA%20support">respirahelp@gmail.com <ArrowUpRight size={18} /></a><p className="small muted">Please leave patient names, scans and clinical notes out of support emails.</p></div><button type="button" className="btn-ghost" onClick={() => void copy()}><Copy size={16} /> <span role="status">{copyState}</span></button></section>;
}
