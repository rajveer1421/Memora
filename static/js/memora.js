// Shared helpers for every page.
// esc() must wrap ANY user-provided value inserted into HTML (names, notes, transcripts...).
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const CSRF = document.querySelector('meta[name="csrf-token"]')?.content || '';

async function api(url, opts = {}) {
    const o = {...opts, headers: {...(opts.headers || {}), 'X-CSRF-Token': CSRF}, credentials: 'same-origin'};
    if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers['Content-Type'] = 'application/json'; delete o.json; }
    let res;
    try { res = await fetch(url, o); } catch (e) { return {status: 'fail', message: 'Network error. Check your connection.', http: 0}; }
    let data;
    try { data = await res.json(); } catch (e) { data = {status: res.ok ? 'success' : 'fail', message: res.statusText}; }
    if (Array.isArray(data)) return data;
    data.http = res.status;
    if (res.status === 401 && !url.startsWith('/api/stats')) {
        const patientPage = /^\/(patient|things|recognize|home-guide)/.test(location.pathname);
        location.href = (patientPage ? '/patient/login' : '/caregiver/login') + '?next=' + encodeURIComponent(location.pathname);
    }
    return data;
}

// A Lucide line icon from the local sprite (same markup as the Jinja icon() helper).
const ic = (name, cls = '') => `<svg class="i ${cls}" aria-hidden="true"><use href="/static/icons.svg?v=1#${name}"></use></svg>`;

// Initials in a coloured circle: Memora never keeps face photos, so people are shown this way.
function avatar(name, size = 44) {
    const hues = [142, 217, 271, 173, 24, 330, 239];
    const h = hues[[...String(name)].reduce((a, c) => a + c.charCodeAt(0), 0) % hues.length];
    const ini = String(name || '?').trim().split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();
    return `<span class="avatar" style="width:${size}px;height:${size}px;font-size:${size * .38}px;background:hsl(${h} 70% 92%);color:hsl(${h} 60% 32%)">${esc(ini)}</span>`;
}

function toast(msg, ok) {
    let box = document.getElementById('toasts');
    if (!box) { box = document.createElement('div'); box.id = 'toasts'; box.className = 'toast-box'; document.body.appendChild(box); }
    const t = document.createElement('div');
    t.className = 'toast ' + (ok ? 'toast-ok' : 'toast-err');
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// Speak `text`, then call `onDone` once speech has finished (or immediately if speech isn't available).
// A timer estimated from the text length backs up the 'end' event, which some browsers don't always fire.
function speak(text, rate = 0.85, onDone = null) {
    let finished = false;
    const done = () => { if (!finished) { finished = true; if (onDone) onDone(); } };
    if (!('speechSynthesis' in window) || !text) { done(); return; }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate; u.pitch = 1.0; u.volume = 1;
    u.onend = done; u.onerror = done;
    speechSynthesis.speak(u);
    setTimeout(done, 1500 + (text.length / (14 * rate)) * 1000);
}

// Say who it is first, then play their personal voice message (never both at once).
function speakThenPlay(text, audioEl) {
    speak(text, 0.85, () => { if (audioEl) audioEl.play().catch(() => {}); });
}

function timeAgo(iso) {
    if (!iso) return '';
    const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
    if (mins < 2) return 'just now';
    if (mins < 60) return mins + ' min ago';
    if (mins < 1440) return Math.round(mins / 60) + ' h ago';
    return Math.round(mins / 1440) + ' d ago';
}

function fmtTime(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}

// Voice recorder that labels the upload with the browser's real format (Safari records MP4, not WebM).
class VoiceRecorder {
    async start() {
        this.stream = await navigator.mediaDevices.getUserMedia({audio: true});
        const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
        const mimeType = types.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
        this.rec = new MediaRecorder(this.stream, mimeType ? {mimeType} : undefined);
        this.chunks = [];
        this.rec.ondataavailable = e => e.data.size && this.chunks.push(e.data);
        this.rec.start();
    }
    stop() {
        return new Promise(resolve => {
            this.rec.onstop = () => {
                this.stream.getTracks().forEach(t => t.stop());
                const type = (this.rec.mimeType || 'audio/webm').split(';')[0];
                const ext = {'audio/mp4': 'm4a', 'audio/ogg': 'ogg'}[type] || 'webm';
                const blob = new Blob(this.chunks, {type});
                resolve(new File([blob], 'voice.' + ext, {type}));
            };
            this.rec.stop();
        });
    }
    get recording() { return this.rec && this.rec.state === 'recording'; }
}

// Web Push for caregivers (free; only works if the server has VAPID keys).
async function enablePush() {
    const key = await api('/api/push/key');
    if (!key.enabled) { toast('Push notifications are not configured on this server.', false); return false; }
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) { toast('This browser does not support push notifications.', false); return false; }
    const perm = await Notification.requestPermission();
    if (perm !== 'granted') { toast('Notifications were not allowed.', false); return false; }
    const reg = await navigator.serviceWorker.register('/sw.js', {scope: '/'});
    const raw = atob(key.public_key.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((key.public_key.length + 3) % 4));
    const sub = await reg.pushManager.subscribe({userVisibleOnly: true, applicationServerKey: Uint8Array.from(raw, c => c.charCodeAt(0))});
    const r = await api('/api/push/subscribe', {method: 'POST', json: sub.toJSON()});
    toast(r.status === 'success' ? 'Alerts will be pushed to this device.' : r.message, r.status === 'success');
    return r.status === 'success';
}
