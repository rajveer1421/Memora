// Memora service worker: offline fallback page, caregiver push alerts (tap opens the dashboard).
const CACHE = 'memora-v1';
const OFFLINE = '/static/offline.html';
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.add(OFFLINE)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));

// Pages always come from the network (patient data must be fresh); only when that fails is the
// cached offline page shown. API calls and files are never cached.
self.addEventListener('fetch', event => {
    if (event.request.mode !== 'navigate') return;
    event.respondWith(fetch(event.request).catch(() => caches.match(OFFLINE)));
});

self.addEventListener('push', event => {
    let data = {title: 'Memora', body: 'New alert', url: '/dashboard'};
    try { data = {...data, ...event.data.json()}; } catch (e) {}
    event.waitUntil(self.registration.showNotification(data.title, {
        body: data.body, data: {url: data.url}, tag: 'memora-alert', renotify: true, requireInteraction: true,
    }));
});

self.addEventListener('notificationclick', event => {
    event.notification.close();
    const url = (event.notification.data && event.notification.data.url) || '/dashboard';
    event.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then(list => {
        for (const c of list) if (c.url.includes(url) && 'focus' in c) return c.focus();
        return self.clients.openWindow(url);
    }));
});
