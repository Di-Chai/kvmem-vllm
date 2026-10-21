// Read-only same-origin preview for the owned CSGO acceptance fixture.
import { request } from 'node:http';

export const name = 'kvmem-csgo-preview';
export const inject = ['webServer'];

export function apply(ctx) {
  ctx.effect(() => ctx.webServer.register({
    kind: 'prefix', path: '/kvmem-csgo',
    async handler(req, res) {
      const url = new URL(req.url, 'http://localhost');
      if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
      if (url.pathname === '/kvmem-csgo') {
        res.writeHead(302, { location: '/kvmem-csgo/' }); res.end(); return;
      }
      const path = url.pathname.slice('/kvmem-csgo'.length);
      if (!['/', '/index.html', '/css/style.css', '/js/game.js', '/js/three.min.js'].includes(path)) {
        res.writeHead(404); res.end(); return;
      }
      await new Promise(resolve => {
        const upstream = request({ hostname: '127.0.0.1', port: 18921,
          method: req.method, path: path + url.search, timeout: 15000 }, response => {
          res.writeHead(response.statusCode || 502, {
            'content-type': response.headers['content-type'] || 'application/octet-stream',
            'cache-control': 'no-store', 'x-content-type-options': 'nosniff',
          });
          response.pipe(res);
          response.on('end', resolve);
          response.on('error', () => { res.destroy(); resolve(); });
        });
        upstream.on('timeout', () => upstream.destroy(new Error('Preview timeout')));
        upstream.on('error', () => {
          if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain; charset=utf-8' });
          res.end('游戏预览服务暂时不可用。'); resolve();
        });
        res.on('close', () => upstream.destroy());
        upstream.end();
      });
    },
  }));
}
