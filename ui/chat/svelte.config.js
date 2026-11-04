import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';

const digest = createHash('sha256');
for (const name of ['package-lock.json', ...readdirSync('src', { recursive: true }).sort().map(name => 'src/' + name)]) {
  if (statSync(name).isFile()) digest.update(name).update(readFileSync(name));
}

export default {
  preprocess: vitePreprocess(),
  kit: {
    version: { name: digest.digest('hex').slice(0, 16) },
    adapter: adapter({ pages: 'dist', assets: 'dist', fallback: 'index.html' }),
    output: { bundleStrategy: 'single' },
    paths: { base: '/chat/view', relative: false },
    router: { type: 'hash' },
  },
};
