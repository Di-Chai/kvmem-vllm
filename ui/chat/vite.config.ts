import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, transformWithEsbuild } from 'vite';
import { resolve } from 'node:path';
import { nerdamerPlugin } from './scripts/vite-plugin-nerdamer';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit(), nerdamerPlugin(), {
    // The pinned Svelte compiler retains optional parameter markers in emitted
    // snippet functions. Erase those TypeScript markers before Rollup parses JS.
    name: 'kvmem:erase-svelte-types',
    enforce: 'post',
    transform(code, id) {
      if (id.endsWith('.svelte'))
        return transformWithEsbuild(code, id, { loader: 'ts', target: 'esnext', sourcemap: false });
    },
  }],
  resolve: { alias: { 'katex-fonts': resolve('node_modules/katex/dist/fonts') } },
  build: { assetsInlineLimit: 32000, chunkSizeWarningLimit: 3072 },
});
