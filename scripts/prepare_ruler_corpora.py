"""Fetch the corpora named by the frozen classic RULER generator.

Preserve downloaded bytes and hashes. Essay ordering and HTML conversion match
the upstream downloader; failed downloads abort instead of dropping essays.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from ruler_protocol import UPSTREAM_COMMIT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                data = response.read()
            if not data:
                raise ValueError('Empty download: ' + url)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare(upstream, cache):
    import html2text
    from bs4 import BeautifulSoup
    import nltk

    target = upstream / 'scripts/data/synthetic/json'
    urls = (target / 'PaulGrahamEssays_URLs.txt').read_text().splitlines()
    urls = [url.strip() for url in urls if url.strip()]

    def essay(url):
        is_html = url.endswith('.html')
        # HTTPS retrieves the same corpus when plaintext HTTP is unavailable.
        resolved = url.replace('http://www.paulgraham.com/', 'https://www.paulgraham.com/')
        resolved = resolved.replace('https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/',
                                    'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/')
        name = resolved.rsplit('/', 1)[1]
        data = fetch(resolved, cache / 'essays' / name)
        if is_html:
            converter = html2text.HTML2Text()
            converter.ignore_images = converter.ignore_tables = converter.escape_all = True
            converter.reference_links = converter.mark_code = False
            soup = BeautifulSoup(data.decode('unicode_escape', 'replace'), 'html.parser')
            tag = soup.find('font')
            if tag is None:
                raise ValueError('Expected upstream essay font element: ' + url)
            text = converter.handle(str(tag))
        else:
            text = data.decode('utf-8')
        if not text.strip():
            raise ValueError('Empty essay: ' + url)
        return dict(url=url, resolved_url=resolved, raw_sha256=digest(data),
                    text_sha256=digest(text.encode()), name=name, html=is_html, text=text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        essays = list(pool.map(essay, urls))
    essays.sort(key=lambda row: (row['html'], row['name'].replace('.html', '.txt')))
    (target / 'PaulGrahamEssays.json').write_text(json.dumps({'text': ''.join(r['text'] for r in essays)}))
    sources = {
        'squad.json': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
        'hotpotqa.json': 'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/'
                        '7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
    }
    for name, url in sources.items():
        data = fetch(url, cache / name)
        parsed = json.loads(data)
        if not parsed:
            raise ValueError('Empty QA dataset: ' + name)
        (target / name).write_bytes(data)
    # GitHub source archives contain a Git LFS pointer for this vocabulary.
    vocabulary_url = (f'https://media.githubusercontent.com/media/NVIDIA/RULER/{UPSTREAM_COMMIT}/'
                      'scripts/data/synthetic/json/english_words.json')
    vocabulary = fetch(vocabulary_url, cache / 'english_words.json')
    if digest(vocabulary) != 'affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca':
        raise ValueError('English vocabulary does not match the upstream Git LFS object')
    if not isinstance(json.loads(vocabulary), dict):
        raise ValueError('Invalid official vocabulary')
    (target / 'english_words.json').write_bytes(vocabulary)
    nltk.download('punkt_tab', download_dir=str(cache / 'nltk_data'), raise_on_error=True)
    manifest = dict(essays=[{k:v for k,v in r.items() if k != 'text'} for r in essays],
        qa_sources=sources, vocabulary_url=vocabulary_url,
        files={p.name:digest(p.read_bytes()) for p in target.glob('*.json')},
        essay_count=len(essays), transport_adaptation='HTTP to HTTPS and equivalent GitHub raw URLs')
    (cache / 'corpora-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='prepared', essays=len(essays), files=manifest['files'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.upstream.resolve(), args.cache.resolve())
