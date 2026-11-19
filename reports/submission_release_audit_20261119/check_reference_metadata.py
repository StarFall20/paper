"""Collect and compare authoritative metadata for the manuscript's cited sources."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from subprocess import run
from tempfile import TemporaryDirectory
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

USER_AGENT = 'ReferenceMetadataAudit/1.0 public-bibliography verification'


class Head(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = defaultdict(list)
        self.title = ''
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag=='meta':
            key = attributes.get('name',attributes.get('property','')).casefold()
            if key and attributes.get('content'):
                self.values[key].append(attributes['content'])
        if tag=='title':
            self.in_title = True

    def handle_endtag(self, tag):
        if tag=='title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def read_references(paths):
    entries = defaultdict(list)
    for name,path in paths.items():
        references = path.read_text().split('## References\n\n',1)[1]
        for line in references.splitlines():
            if not line.strip():
                continue
            assert re.search(r'\(\d{4}\)',line),line
            urls = re.findall(r'https?://[^\s]+',line)
            assert urls,line
            for url in urls:
                entries[url.rstrip('.,;')].append(dict(file=name,text=line))
    return entries


def fetch(url):
    quote_error = None
    for retry in range(2):
        try:
            req = Request(url,headers={'User-Agent': USER_AGENT,'Accept':'*/*'})
            with urlopen(req,timeout=14) as f:
                data = f.read(20_000_001)
                if len(data)>20_000_000:
                    raise OSError('Response exceeds metadata audit limit')
                return data, f.geturl()
        except (HTTPError,URLError,TimeoutError,OSError) as error:
            quote_error = error
            if retry==0 and isinstance(error,HTTPError) and error.code in (429,500,502,503,504):
                time.sleep(2)
                continue
            break
    raise quote_error


def crossref(doi):
    raw,route=fetch('https://api.crossref.org/works/'+quote(doi,safe=''))
    m=json.loads(raw)['message']
    roles=m.get('author') or m.get('editor') or []
    return dict(title=(m.get('title') or [''])[0],
                families=[x.get('family',x.get('name','')) for x in roles],
                title_language=m.get('language'),
                container=(m.get('container-title') or [''])[0],
                volume=m.get('volume'),issue=m.get('issue'),pages=m.get('page'),
                publisher=m.get('publisher'),received=m.get('published',{}).get('date-parts',[[]])[0],
                metadata='Crossref',metadata_url=route,body_sha256=sha256(raw).hexdigest())


def datacite(doi):
    raw,route=fetch('https://api.datacite.org/dois/'+quote(doi,safe='/'))
    m=json.loads(raw)['data']['attributes']
    return dict(title=m['titles'][0]['title'],families=[x.get('familyName',x.get('name','')) for x in m['creators']],
                received=[m.get('publicationYear')],publisher=m.get('publisher'),
                metadata='DataCite',metadata_url=route,body_sha256=sha256(raw).hexdigest())


def page(url, executable):
    alternative=url.replace('://arxiv.org/html/','://arxiv.org/abs/')
    raw,route=fetch(alternative)
    if raw.startswith(b'%PDF-'):
        with TemporaryDirectory() as td:
            loc=Path(td)/'source.pdf'
            loc.write_bytes(raw)
            extracted=run([executable,'-f','1','-l','1','-enc','UTF-8',str(loc),'-'],
                          check=True,capture_output=True).stdout.decode()
        return dict(title='',pdf_text=extracted[:8000],metadata='public first page',
                    metadata_url=route,body_sha256=sha256(raw).hexdigest())
    h=Head()
    h.feed(raw.decode('utf-8',errors='replace'))
    title=(h.values.get('citation_title') or h.values.get('dc.title') or h.values.get('og:title') or [h.title.strip()])[0]
    return dict(title=title,authors=h.values.get('citation_author',[])[:16],
                metadata='primary HTML',metadata_url=route,body_sha256=sha256(raw).hexdigest())


def retrieve(url, poppler):
    try:
        if url.startswith('https://doi.org/'):
            doi=url.split('doi.org/',1)[1]
            record=datacite(doi) if doi.startswith('10.5281/') else crossref(doi)
        else:
            record=page(url,poppler)
        return record
    except (HTTPError,URLError,TimeoutError,OSError,ValueError,KeyError,IndexError) as error:
        return dict(metadata='unavailable',reason=f'{type(error).__name__}: {error}')


def normalize(t):
    t=unicodedata.normalize('NFKD',unescape(t).casefold())
    t=''.join(c for c in t if not unicodedata.combining(c))
    return re.findall(r'[^\W_]+',t)


def similarity(title, printed):
    target=normalize(title)
    source=normalize(' '.join(printed.split()))
    if not target:
        return 0.0
    candidates=[source[:size] for size in range(max(1,len(target)-2),len(target)+4)]
    whole=' '.join(source)
    if ' '.join(target) in whole:
        return 1.0
    return max(SequenceMatcher(None,target,c).ratio() for c in candidates)


def describe_match(url, references, meta):
    state = []
    for ref in references:
        author,publication=re.split(r'\(\d{4}\)\.',ref['text'],maxsplit=1)
        author=author.strip()
        first=normalize(author.split(',',1)[0])
        titles = meta.get('title','')
        author_compare = meta.get('families',[]) or meta.get('authors',[])
        author_known=bool(author_compare)
        first_ok = any(' '.join(first) in ' '.join(normalize(a)) for a in author_compare[:1]) if author_known else None
        if meta.get('pdf_text'):
            doc=' '.join(normalize(meta['pdf_text']))
            pieces=re.split(r'\.\s+',publication)
            major=normalize(pieces[0])
            score=float(' '.join(major) in doc) if len(major) >= 3 else 0.0
        else:
            score=similarity(titles,publication) if titles else 0.0
        state.append(dict(file=ref['file'],score=round(score,4),first_author_agrees=first_ok))
    best=max(r['score'] for r in state)
    clean=best>=.85 and all(r['first_author_agrees'] is not False for r in state)
    classification='matched' if clean else ('unavailable' if meta.get('metadata')=='unavailable' else 'review')
    meta={k:v for k,v in meta.items() if k!='pdf_text'}
    return dict(url=url,classification=classification,metadata=meta,comparisons=state,
                printed=list(dict.fromkeys(r['text'] for r in references)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--pdf-text-executable',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sources={k:args.sources/f'{k}.md' for k in ('manuscript','supplement')}
    urls=read_references(sources)
    keys=list(urls)
    with ThreadPoolExecutor(max_workers=4) as executor:
        fetched=list(executor.map(lambda url:retrieve(url,args.pdf_text_executable),keys))
    record=[describe_match(url,urls[url],result) for url,result in zip(keys,fetched)]
    counts={flag:sum(x['classification']==flag for x in record) for flag in ('matched','review','unavailable')}
    payload=dict(source_sha256={k:sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
                 unique_links=len(record),counts=counts,details=record)
    args.out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(counts,flush=True)
    for item in record:
        if item['classification']!='matched':
            print(item['classification'],item['url'],item['metadata'].get('title',item['metadata'].get('reason','')))


if __name__=='__main__':
    main()
