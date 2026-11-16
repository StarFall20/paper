"""Bind the repository's current status to the delivered manuscript sources.

Reads existing Word XML and current integration/visual-review records. It does
not render pages or claim a new visual review. Historical evidence stays dated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as E
import zipfile


def verify(repo, out):
    package = repo / 'submission/finite_choice_20261110'
    status = json.loads((repo / 'reports/public_conjoint_audit_20261109/submission_evidence_status.json').read_text())
    pointer = json.loads((package / 'reports/current_manuscript.json').read_text())
    integrated = json.loads((package / pointer['integration']).read_text())
    reviewed = json.loads((package / pointer['visual_qa']).read_text())
    current = status['current_verified_release']
    assert status['goal_complete'] is False
    result = {}
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                  'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    assert reviewed['status'] == integrated['status'] == 'complete'
    for name in ['paper', 'main_text']:
        md = package / 'manuscript' / (name + '.md')
        docx = package / 'manuscript' / (name + '.docx')
        md_hash = hashlib.sha256(md.read_bytes()).hexdigest()
        assert md_hash == current[name + '_md_sha256']
        assert md_hash == integrated['new_manuscript_sha256' if name == 'paper' else 'main_text_sha256']
        with zipfile.ZipFile(docx) as z:
            root = E.fromstring(z.read('word/document.xml'))
        objects = len(root.findall('.//m:oMath', namespaces))
        tables = len(root.findall('.//w:tbl', namespaces))
        displays = len(re.findall(r'\$\$.*?\$\$', md.read_text(), re.S))
        assert (objects, tables, displays) == tuple(current[name][k] for k in ['native_math_objects', 'tables', 'source_display_blocks'])
        docx_hash = hashlib.sha256(docx.read_bytes()).hexdigest()
        assert docx_hash == current[name + '_docx_sha256']
        visual = reviewed['full' if name == 'paper' else 'main']
        assert (md_hash, docx_hash, objects, tables, displays) == tuple(
            visual[k] for k in ['source_sha256', 'docx_sha256', 'native_math_count',
                                'tables', 'display_equation_paragraphs'])
        assert current[name]['pages_from_existing_visual_review'] == visual['pages']
        assert set(visual['page_review_basis']) == {str(n) for n in range(1, visual['pages'] + 1)}
        assert set(visual['page_png_sha256']) == set(visual['page_review_basis'])
        result[name] = dict(native_math_objects=objects, tables=tables,
                            source_display_blocks=displays, md_sha256=md_hash,
                            docx_sha256=docx_hash)
    cumulative = status['new_completed_evidence']
    assert cumulative['native_math_objects'] == result['paper']['native_math_objects']
    assert cumulative['manuscript_tables'] == result['paper']['tables']
    assert cumulative['main_reading_Word_pages'] == current['main_text']['pages_from_existing_visual_review']
    assert current['paper']['pages_from_existing_visual_review'] == integrated['full_pages']
    assert current['main_text']['pages_from_existing_visual_review'] == integrated['main_pages']
    assert (package / pointer['visual_qa']).is_file()
    sources = list((package / 'analysis').glob('*.py'))
    assert len(sources) == current['package_analysis_sources'] == integrated['current_science_source_count']
    result.update(state='passed', package_analysis_sources=len(sources),
                  page_counts_source=pointer['visual_qa'], new_render_performed=False,
                  manuscript_changed=False, originality_or_submission_threshold_certified=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify(a.repo, a.out)
