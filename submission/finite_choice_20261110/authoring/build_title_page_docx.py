"""Build a separate author page using only confirmed author facts."""
from pathlib import Path
import argparse
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn

p = argparse.ArgumentParser()
p.add_argument('--facts', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
facts = json.loads(a.facts.read_text())
d = Document()
s = d.sections[0]
s.page_width, s.page_height = Inches(8.5), Inches(11)
s.top_margin = s.bottom_margin = Inches(1)
s.left_margin = s.right_margin = Inches(1)
for name, size in [('Normal', 12), ('Title', 18), ('Heading 1', 12)]:
    st = d.styles[name]
    st.font.name = 'Times New Roman'
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.space_after = Pt(12)
    pr = st.element.get_or_add_pPr()
    for border in list(pr.findall(qn('w:pBdr'))):
        pr.remove(border)
d.add_paragraph(facts['manuscript_title'], 'Title')
for author in facts['provided_authors']:
    d.add_paragraph(author['name_as_supplied'])
    d.add_paragraph(author['affiliation_english'])
    if facts['corresponding_author_confirmed'] and author.get('email'):
        d.add_paragraph('Correspondence: ' + author['email'])
if facts['funding']['confirmed']:
    d.add_paragraph('Funding', 'Heading 1')
    d.add_paragraph(facts['funding']['statement'])
d.core_properties.title = facts['manuscript_title']
d.core_properties.author = '; '.join(x['name_as_supplied'] for x in facts['provided_authors'])
d.core_properties.last_modified_by = ''
d.save(a.out)
print(json.dumps({'provided_authors': len(facts['provided_authors']), 'out': str(a.out)}))
