"""Build the separate editable highlights file from its reviewed source."""
from pathlib import Path
import argparse
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
items=[line[2:] for line in a.source.read_text().splitlines() if line.startswith('- ')]
assert 3<=len(items)<=5 and all(len(x)<=85 for x in items)
d=Document();s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)
s.top_margin=s.bottom_margin=Inches(.85);s.left_margin=s.right_margin=Inches(1)
for name,size in [('Normal',11),('Title',18),('List Bullet',11)]:
 st=d.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
 st.paragraph_format.line_spacing=1.15;st.paragraph_format.space_after=Pt(10)
 pr=st.element.get_or_add_pPr()
 for border in list(pr.findall(qn('w:pBdr'))):pr.remove(border)
d.add_paragraph('Highlights','Title')
for item in items:d.add_paragraph(item,'List Bullet')
d.core_properties.author='';d.core_properties.last_modified_by='';d.core_properties.title='Highlights'
d.save(a.out)
print({'items':len(items),'characters':[len(x) for x in items]})
