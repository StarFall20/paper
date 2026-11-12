"""Native OMML authoring for the verified manuscript, using a strict TeX subset."""
from pathlib import Path
import re, json, hashlib
from lxml import etree
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('manuscript/paper.md'));p.add_argument('--out',type=Path,default=Path('manuscript/paper.docx'));p.add_argument('--report',type=Path,default=Path('regenerated/docx_structure.json'));args=p.parse_args()
SOURCE=args.source
DEST=args.out
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
def el(tag,**attrs):
 e=etree.Element('{'+M+'}'+tag)
 for k,v in attrs.items():e.set('{'+M+'}'+k,str(v))
 return e
def seq(tag,children):
 e=el(tag)
 for c in children:e.append(c)
 return e
def run(text,plain=False):
 e=el('r');pr=el('rPr')
 if plain:
  pr.append(el('sty',val='p'));pr.append(el('nor'))
 e.append(pr);rpr=OxmlElement('w:rPr');font=OxmlElement('w:rFonts');font.set(qn('w:ascii'),'Cambria Math');font.set(qn('w:hAnsi'),'Cambria Math');rpr.append(font)
 size=OxmlElement('w:sz');size.set(qn('w:val'),'22');rpr.append(size);e.append(rpr)
 t=el('t');t.text=text;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');e.append(t);return e

SYMS={'partial':'∂','alpha':'α','nabla':'∇','lambda':'λ','tau':'τ','kappa':'κ','beta':'β','eta':'η','gamma':'γ','epsilon':'ε','sigma':'σ','mu':'μ','rho':'ρ','pi':'π','varphi':'φ','Phi':'Φ','Psi':'Ψ','Pi':'Π','Delta':'Δ','delta':'δ','ell':'ℓ','infty':'∞','geq':'≥','leq':'≤','in':'∈','cap':'∩','circ':'∘','times':'×','to':'→','mid':'|','sim':'∼','sum':'∑','prod':'∏'}
OPS={'inf','sup','min','max','log','ker','lim','sin','cos','exp'}
class Math:
 def __init__(self,s):self.s=s;self.i=0
 def spaces(self):
  while self.i<len(self.s) and self.s[self.i].isspace():self.i+=1
 def group_text(self):
  self.spaces()
  if self.i>=len(self.s) or self.s[self.i]!='{':raise ValueError('Expected text group '+self.s[self.i:])
  self.i+=1;start=self.i;depth=1
  while self.i<len(self.s) and depth:
   if self.s[self.i]=='{':depth+=1
   if self.s[self.i]=='}':depth-=1
   self.i+=1
  if depth:raise ValueError('Unclosed text group')
  return self.s[start:self.i-1]
 def atom(self):
  self.spaces()
  if self.i>=len(self.s):raise ValueError('Missing atom')
  c=self.s[self.i];self.i+=1
  if c=='{':return self.parse('}')
  if c=='\\':
   match=re.match('[A-Za-z]+',self.s[self.i:])
   if not match:
    if self.i>=len(self.s):raise ValueError('Trailing escape')
    c=self.s[self.i];self.i+=1
    return [run({'|':'‖',',':'\u2009',';':'\u2004','!':'',' ':' '}.get(c,c),True)]
   cmd=match.group();self.i+=len(cmd)
   if cmd in SYMS:return [run(SYMS[cmd])]
   if cmd in OPS:return [run(cmd,True)]
   if cmd in ['quad','qquad']:return [run('\u2003'*(1 if cmd=='quad' else 2),True)]
   if cmd in ['text','operatorname']:return [run(self.group_text(),True)]
   if cmd in ['mathbb','mathcal']:
    nodes=self.atom();text=''.join(''.join(n.itertext()) for n in nodes).strip()
    mp={'R':'ℝ','N':'ℕ','E':'𝔼'} if cmd=='mathbb' else {'Z':'𝒵','Q':'𝒬','C':'𝒞'}
    if text not in mp:raise ValueError('Unsupported alphabet '+text)
    return [run(mp[text])]
   if cmd in ['frac','tfrac']:
    parts=[]
    for tag in ['num','den']:
     self.spaces()
     if self.i>=len(self.s) or self.s[self.i]!='{':raise ValueError('Fraction arguments must be braced in this strict TeX subset')
     parts.append(seq(tag,self.atom()))
    return [seq('f',parts)]
   if cmd=='sqrt':
    pr=el('radPr');pr.append(el('degHide',val='1'))
    return [seq('rad',[pr,el('deg'),seq('e',self.atom())])]
   if cmd=='widehat':
    pr=el('accPr');pr.append(el('chr',val='̂'));return [seq('acc',[pr,seq('e',self.atom())])]
   if cmd=='hphantom':
    pr=el('phantPr');pr.append(el('show',val='0'));pr.append(el('zeroAsc',val='1'));pr.append(el('zeroDesc',val='1'));return [seq('phant',[pr,seq('e',self.atom())])]
   raise ValueError('Unsupported TeX command '+cmd)
  if c in ['}','^','_']:raise ValueError('Unexpected '+c)
  if c.isdigit():
   while self.i<len(self.s) and (self.s[self.i].isdigit() or self.s[self.i]=='.' and self.i+1<len(self.s) and self.s[self.i+1].isdigit()):
    c+=self.s[self.i];self.i+=1
  return [run(c,plain=c.isdigit() or c in '=+-/[](),.:;|<>')]
 def parse(self,stop=None):
  out=[]
  while self.i<len(self.s):
   self.spaces()
   if self.i==len(self.s):break
   if stop and self.s[self.i]==stop:self.i+=1;return out
   base=self.atom();sub=sup=None
   while self.i<len(self.s):
    old=self.i;self.spaces()
    if self.i==len(self.s) or self.s[self.i] not in '_^':self.i=old;break
    op=self.s[self.i];self.i+=1;arg=self.atom()
    if op=='_':
     if sub is not None:raise ValueError('Double subscript')
     sub=arg
    else:
     if sup is not None:raise ValueError('Double superscript')
     sup=arg
   if sub is not None and sup is not None:out.append(seq('sSubSup',[seq('e',base),seq('sub',sub),seq('sup',sup)]))
   elif sub is not None:out.append(seq('sSub',[seq('e',base),seq('sub',sub)]))
   elif sup is not None:out.append(seq('sSup',[seq('e',base),seq('sup',sup)]))
   else:out.extend(base)
  if stop:raise ValueError('Unclosed math group')
  return out

def math(p,tex):p._p.append(seq('oMath',Math(tex).parse()))
def rich(p,s):
 for i,part in enumerate(re.split(r'\$([^$]+)\$',s)):
  if i%2:math(p,part)
  elif part:p.add_run(part)

def cell_border(cell):
 pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
 for side in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
 pr.append(b);m=OxmlElement('w:tcMar')
 for side in ['top','bottom','left','right']:
  e=OxmlElement('w:'+side);e.set(qn('w:w'),'85');e.set(qn('w:type'),'dxa');m.append(e)
 pr.append(m)

def build():
 text=SOURCE.read_text()
 for i in [1,2,3]:text=text.replace('{{TABLE'+str(i)+'}}',(SOURCE.parent/f'table{i}.md').read_text().strip())
 if '{{TABLE' in text:raise ValueError('Unresolved table')
 SOURCE.write_text(text)
 d=Document();s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11);s.top_margin=Inches(.8);s.bottom_margin=Inches(.8);s.left_margin=Inches(1);s.right_margin=Inches(1)
 normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(11);normal.font.color.rgb=RGBColor(0,0,0);normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(7)
 for name,size in [('Title',19),('Heading 1',14),('Heading 2',12)]:
  st=d.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0);st.font.bold=name!='Title';st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(7)
  st.paragraph_format.keep_with_next=True
  pr=st.element.get_or_add_pPr()
  for border in list(pr.findall(qn('w:pBdr'))):pr.remove(border)
 footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
 field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
 d.core_properties.title='When prediction assisted logit estimates exist';d.core_properties.author='';d.core_properties.subject='Binary and multinomial choice estimation with finite auxiliary data'
 lines=text.splitlines();i=0;mathcount=0
 while i<len(lines):
  line=lines[i].strip();i+=1
  if not line:continue
  if line.startswith('# '):d.add_paragraph(line[2:],style='Title');continue
  if line.startswith('## '):
   p=d.add_paragraph(line[3:],style='Heading 1')
   if line.startswith('## Appendix A') or line=='## References':p.paragraph_format.page_break_before=True
   continue
  if line.startswith('### '):d.add_paragraph(line[4:],style='Heading 2');continue
  if line.startswith('$$'):
   assert line.endswith('$$'),line
   p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(7);p.paragraph_format.keep_together=True
   math(p,line[2:-2]);mathcount+=1;continue
  if line.startswith('!['):
   match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line);assert match
   p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
   p.add_run().add_picture(str(SOURCE.parent/match[2]),width=Inches(6.5));continue
  if line.startswith('|'):
   tablelines=[line]
   while i<len(lines) and lines[i].strip().startswith('|'):tablelines.append(lines[i].strip());i+=1
   rows=[[x.strip() for x in row.strip('|').split('|')] for j,row in enumerate(tablelines) if j!=1]
   table=d.add_table(rows=0,cols=len(rows[0]));table.autofit=False;table.alignment=WD_TABLE_ALIGNMENT.CENTER
   widths=[.85,.65,1.,1.3,1.4,1.3] if rows[0][0]=='Total labels' else [.75,.55,.75,1.05,1.05,1.05] if rows[0][0]=='Alternatives' else {4:[1.35,1.7,.8,2.65],5:[.7,.85,1.2,1.2,1.65],6:[1.2,1.65,.7,.65,.8,1.5]}[len(rows[0])]
   widths=[x*6.5/sum(widths) for x in widths]
   for c,w in zip(table.columns,widths):c.width=Inches(w)
   for j,values in enumerate(rows):
    cells=table.add_row().cells
    for k,(cell,value) in enumerate(zip(cells,values)):
     cell.width=Inches(widths[k]);cell_border(cell);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
     p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
     if len(rows[0])==5:p.paragraph_format.keep_with_next=j<len(rows)-1
     p.alignment=WD_ALIGN_PARAGRAPH.LEFT if k in [0,1] else WD_ALIGN_PARAGRAPH.CENTER
     rich(p,value)
     for r in p.runs:r.font.size=Pt(9.5);r.bold=j==0
     if j==0:
      sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EDEDED');cell._tc.get_or_add_tcPr().append(sh)
    trpr=table.rows[j]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
    if j==0:trpr.append(OxmlElement('w:tblHeader'))
   d.add_paragraph().paragraph_format.space_after=Pt(2);continue
  p=d.add_paragraph()
  lead=re.match(r'^((?:Proposition|Theorem|Corollary) [0-9]+\.)',line)
  if lead:p.add_run(lead[0]).bold=True;rich(p,line[len(lead[0]):])
  else:rich(p,line)
  if re.match(r'^Figure [0-9]+ ',line):
   p.paragraph_format.space_after=Pt(10);p.paragraph_format.keep_together=True
  if re.match(r'^Table (?:[0-9]+|[A-Z][0-9]+) ',line):p.paragraph_format.keep_with_next=True
 DEST.parent.mkdir(parents=True,exist_ok=True);d.save(DEST)
 from zipfile import ZipFile
 with ZipFile(DEST) as z:
  xml=etree.fromstring(z.read('word/document.xml'));native=len(xml.findall('.//{'+M+'}oMath'))
  assert native>100
 report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'native_math_count':native,'display_equation_paragraphs':mathcount,'rasterized_equations':0,'raw_tex_commands_left_in_text':bool(re.search(r'\\(?:frac|lambda|sigma)', ''.join(xml.itertext()))),'tables':len(d.tables)}
 assert not report['raw_tex_commands_left_in_text']
 args.report.parent.mkdir(parents=True,exist_ok=True)
 (args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':build()
