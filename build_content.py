"""Build the source-backed TT-Lang course and its downloadable materials."""
import html, json, re, shutil, hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parent
DIST=ROOT/'dist'
DL=DIST/'downloads'
ASSETS=ROOT/'teaching_assets'
PIN='a47484458bd964a6b92c9e4698aeb922585f08cc'
UP=f'https://github.com/tenstorrent/tt-lang/blob/{PIN}/'
DOC='https://docs.tenstorrent.com/tt-lang/'
LAB='https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/'
OJ='http://98.93.3.136:28888/'
def code(text,title='Python · TT-Lang'):
    return '<div class="code-block"><div class="code-top"><span>'+html.escape(title)+'</span><button class="copy-code">Copy</button></div><pre><code>'+html.escape(text.strip())+'</code></pre></div>'
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def note(title,body,kind=''):
    return '<div class="callout '+kind+'"><strong>'+title+'</strong>'+body+'</div>'
def header(kicker,title,lead,meta):
    return '<p class="eyebrow">'+kicker+'</p><h1>'+title+'</h1><p class="lead">'+lead+'</p>'
def hint(title,body):
    return '<details><summary>'+title+'</summary><div>'+body+'</div></details>'
def source(label,url):
    return '<a href="'+url+'" target="_blank" rel="noopener">'+label+' ↗</a>'
def refs(*items):
    return '<p class="footer-source">Read the source: '+' · '.join(source(label,url) for label,url in items)+'</p>'
def section(id_,title,body):
    return '<h2 id="'+id_+'">'+title+'</h2>'+body
def kernel(name):
    raw=(ASSETS/name).read_text()
    return raw[raw.index('import ttl'):]
LESSONS=[]
def lesson(id_,short,summary,body):
    LESSONS.append(dict(id=id_,short=short,summary=summary,html=body))
for part in ['course_figures.py','course_architecture.py','course_language.py','course_matmul_trace.py','course_matmul.py','course_compiler.py','course_exercises.py','course_package.py']:
    exec((ROOT/part).read_text())
