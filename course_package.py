source_items=[('Tenstorrent hardware documentation','https://docs.tenstorrent.com/index.html','Architecture families, hardware products, and legacy documentation.'),('Tensix ISA manual',ISA+'README.md','Wormhole B0 and Blackhole A0 architecture details.'),('Official TT-Lang tour',DOC+'tour/index.html','Operations, blocks, execution nodes, and dataflow buffers.'),('Official TT-Lang documentation',DOC,'Language, tutorials, compiler options and simulator.'),('TT-Lang specification',UP+'docs/sphinx/specs/TTLangSpecification.md','Shape units, operation forms, queue ownership, block math, pipes, support matrix.'),('Compiler options',UP+'docs/sphinx/reference/compiler-options.md','Exact flags, precision caveats, environment variables, and pass behavior.'),('Single-node matmul',UP+'examples/single_node_matmul.py','The pure matmul baseline used for the course adaptation.'),('Multi-tile matmul',UP+'examples/matmul-tutorial/step_2_single_node_multitile_block.py','Block shapes and fused epilogue.'),('Full-grid matmul',UP+'examples/matmul-tutorial/step_4_multinode_grid_full.py','Complete-block assignment and matching guards in each kernel.'),('Local-reuse matmul',UP+'examples/metal_examples/multinode_reuse_matmul/ttlang/multinode_reuse_matmul.py','A richer local reuse schedule.'),('2D multicast matmul',UP+'examples/metal_examples/2d_mcast_matmul/ttlang/2d_mcast_matmul.py','PipeNet and source/destination callbacks.'),('Metalium Lab1',LAB+'lab1/lab1.html','Architecture and single-core dataflow.'),('Metalium Lab2',LAB+'lab2/lab2.html','Multicore work distribution and reuse.'),('Metalium Lab3',LAB+'lab3/lab3.html','NoC multicast, semaphores, readiness, and completion.'),('Course platform',OJ,'Execution and submission; assigned runner contract remains authoritative.')]

# Regenerate the publication directory to remove superseded teaching artifacts.
for old in DL.iterdir():
    if old.is_file(): old.unlink()
for f in ASSETS.iterdir():
    if f.is_file(): shutil.copy2(f,DL/f.name)
for name in ['matmul_to_compute.mlir','matmul_block_lowering.mlir']:
    shutil.copy2(ROOT/'research/ttlang'/name,DL/name)
shutil.copy2(ROOT/'research/ttlang/LICENSE',DL/'LICENSE-tt-lang.txt')
shutil.copy2(ROOT/'research/ttlang/NOTICE',DL/'NOTICE-tt-lang.txt')
shutil.copy2(ROOT/'research/blackhole_instruction_macros_excerpt.h',DL/'blackhole_instruction_macros_excerpt.h')
trace_text='CS540 MATMUL WALKTHROUGH — ILLUSTRATIVE PLACEHOLDERS\n\nNo compilation or assembly capture was performed. Only the Python excerpt is\nactual course source; every lowered representation is an authored teaching sketch.\nSource excerpt adapted from Tenstorrent TT-Lang; Apache-2.0 (see bundled license).\n\nKernel: matmul.py; A[64,128], B[128,96], out[64,96]; BF16; grid(1,1).\nOne-tile blocks, queue depth2; selected output C_tile[1,2].\n\n'
for stage in TRACE_STAGES:
    trace_prose=html.unescape(re.sub('<[^>]+>','',stage['description'].replace('</p>','\n\n'))).strip()
    trace_prose='\n\n'.join(textwrap.fill(p.strip(),width=88) for p in trace_prose.split('\n\n') if p.strip())
    trace_text+=stage['title']+'\n'+stage['file']+'\n\n'+stage['code']+'\n\n'+trace_prose+'\n\n'
(DL/'matmul-trace-placeholders.txt').write_text(trace_text)
manifest={'status':'placeholder_template_not_a_compilation_record','compiled':False,'executed':False,'source':{'file':'matmul.py','sha256':hashlib.sha256((ASSETS/'matmul.py').read_bytes()).hexdigest()},'intended_specialization':{'a_shape':[64,128],'b_shape':[128,96],'out_shape':[64,96],'input_output_dtype':'BF16','layout':'tiled','tile_shape':[32,32],'grid':[1,1],'dfb_block_shape':[1,1],'dfb_depth':2},'actual_capture':{'run_id':None,'ttlang_revision':None,'ttmetal_revision':None,'sfpi_revision':None,'target_architecture':None,'board_and_stepping':None,'compiler_options':None,'compute_configuration':None,'artifacts':{'initial_ttl':None,'per_pass_checkpoints':None,'ttkernel':None,'final_mlir':None,'generated_cpp_by_role':None,'elf_by_processor':None,'disassembly_by_processor':None}},'instructions':'Replace null fields only with values from one real capture. Record paths and SHA-256 for each artifact. Do not substitute these illustrative sketches for generated outputs.'}
(DL/'matmul-capture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
prov=json.loads((ROOT/'research/ttlang/provenance.json').read_text())
prov['files']=[x for x in prov['files'] if x['file'] in ['matmul_to_compute.mlir','matmul_block_lowering.mlir']]
prov['scope']='Compiler fixtures are exact upstream inputs. Teaching kernels are adaptations. No device compilation or timing performed during authoring.'
prov['teaching_files']=[{'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'modified':True,'scope':'Course teaching adaptation; source review complete; device tests pending'} for f in ASSETS.glob('*.py')]
prov['illustrative_matmul_trace']={'file':'matmul-trace-placeholders.txt','status':'Authored placeholders; compiler capture pending','source_file':'matmul.py','source_sha256':manifest['source']['sha256'],'manifest':'matmul-capture-manifest.json'}
prov['blackhole_macros']={'url':'https://github.com/tenstorrent/tt-metal/blob/7c3dd90a675f0f00316786d01e4f60be42c3c9c0/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h','scope':'Selected inline-assembly source from the device library'}
prov['q7_mlir_excerpts']={'files':[{'file':name,'sha256':hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()} for name in ['matmul_pack_before.mlir','matmul_pack_after.mlir']],'status':'Manually transcribed teaching excerpts, not compiler captures','sources':[UP+'test/ttlang/Conversion/TTLToTTKernel/matmul_block_lowering.mlir',UP+'test/ttlang/Dialect/TTKernel/Transforms/combine_pack_tiles.mlir',UP+'lib/Dialect/TTKernel/Transforms/TTKernelCombinePackTiles.cpp']}
prov['references']=[{'title':t,'url':u} for t,u,d in source_items]
(DL/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
cases={'schema_version':5,'language':'TT-Lang','status':'Proposed assignment; enable and validate the course runner before submissions.','seed':540,'dtype':'FP32 exercise inputs, arithmetic, and outputs; row lengths are int64','matmul':{'dtype':'BF16 tutorial example','cases':[[64,128,96],[1024,1024,1024],[32,4096,128],[4096,128,4096],[96,160,224],[65,97,33]],'baseline_alignment':32,'blocked_alignment':'M multiple of32*BM, N of32*BN, K of32*BK','tails':'Harness zero-pads and crops; include preprocessing in end-to-end timing'},'vector':{'lengths':VECTOR_LENGTHS,'input_shapes':{'a':'[N]','b':'[N]'},'input_dtype':'torch.float32','output_shape':'[N]','output_dtype':'torch.float32','reference':'pytorch_baselines.py:vector_add_reference'},'ragged':{'input_shape':'[R,L]','input_dtype':'torch.float32','lengths_shape':'[R]','lengths_dtype':'torch.int64','lengths_range':'0 <= lengths[r] <= L; R >= 1 and L >= 1','sum_output':'torch.float32 [R]; empty rows return zero','softmax_output':'torch.float32 [R,L]; entries outside valid prefixes and empty rows are zero','sum_domain':'finite values','softmax_domain':'all entries finite in [-80,80]','reference':'pytorch_baselines.py','cases':RAGGED_SHAPES,'row_length_generation':RAGGED_LENGTH_RULE}}
(DL/'test-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
template='# CS540 · TT-Lang lab report\n\nName:\nGMU ID:\n\nSubmit the completed report as a separate PDF.\n\n'
for section_ in REPORT_SECTIONS:
    template+=f'## {section_["title"]} — {section_["weight"]} points\n\n'
    template+='\n'.join('- '+item for item in section_['items'])+'\n\n[Your response]\n\n'
template+='## 4. IR analysis — 40 points\n\n'+IR_REPORT_INTRO+'\n\n'
for label,url in IR_REPORT_SOURCES:
    if url.startswith('downloads/'):url=url.removeprefix('downloads/')
    elif url.startswith('#'):url='https://cs540-tenstorrent-lab.jokeren.chatgpt.site/'+url
    template+=f'- [{label}]({url})\n'
template+='\n'
for i,q in enumerate(QUESTIONS,1):template+=f'### Q{i}. {q["title"]} — {q["points"]} points\n\n{q["prompt"]}\n\n{q.get("details_markdown", "")}\n\n[Your analysis]\n\n'
(DL/'report-template.md').write_text(template)
results='exercise,case,initial_submission_id,final_submission_id,initial_time,final_time,time_unit,timing_scope,speedup,correctness,notes\n'
for exercise in ['vector-add','ragged-sum','ragged-softmax']:
    for case in range(1,11):results+=f'{exercise},{case},,,,,,,,,\n'
(DL/'results-template.csv').write_text(results)
(DL/'README.txt').write_text('''CS540 TT-Lang Lab — Tutorial materials

Read the website or reading-guide.html for the full tutorial.
Kernels import ttl, receive preallocated tiled device tensors, write output
arguments, and return None. They are not standalone host launchers and cannot
be called with ordinary NumPy/Torch tensors. Use the instructor's TT-Lang runner.

matmul.py: single core, scalar dimensions divisible by32.
matmul_larger.py: larger blocks on one core; dimensions divisible by64.
matmul_blocked.py: blocked/full-grid adaptation; default dimensions divisible
by64. Pad inputs with zeros and crop outputs in the runner.
pytorch_baselines.py: FP32 reference functions for all three exercise definitions.
Exercise inputs and outputs use FP32; row lengths use int64.
vector_add.py: optional low-level example using packed physical storage.
ragged_baselines.py: optional low-level examples using a mask tensor and padded
storage; these use a different interface from the logical PyTorch references.
Their full-row implementation supports padded width<=256.
Long ragged cases require a streaming extension with bounded buffers.

Teaching kernels have been source-reviewed and Python syntax-checked. They await
compilation and execution on course hardware. Pin and validate the course image.
The course OJ still needs a compatible TT-Lang runner and problem setup.

CPU semantics: python check_references.py (requires NumPy).
References check the mathematical contract; they are not accelerator kernels.
matmul-trace-placeholders.txt: annotated teaching sketches; compiler capture pending.
matmul-capture-manifest.json: source/shape contract with capture fields left null.
Compiler .mlir files are authentic upstream test inputs with expected patterns. The .h file contains inline assembly from
the Blackhole library. See provenance.json and licenses.
''')

def readable_spacing(text):
    return re.sub(r'\b([A-Z]?[a-z]{2,})(?=\d)',lambda m:m[1] if m[1] in {'tf','fp','bf'} else m[1]+' ',text)
def polish_html(text):
    pieces=re.split(r'(<[^>]+>)',text)
    in_code=False
    for i,piece in enumerate(pieces):
        if piece.startswith('<code') or piece.startswith('<pre'):in_code=True
        elif piece in ['</code>','</pre>']:in_code=False
        elif not piece.startswith('<') and not in_code:pieces[i]=readable_spacing(piece)
    return ''.join(pieces)
for page in LESSONS:
    headings=re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',page['html'])
    outline='<details class="chapter-outline"><summary>In this chapter</summary><nav aria-label="Chapter sections">'+''.join('<a href="#'+page['id']+'/'+anchor+'">'+title+'</a>' for anchor,title in headings)+'</nav></details>'
    if page['id']!='overview':
        page['html']=page['html'].replace('<h2 ',outline+'<h2 ',1)
    page['html']=polish_html(page['html'])
for q in QUESTIONS:q['prompt']=readable_spacing(q['prompt'])
(DIST/'lessons.js').write_text('window.LESSONS = '+json.dumps(LESSONS)+';\nwindow.QUESTIONS = '+json.dumps(QUESTIONS)+';\nwindow.REPORT_SECTIONS = '+json.dumps(REPORT_SECTIONS)+';\nwindow.IR_REPORT_INTRO = '+json.dumps(IR_REPORT_INTRO)+';\nwindow.IR_REPORT_SOURCES = '+json.dumps(IR_REPORT_SOURCES)+';\nwindow.PIPELINE = '+json.dumps(PIPELINE)+';\n')
reading='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS540 · TT-Lang reading guide</title><style>body{max-width:950px;margin:40px auto;padding:0 24px;font:16px/1.75 Arial;color:#25282b}h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.25}h3{line-height:1.4}h1{margin-top:65px}h2{margin-top:40px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:20px;font-size:12px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:10px;border:1px solid #ddd;text-align:left}a{color:#244f78}.callout{padding:16px;background:#f6f7f8}.copy-code,.interactive,.hero,.actions,.upload-box,input,textarea,.report-meta,.save-note{display:none}.code-top{font-size:11px;color:#55616c}.path-row{display:block;margin:20px 0}.site-divider{border:0;border-top:1px solid #ccc;margin:50px 0}details>div{display:block}button{display:none}.formula{padding:16px;background:#f6f7f8;font:14px/1.8 monospace;white-space:pre-wrap}.arch-diagram{padding:20px;background:#f7f8fa}.arch-diagram span,.arch-diagram small{display:block}.arch-stages{display:flex;gap:15px;margin:14px 0}.reuse-diagram{display:none}.hardware-figure{margin:24px 0}.figure-scroll{overflow-x:auto}.hardware-figure svg{display:block;width:100%;min-width:620px;height:auto}.hardware-figure figcaption{font-size:13px;color:#53616b;margin-top:12px}</style></head><body><h1>CS540 · TT-Lang Lab</h1><p>Complete reading companion. Interactive diagrams are on the website.</p>'
for p in LESSONS:
    body=re.sub(r'id="([^"]+)"',lambda m:'id="'+p['id']+'-'+m[1]+'"',p['html'])
    body=body.replace('<details>','<details open>')
    if p['id']=='report':
        questions_html=''.join('<h3>Q'+str(i)+'. '+q['title']+' — '+str(q['points'])+' points</h3><p>'+q['prompt']+'</p>'+q.get('details','') for i,q in enumerate(QUESTIONS,1))
        body=body.replace('<div id="report-report-questions"></div>',questions_html)
    reading+='<hr class="site-divider"><section id="'+p['id']+'">'+body
    reading+='</section>'
reading+='</body></html>'
reading=reading.replace('href="downloads/','href="')
reading=re.sub(r'href="#([^"/]+)/([^"/]+)"',r'href="#\1-\2"',reading)
(DL/'reading-guide.html').write_text(reading)
with ZipFile(DL/'student-kit.zip','w',ZIP_DEFLATED) as z:
    for f in sorted(DL.iterdir()):
        if f.is_file() and f.name!='student-kit.zip':z.write(f,f.name)
index=(DIST/'index.html').read_text()
for asset in ['style.css','lessons.js','app.js']:
    digest=hashlib.sha256((DIST/asset).read_bytes()).hexdigest()[:12]
    index=re.sub(r'((?:src|href)=")'+re.escape(asset)+r'(?:\?[^\"]*)?"',lambda m:m[1]+asset+'?v='+digest+'"',index)
(DIST/'index.html').write_text(index)
print(f'Built {len(LESSONS)} TT-Lang lessons, {len(QUESTIONS)} questions, and {len(PIPELINE)} representation stages.')
