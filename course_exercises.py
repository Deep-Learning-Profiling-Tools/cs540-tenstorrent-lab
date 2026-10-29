import ast
import pprint

def pytorch_baseline(name):
    text=(ASSETS/'pytorch_baselines.py').read_text()
    fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return code('import torch\n\n'+ast.get_source_segment(text,fn),'Python · PyTorch')

EXERCISE_EXAMPLES=json.loads((ASSETS/'exercise_examples.json').read_text())

def example_array(value):
    if value and isinstance(value[0],list):
        return '[\n    '+',\n    '.join(pprint.pformat(row,width=60) for row in value)+'\n]'
    return pprint.pformat(value,width=60)

def input_output_examples(key):
    blocks=[]
    for i,example in enumerate(EXERCISE_EXAMPLES[key],1):
        inputs='\n'.join(name+' = '+example_array(value) for name,value in example['inputs'].items())
        output='out = '+example_array(example['out'])
        blocks.append(code('# Input\n'+inputs+'\n\n# Output\n'+output,f'Example {i}'))
    return ''.join(blocks)

VECTOR_LENGTHS=[1,31,33,1023,1024,1025,65537,1048577,16777216,16777217]

vector=header('05 / Practice 1','Vector addition','Compute the elementwise sum of two vectors.',[])
vector+=section('contract','Problem definition',
    table(['Tensor','Data type','Shape'],[['<code>a</code>','<code>torch.float32</code>','<code>[N]</code>'],['<code>b</code>','<code>torch.float32</code>','<code>[N]</code>'],['Output','<code>torch.float32</code>','<code>[N]</code>']])
    +'<p>All input values are finite. Return <code>out[i] = a[i] + b[i]</code> for every index.</p>')
vector+=section('examples','Examples',input_output_examples('vector-add'))
vector+=section('baseline','PyTorch baseline',pytorch_baseline('vector_add_reference')+'<p>Inputs, addition, and output use FP32.</p>')
vector+=section('cases','Test shapes',table(['Case','N'],[[str(i),f'{n:,}'] for i,n in enumerate(VECTOR_LENGTHS,1)]))
lesson('vector-add','Vector addition','Vector addition: inputs, PyTorch baseline, and test shapes.',vector)

ragged_inputs=table(['Tensor','Data type','Shape'],[['<code>x</code>','<code>torch.float32</code>','<code>[R, L]</code>'],['<code>lengths</code>','<code>torch.int64</code>','<code>[R]</code>']])+'<p><code>R ≥ 1</code> and <code>L ≥ 1</code>. Row <code>r</code> contains <code>lengths[r]</code> valid values, with <code>0 ≤ lengths[r] ≤ L</code>. Only <code>x[r, :lengths[r]]</code> contributes to the result. Entries beyond that prefix may be nonzero. All inputs are on the same device.</p>'
RAGGED_SHAPES=[
    {'R':256,'L':65},
    {'R':8192,'L':96},
    {'R':1024,'L':542},
    {'R':1024,'L':8192},
    {'R':32,'L':16384},
    {'R':7,'L':17},
    {'R':33,'L':1},
    {'R':127,'L':37},
    {'R':4097,'L':257},
    {'R':1,'L':8193},
]
RAGGED_LENGTH_RULE='For each case, row lengths are randomly generated integers from 0 through L.'
families=table(['Case','R','L'],[
    [str(i),f"{case['R']:,}",f"{case['L']:,}"]
    for i,case in enumerate(RAGGED_SHAPES,1)])+'<p>'+RAGGED_LENGTH_RULE+'</p>'

sum_page=header('06 / Practice 2','Ragged sum','Sum the valid prefix of each row.',[])
sum_page+=section('contract','Problem definition',ragged_inputs
    +'<p>All values in <code>x</code> are finite. Return a <code>torch.float32</code> tensor of shape <code>[R]</code>, where <code>out[r] = sum(x[r, :lengths[r]])</code>. An empty row returns zero.</p>'
    +table(['Row length','Output'],[['<code>0</code>','The valid interval <code>[0, 0)</code> is empty; return <code>0</code>.'],['<code>1</code>','Return <code>x[r, 0]</code>.'],['<code>L</code>','Sum all <code>L</code> values. The unused tail <code>[L, L)</code> is empty.']]))
sum_page+=section('examples','Examples',input_output_examples('ragged-sum')+'<p>Only the prefix specified by each row length contributes; a row of length zero returns zero.</p>')
sum_page+=section('baseline','PyTorch baseline',pytorch_baseline('ragged_sum_reference')+'<p>Inputs, accumulation, and output use FP32.</p>')
sum_page+=section('cases','Test shapes',families)
lesson('ragged-sum','Ragged sum','Row sums: inputs, PyTorch baseline, and test shapes.',sum_page)

soft=header('07 / Practice 3','Ragged softmax','Apply softmax to the valid prefix of each row.',[])
soft+=section('contract','Problem definition',ragged_inputs
    +'<p>All values in <code>x</code> are finite and lie in <code>[-80, 80]</code>. Return a <code>torch.float32</code> tensor of shape <code>[R, L]</code>. For each nonempty row, <code>out[r, :lengths[r]] = softmax(x[r, :lengths[r]])</code>. All remaining entries, including every entry of an empty row, must be zero.</p>'
    +table(['Row length','Output'],[['<code>0</code>','The valid interval <code>[0, 0)</code> is empty; return an all-zero row of length <code>L</code>.'],['<code>1</code>','Return <code>1</code> at index zero and <code>0</code> in the remaining entries.'],['<code>L</code>','Apply softmax to all <code>L</code> values. The tail <code>[L, L)</code> has no entries to fill.']])
    +'<p>Intervals are half-open: <code>[a, b)</code> includes <code>a</code> and excludes <code>b</code>. An interval with equal endpoints contains no elements. The output still has shape <code>[R, L]</code>.</p>')
soft+=section('examples','Examples',input_output_examples('ragged-softmax')+'<p>Entries outside the valid prefix are zero. The second output shows FP32 probabilities rounded to eight decimal places.</p>')
soft+=section('baseline','PyTorch baseline',pytorch_baseline('ragged_softmax_reference')+'<p>Inputs, softmax computation, and output use FP32.</p>')
soft+=section('cases','Test shapes',families)
lesson('ragged-softmax','Ragged softmax','Row softmax: inputs, PyTorch baseline, and test shapes.',soft)

QUESTIONS=[
 dict(title='Host and device compilation paths',points=5,prompt='Provide a table of the compilation and execution stages, in sequence, from a TT-Lang Python operation call to execution on the accelerator. Use four columns: stage, execution hardware (CPU or accelerator), input, and output.'),
 dict(title='Tensor types and dimensions',points=5,prompt='In matmul_to_compute.mlir, inspect matmul_2x4_4x3. Write the full types, tile dimensions, and number of tiles of the two input tensors and the result.'),
 dict(title='Iteration space and indexing maps',points=5,prompt='For matmul_2x4_4x3 in matmul_to_compute.mlir, explain the three affine indexing maps for A, B, and the output.'),
 dict(title='Circular-buffer capacities',points=5,prompt='In matmul_2x4_4x3 in matmul_block_lowering.mlir, identify the three ttl.bind_cb declarations and their block_count values. Calculate each buffer’s capacity in tiles and BF16 payload bytes.'),
 dict(title='Register and packing sequence',points=5,prompt='Explain the CHECK expectations for matmul_1x1_bf16 in matmul_block_lowering.mlir. What are we checking?'),
 dict(title='Reduction loop after lowering',points=5,prompt='In the CHECK expectations for matmul_2x4_4x3 in matmul_block_lowering.mlir, give the scf.for lower and upper bounds.'),
 dict(title='Pack-combining transformation',points=5,prompt='Compare the supplied matmul_pack_before.mlir and matmul_pack_after.mlir files. How many packing operations are present before and after the transformation? You do not need to run a compiler.'),
 dict(title='Circular-buffer lifetimes',points=5,prompt='In matmul_add_accumulator in matmul_block_lowering.mlir, identify the input and output circular buffers. Explain the roles of ttl.cb_wait, ttl.cb_reserve, ttl.cb_push, and ttl.cb_pop.')]

QUESTIONS[6]['details']='<p><a href="downloads/matmul_pack_before.mlir">Before transformation (.mlir)</a> · <a href="downloads/matmul_pack_after.mlir">After transformation (.mlir)</a></p>'
QUESTIONS[6]['details_markdown']='[Before transformation](matmul_pack_before.mlir) · [After transformation](matmul_pack_after.mlir)'

QUESTIONS[7]['details']='<p><a href="downloads/matmul_block_lowering.mlir">Read the supplied MLIR file</a></p>'
QUESTIONS[7]['details_markdown']='[Read the supplied MLIR file](matmul_block_lowering.mlir)'

REPORT_SECTIONS=[
 dict(id='timing',key='timing',title='1. Time reporting',weight=20,items=[
     'Use figures or tables to report execution times for all 10 cases of each exercise. Label the exercise, case, implementation, time unit, and submission ID.',
     'Compare your initial correct implementation with your final implementation. Report speedup as initial time / final time, using the same timing scope. Label missing or failed results.',
     'Use the measurements returned by the submission platform and state what its reported time includes.'
 ],placeholder='Add your timing tables or figure references, submission IDs, and a short description of the timing scope.'),
 dict(id='implementation',key='implementation',title='2. Implementation sketch',weight=20,items=[
     'For each exercise, give concise pseudocode or a diagram of your final implementation.',
     'Describe work assignment, data movement, and computation. Explain how the implementation handles irregular sizes, empty rows, and singleton rows where applicable.',
 ],placeholder='Vector addition: ...\nRagged sum: ...\nRagged softmax: ...'),
 dict(id='optimization',key='optimization',title='3. Optimization and profiling metric analysis',weight=20,items=[
     'For each exercise, explain at least one change from the initial implementation, the cost it targets, and its effect on execution time across cases.',
     'Analyze the profiling metrics returned by the platform. Give each metric’s name, unit, scope, and before/after values, and explain how it supports or challenges your hypothesis.',
     'Discuss a regression or a case with little improvement. If a relevant metric is unavailable, identify it and explain what it would help determine; separate measured results from hypotheses.'
 ],placeholder='For each exercise: change → expected effect → timing and profiling evidence → interpretation.')]

report=header('08 / Analysis & report','Lab report','Report your implementations and results for the three exercises, then answer eight compiler-analysis questions.',[])
report+=table(['Section','Points'],[[s['title'],str(s['weight'])+' points'] for s in REPORT_SECTIONS]+[['4. IR analysis','40 points · 8 questions × 5 points']])
report+='<p>Submit a separate PDF report with your name and GMU ID.</p>'
for s in REPORT_SECTIONS:
    body='<ul>'+''.join('<li>'+html.escape(item)+'</li>' for item in s['items'])+'</ul>'
    report+=section(s['id'],s['title']+' — '+str(s['weight'])+' points',body)
IR_REPORT_INTRO='Answer all eight questions. Each is worth 5 points out of 100. Use the supplied MLIR files and linked sources; cite file names and line numbers. No compiler or accelerator access is required.'
IR_REPORT_SOURCES=[('matmul_to_compute.mlir','downloads/matmul_to_compute.mlir'),('matmul_block_lowering.mlir','downloads/matmul_block_lowering.mlir'),('matmul_pack_before.mlir','downloads/matmul_pack_before.mlir'),('matmul_pack_after.mlir','downloads/matmul_pack_after.mlir')]
IR_REPORT_ATTACHMENTS=table(['MLIR file','Questions','Contents'],[
    ['<a href="downloads/matmul_to_compute.mlir" download><code>matmul_to_compute.mlir</code></a>','Q2–Q3','Tensor types, matrix dimensions, and iteration-space checks.'],
    ['<a href="downloads/matmul_block_lowering.mlir" download><code>matmul_block_lowering.mlir</code></a>','Q4–Q6, Q8','Circular-buffer capacities and lifetimes, register operations, and reduction loops.'],
    ['<a href="downloads/matmul_pack_before.mlir"><code>matmul_pack_before.mlir</code></a>','Q7','Packing region before combining.'],
    ['<a href="downloads/matmul_pack_after.mlir"><code>matmul_pack_after.mlir</code></a>','Q7','Packing region after combining.']])

report+=section('questions','4. IR analysis — 40 points',
    '<p>'+IR_REPORT_INTRO+'</p>'
    +'<h3>MLIR files for the questions</h3>'+IR_REPORT_ATTACHMENTS+'<p>The first two files are upstream compiler tests with <code>CHECK</code> expectations. The Q7 files are supplied teaching excerpts of the packing region, transcribed from those expectations and the pack-combining implementation. For Q1, use the <a href="#compiler">IR &amp; assembly chapter</a>.</p>'
    +'<div id="report-questions"></div>')
lesson('report','Analysis & report','Time reporting, implementation sketches, optimization and profiling, and eight IR questions.',report)
