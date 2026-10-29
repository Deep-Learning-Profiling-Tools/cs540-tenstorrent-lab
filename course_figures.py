"""Original teaching diagrams, drawn from the references beside each figure."""
def figure_svg(name, height, body, caption):
    return f'<figure class="hardware-figure"><div class="figure-scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 {height}" role="img" aria-label="{html.escape(name,quote=True)}"><title>{html.escape(name)}</title><style>text{{font-family:Arial,sans-serif;fill:#293540;font-size:17px}}.small{{font-size:14px}}.label{{font-weight:600}}</style>{body}</svg></div><figcaption>{caption}</figcaption></figure>'
def ft(x,y,s,cls='',anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{html.escape(s)}</text>'
def fb(x,y,w,h,lines,fill='#edf2f6'):
    if isinstance(lines,str):lines=[lines]
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#8a9ba8"/>'+''.join(ft(x+w/2,y+h/2+(i-(len(lines)-1)/2)*22+6,s,'label' if i==0 else 'small') for i,s in enumerate(lines))
def fa(x1,y1,x2,y2,color='#45657e',both=False):
    import math
    angle=math.atan2(y2-y1,x2-x1)
    def tip(x,y,a):
        pts=[(x,y),(x-9*math.cos(a)+4*math.sin(a),y-9*math.sin(a)-4*math.cos(a)),(x-9*math.cos(a)-4*math.sin(a),y-9*math.sin(a)+4*math.cos(a))]
        return '<polygon points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)+f'" fill="{color}"/>'
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="2" fill="none"/>'+tip(x2,y2,angle)+(tip(x1,y1,angle+math.pi) if both else '')

b=fb(65,12,360,55,['Host CPU + system memory'])+ft(218,96,'PCIe','small')
b+='<rect x="35" y="107" width="730" height="403" rx="5" fill="#fafbfc" stroke="#8295a3"/>'+ft(55,135,'Accelerator chip','label','start')
b+=fb(75,153,210,55,'PCIe endpoint')+fb(515,153,210,55,'DRAM controllers')+fa(620,153,620,68,both=True)+fb(530,18,235,50,'GDDR6',fill='#f4efe6')
# Transfer crosses the chip boundary; draw it after the enclosing rectangle.
b+=fa(180,67,180,153,both=True)
b+=fa(180,208,180,234,both=True)+fa(620,208,620,234,both=True)+fb(75,234,650,45,'Network on chip (NoC): links between endpoints','#e9f0ed')
for r in range(3):
    for c in range(4):
        x=75+c*166;y=307+r*60
        b+=fb(x,y,151,47,[f'Tensix {4*r+c}','private L1 + compute'])
        if r==0:b+=fa(x+75,279,x+75,307,both=True)
        else:b+=fa(x+75,y-13,x+75,y,both=True)
        if c>0:b+=fa(x-15,y+24,x,y+24,both=True)
b+=ft(400,491,'12 representative cores shown; each connects to both NoCs.','small')
CHIP_FIGURE=figure_svg('Host, external GDDR6, and a multicore Tenstorrent chip',525,b,'Figure 1. Device organization. The core array is schematic; board core counts appear in the specifications table.')

def control_arrow(x1,y1,x2,y2):
    return fa(x1,y1,x2,y2,'#96602f').replace('stroke-width="2"', 'stroke-width="2" stroke-dasharray="5 4"')

b=fb(30,15,280,60,['B + NC','data-movement processors'],'#fcf2e7')+control_arrow(310,45,390,45)
b+=fb(390,15,380,60,['Two NoC interfaces','transfers to DRAM and other cores'],'#e9f0ed')+fa(580,75,580,120,both=True)
b+=fb(30,120,740,60,['L1 scratchpad SRAM','Wormhole B0: 1,464 KiB · Blackhole A0: 1,536 KiB'])
# Controllers sit immediately above the resource group they drive.
b+=fb(30,220,140,45,'T0 · unpack','#fcf2e7')+control_arrow(100,265,100,305)
b+=fb(386,220,140,45,'T1 · math','#fcf2e7')+control_arrow(456,265,456,289)
b+='<rect x="366" y="289" width="180" height="208" rx="3" fill="#fafbfc" stroke="#96602f" stroke-dasharray="5 4"/>'
b+=fb(30,305,140,58,['2 unpackers','format conversion'])+fa(170,334,208,334)+fb(208,305,140,58,['SrcA / SrcB','input registers'])+fa(348,334,386,334)
b+=fb(386,305,140,58,['Matrix engine','FPU'])+fa(526,334,564,334)+fb(564,299,206,70,['Destination registers','DST · 32 KiB'],'#f4efe6')
b+=fb(376,420,160,60,['Vector engine','SFPU: 32 × 32-bit'])+fa(536,450,667,450,both=True)+fa(667,369,667,450,both=True)
# L1-to-unpack path and pack-to-L1 return run along the left margin.
b+='<path d="M55,180 V195 H18 V334 H22" fill="none" stroke="#45657e" stroke-width="2"/>'+fa(22,334,30,334)
b+=fb(30,410,140,45,'T2 · pack','#fcf2e7')+control_arrow(100,455,100,495)+fb(30,495,140,58,['4 packers','output format'])
b+='<path d="M667,450 V524 H178" fill="none" stroke="#45657e" stroke-width="2"/>'+fa(178,524,170,524)
b+='<path d="M30,524 H6 V150 H22" fill="none" stroke="#45657e" stroke-width="2"/>'+fa(22,150,30,150)
b+=fa(65,593,120,593)+ft(130,598,'Data movement','small','start')+control_arrow(405,593,460,593)+ft(470,598,'Instructions / control','small','start')
CORE_FIGURE=figure_svg('Control and data paths: B and NC drive NoC transfers; T0 drives unpack; T1 drives FPU and SFPU; T2 drives pack',620,b,'Figure 2. Solid blue arrows carry data; dashed brown arrows show the usual control assignment. The T1 group includes both math engines. Each B, NC, T0, T1, and T2 label names a Baby RISC-V processor.')

b=fb(170,15,460,55,['Host system memory','Capacity depends on the host'])+fa(400,70,400,108,both=True)+ft(468,96,'PCIe','small')
b+=fb(100,108,600,65,['Device GDDR6','n150: 12 GB · p100a: 28 GB · p150a: 32 GB'],'#f4efe6')+fa(400,173,400,214,both=True)+ft(489,209,'NoC transfers','small')
for i in range(3):
    x=25+i*260
    b+=fb(x,214,230,62,[f'Core {i}: L1 SRAM','BH: 1,536 KiB per core'])
    if i!=1:b+=fa(400,193,x+115,193)+fa(x+115,193,x+115,214)
b+=fa(400,276,400,328,both=True)+ft(498,308,'unpack / pack','small')+fb(220,328,360,70,['Core 1: destination registers','DST: 32 KiB total'],'#f4efe6')
b+=ft(400,431,'Other cores have their own L1 and destination registers.','small')
MEMORY_FIGURE=figure_svg('Memory locations from host DRAM to per-core destination registers',450,b,'Figure 3. Tensor storage at three scales: board DRAM, core-local SRAM, and compute registers. Transfers between these locations are explicit.')

b=''
for panel,reverse in enumerate([False,True]):
    ox=30+panel*400
    b+=ft(ox+170,30,f'NoC {panel}','label')
    for r in range(3):
        for c in range(3):
            x=ox+60+c*105;y=95+r*85
            b+=f'<circle cx="{x}" cy="{y}" r="13" fill="#edf2f6" stroke="#8295a3"/>'
            if c<2:
                a=(x+15,y,x+90,y);b+=fa(*(a[2:]+a[:2] if reverse else a))
            if r<2:
                a=(x,y+15,x,y+70);b+=fa(*(a[2:]+a[:2] if reverse else a))
    # Representative wraparound links: top row and left column.
    b+=f'<path d="M{ox+270},82 V57 H{ox+60} V80" fill="none" stroke="#45657e" stroke-width="2"/>'
    b+=fa(ox+60,69,ox+60,80) if not reverse else fa(ox+270,69,ox+270,80)
    b+=f'<path d="M{ox+47},265 H{ox+24} V95 H{ox+45}" fill="none" stroke="#45657e" stroke-width="2"/>'
    b+=fa(ox+34,95,ox+45,95) if not reverse else fa(ox+34,265,ox+45,265)
    b+=ft(ox+170,306,'Edges wrap around','small')+ft(ox+170,332,'Each circle = router + endpoint','small')
b+=ft(400,379,'Same endpoints · two physically separate networks · opposite directions','label')
NOC_FIGURE=figure_svg('Two separate three-by-three schematic torus networks with opposite directions',400,b,'Figure 4. Simplified NoC topology. Each endpoint can serve a Tensix core, a DRAM controller, or another device resource. One horizontal and one vertical wraparound link are drawn per network; every row and column wraps.')

b=ft(185,27,'Logical 32 × 32 tile','label')+ft(590,27,'BF16 bytes in DRAM or L1','label')
for i,(r,c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
    b+=fb(40+c*145,52+r*100,145,100,[f'Face {i}','16 × 16 values'],['#e7eef5','#edf2e7','#f4ede4','#eeeaf4'][i])
for i in range(4):
    b+=fb(425,52+i*50,330,50,[f'Face {i}: bytes {i*512}–{(i+1)*512-1}'],['#e7eef5','#edf2e7','#f4ede4','#eeeaf4'][i])
b+=fa(340,152,411,152)+ft(400,290,'4 faces × 256 values × 2 bytes = 2,048 bytes','label')
TILE_FIGURE=figure_svg('A tile divided into four faces and its linear BF16 byte order',315,b,'Figure 5. Standard tiled storage for one BF16 tile. Each face occupies 512 consecutive bytes; rows within a face are contiguous.')

b=ft(400,28,'32 KiB destination register storage, split into two 16 KiB sections','label')
for r in range(2):
    y=65+r*100
    b+=ft(75,y+31,f'Step {r+1}','label')
    for c in range(2):
        role='Math writes' if r==c else 'Pack reads'
        b+=fb(140+c*315,y,300,64,[f'Section {c} · {role}','4 FP32 tiles or 8 BF16 tiles'],'#edf2f6' if r==c else '#f4efe6')
b+=ft(400,271,'Handoff and release let the two sections exchange roles.','small')
DST_FIGURE=figure_svg('Double buffering of destination registers across math and pack',290,b,'Figure 6. Double buffering lets math and pack operate on different sections. Full synchronization makes the full 32 KiB available to one phase at a time.')
