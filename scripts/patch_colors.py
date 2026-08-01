import re,sys
f = sys.argv[1]
with open(f,'r') as fh: c = fh.read()
lines = c.split('\n')
new = []
for l in lines:
    sk = l.strip().startswith(('PAGE_BG','HEADER','ACCENT','TEXT=','TEXT_MUTED','RED','GREEN','ORANGE','BORDER'))
    if sk: continue
    new.append(l)
with open(f,'w') as fh: fh.write('\n'.join(new))
print('Patched',f)