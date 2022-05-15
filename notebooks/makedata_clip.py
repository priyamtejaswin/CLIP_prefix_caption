import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--features', type=str, required=True)
parser.add_argument('--clipout', type=str, required=True)
parser.add_argument('--outpath', type=str, required=True)

args = parser.parse_args()

assert os.path.exists(args.outpath) is False, "--outpath already exists. Please give a new one."
assert args.outpath.endswith('.npy'), "--outpath does not specify a npy file. Exiting."

with open(args.clipout) as fp:
    talks = {x['img_id']: x['clipcap'].lower() for x in json.loads(fp.read())}
    
towrite = []
data = np.load(args.features, allow_pickle=True, encoding='latin1')
print(data[0])
print(data[1])

towrite.append(data[0])
failed = 0
for d in data[1:]:
    imid = d['image_id']
    if imid in talks:
        d['question'] = d['question'] + ' <clip> ' + talks[imid]
        d['question_tokens'].append('<clip>')
        d['question_tokens'].extend(talks[imid][:-1].split() if talks[imid][-1] == '.' else talks[imid].split())
    else:
        failed += 1
        
    towrite.append(d)
    
with open(args.outpath, 'wb') as fp:
    np.save(fp, towrite)
    
print("Done. Written to", args.outpath)
print("Could not find clip selftalks for", failed)
