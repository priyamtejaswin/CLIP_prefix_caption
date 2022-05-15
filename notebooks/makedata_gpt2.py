import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--features', type=str, required=True)
parser.add_argument('--gpt2out', type=str, required=True)
parser.add_argument('--outpath', type=str, required=True)

args = parser.parse_args()

assert os.path.exists(args.outpath) is False, "--outpath already exists. Please give a new one."
assert args.outpath.endswith('.npy'), "--outpath does not specify a npy file. Exiting."

with open(args.gpt2out) as fp:
    talks = {x['question_id']: x['wgpt2'].lower() for x in json.loads(fp.read())}
    
towrite = []
data = np.load(args.features, allow_pickle=True, encoding='latin1')
print(data[0])
# print(data[1])
towrite.append(data[0])
failed = 0
for d in data[1:]:
    qid = d['question_id']
    if qid in talks:
        size = len(d['question'])+1
        d['question'] = talks[qid]
        d['question_tokens'].append('<gpt2>')
        d['question_tokens'].extend(talks[qid][size:-1].split() if talks[qid][-1] == '.' else talks[qid][size:].split())
    else:
        failed += 1
        
    towrite.append(d)
    
with open(args.outpath, 'wb') as fp:
    np.save(fp, towrite)
    
print("Done. Written to", args.outpath)
print("Could not find gpt2 selftalks for", failed)
