#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from tqdm import tqdm
import os
import requests
import skimage.io as io
import PIL.Image
import time


# In[ ]:


with open('/usr0/home/ptejaswi/Download/TextVQA_0.5.1_train.json') as fp:
    train = json.loads(fp.read())
    
print(train.keys())


# In[ ]:


def download_image(url, imid, extension):
    # Path is fixed!
    if extension == '':
        print("No extension found!", url, imid, extension)
        extension = 'jpg'
        
    savepath = os.path.join('/usr0/home/ptejaswi/Download/tvqa_images', imid+'.'+extension)
    
    img_data = requests.get(url).content
    with open(savepath, 'wb') as handler:
        handler.write(img_data)
    
    try:
        image = io.imread(savepath)
        pil_image = PIL.Image.fromarray(image)
    except ValueError:
        os.remove(savepath)
        return False
    
    return True


# In[ ]:


failed = 0
dev_qs = []
available = set([f.split('.')[0] for f in os.listdir('/usr0/home/ptejaswi/Download/tvqa_images/') if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])
for x in tqdm(train['data']):
    imid = x['image_id']
    if imid not in available:
        time.sleep(1)
        url = x['flickr_original_url']
        if download_image(url, x['image_id'], url.split('.')[-1]) is False:
            # Try again.
            url = x['flickr_300k_url']
            if download_image(url, x['image_id'], url.split('.')[-1]) is False:
                failed += 1
                
print("Failed:", failed)


# In[ ]:




