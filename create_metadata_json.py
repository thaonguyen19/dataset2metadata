import json
import hashlib
import os
import yaml
from pathlib import Path
import fsspec

parquet_dir = '/fsx/home-thaottn/dataset2metadata/100M_laion_coco_blip2_captions_temp_1.0/' #TODO: update this
job_dir = 'examples/jobs/' #TODO: update this

result_dict = {}
for job_file in os.listdir(job_dir):
    full_job_file = os.path.join(job_dir, job_file)
    yml = yaml.safe_load(Path(full_job_file).read_text())
    hashname = hashlib.md5(str(yml['input_tars']).encode()).hexdigest()
    result_dict[hashname] = {'parquet': f'{parquet_dir}/{hashname}.parquet', 'shards': [s.split(' ')[-2] for s in yml['input_tars']]}
print(result_dict)

fs, output_path = fsspec.core.url_to_fs(f'{parquet_dir}/100M_blip2_captions_metadata.json')
with fs.open(output_path, 'w') as f:
    json.dump(result_dict, f)
all_parquets = [result_dict[k]['parquet'] for k in result_dict]
print(len(all_parquets))
print(all_parquets[0])
