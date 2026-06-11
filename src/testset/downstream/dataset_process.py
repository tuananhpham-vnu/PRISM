import pandas as pd
import json

def clean_name(path_or_id: str):
    import os
    name = path_or_id.split("/")[-1]        
    name = name.split(":")[0]            
    return os.path.splitext(name)[0]     

# Đường dẫn file
file_path = [
    # 'gsm8k.parquet', 
    # 'arc-c.parquet', 
    # 'arc-e.parquet',
    'piqa.parquet']
for path in file_path:
    df = pd.read_parquet(path)
    output_json = f"{clean_name(path)}.json"
    df.to_json(output_json, orient='records', indent=2, force_ascii=False)
    print(f"✅ Đã lưu thành: {output_json}")
    print(f"\n📊 Total records: {len(df)}")