import zipfile
with zipfile.ZipFile('/workspace/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('data2')