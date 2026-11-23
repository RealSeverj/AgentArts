from pathlib import Path
import requests,json,time,hashlib,os,subprocess
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'source_metadata.json').read_text(encoding='utf-8'))
reports=[]
for f in sorted(meta['files'],key=lambda f:f['key']):
 archive=ROOT/f['key'];part=archive.with_suffix(archive.suffix+'.part')
 size=f['size'];expected=f['checksum'].split(':')[1]
 def log(d):
  print(json.dumps({'dataset':'IDRiD','file':f['key'],**d}),flush=True)
  (ROOT/'progress.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
 if not archive.exists():
  for attempt in range(8):
   start=part.stat().st_size if part.exists() else 0
   if start==size:break
   try:
    url=f['links']['self']+'?download=1&fresh='+str(time.time_ns())
    with requests.get(url,headers={'Range':f'bytes={start}-','Cache-Control':'no-cache'},stream=True,timeout=(25,90)) as r:
     r.raise_for_status()
     if not (r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{size-1}/{size}' or r.status_code==200 and start==0):
      raise RuntimeError(f"Unexpected range: {r.status_code}, {r.headers.get('Content-Range')}")
     last=time.monotonic();done=start
     with part.open('ab') as out:
      for block in r.iter_content(1024*1024):
       if not block:continue
       done+=len(block)
       if done>size:raise RuntimeError('Response too long')
       out.write(block)
       if time.monotonic()-last>=15:
        out.flush();log({'event':'download','bytes':done,'total':size,'percent':round(done/size*100,2)});last=time.monotonic()
    if part.stat().st_size!=size:raise RuntimeError('Incomplete response')
    break
   except Exception as e:
    log({'event':'retry','attempt':attempt+1,'error':str(e)[:160]})
    if attempt==7:raise
    time.sleep(min(2**attempt,15))
  check=part
 else:check=archive
 md5=hashlib.md5();sha=hashlib.sha256()
 with check.open('rb') as h:
  for b in iter(lambda:h.read(8*1024*1024),b''):md5.update(b);sha.update(b)
 if check.stat().st_size!=size or md5.hexdigest()!=expected:raise RuntimeError('Archive size or published MD5 mismatch')
 if check!=archive:os.replace(check,archive)
 report={'archive':archive.name,'bytes':size,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'mirror_published_md5_match':True}
 reports.append(report)
 (ROOT/'download_report.json').write_text(json.dumps({'dataset':'IDRiD','provenance':'Third-party Zenodo mirror; original creator Porwal et al., IEEE DataPort','source_record':'https://zenodo.org/records/17219542','original_source':'https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid','license':'CC BY 4.0','files':reports},ensure_ascii=False,indent=2),encoding='utf-8')
 log({'event':'download_verified','bytes':size})
 out=ROOT/'extracted';out.mkdir(exist_ok=True)
 exe=Path('C:/Program Files/7-Zip/7z.exe')
 res=subprocess.run([str(exe),'x','-y','-o'+str(out),str(archive)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
 (ROOT/(archive.stem+'.extraction.log')).write_text(res.stdout,encoding='utf-8')
 if res.returncode:raise RuntimeError('7-Zip extraction failed; see extraction log')
 log({'event':'extracted'})
