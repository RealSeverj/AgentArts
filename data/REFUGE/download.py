from pathlib import Path
import requests,json,time,hashlib,os,subprocess,concurrent.futures as cf,threading
ROOT=Path(__file__).resolve().parent
SIZE=1365186829
CHUNK=16*1024*1024
archive=ROOT/'refuge2_kaggle_mirror.zip';part=archive.with_suffix('.zip.part')
statefile=ROOT/'download_ranges.json'
url='https://www.kaggle.com/api/v1/datasets/download/victorlemosml/refuge2'
def log(d):
 print(json.dumps({'dataset':'REFUGE mirror',**d}),flush=True)
 (ROOT/'progress.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
def save(d):
 p=statefile.with_suffix('.json.tmp');p.write_text(json.dumps(d,indent=2),encoding='utf-8');os.replace(p,statefile)
LOCAL=threading.local()
def get_range(start,end):
 if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
 for attempt in range(6):
  try:
   with LOCAL.session.get(url,params={'datasetVersionNumber':1},headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'},stream=True,timeout=(25,90)) as r:
    r.raise_for_status()
    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
     raise RuntimeError(f"Unexpected range: {r.status_code}, {r.headers.get('Content-Range')}")
    count=0
    with part.open('r+b') as h:
     h.seek(start)
     for b in r.iter_content(1024*1024):
      if not b:continue
      count+=len(b)
      if count>end-start+1:raise RuntimeError('Response too long')
      h.write(b)
     h.flush()
    if count!=end-start+1:raise RuntimeError('Incomplete range')
   return start,end
  except Exception as e:
   log({'event':'range_retry','start':start,'attempt':attempt+1,'error':str(e)[:130]})
   if attempt==5:raise
   time.sleep(min(2**attempt,15))
if not archive.exists():
 if statefile.exists():
  state=json.loads(statefile.read_text(encoding='utf-8'))
  if state['size']!=SIZE or state['chunk']!=CHUNK:raise RuntimeError('State mismatch')
 else:
  prefix=part.stat().st_size if part.exists() else 0
  state={'size':SIZE,'chunk':CHUNK,'prefix_bytes':prefix,'completed':[]};save(state)
 with part.open('r+b' if part.exists() else 'wb') as h:h.truncate(SIZE)
 done={tuple(x) for x in state['completed']}
 jobs=[(s,min(s+CHUNK,SIZE)-1) for s in range(state['prefix_bytes'],SIZE,CHUNK) if (s,min(s+CHUNK,SIZE)-1) not in done]
 received=state['prefix_bytes']+sum(e-s+1 for s,e in done)
 with cf.ThreadPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(get_range,*j):j for j in jobs};last=0
  while pending:
   finished,_=cf.wait(pending,timeout=10,return_when=cf.FIRST_COMPLETED)
   for fut in finished:
    pending.pop(fut);s,e=fut.result();state['completed'].append([s,e]);save(state);received+=e-s+1
   if time.monotonic()-last>=15 or not pending:
    log({'event':'download','bytes':received,'total':SIZE,'percent':round(received/SIZE*100,2),'connections':2});last=time.monotonic()
 os.replace(part,archive)
sha=hashlib.sha256()
with archive.open('rb') as h:
 for b in iter(lambda:h.read(8*1024*1024),b''):sha.update(b)
if archive.stat().st_size!=SIZE:raise RuntimeError('Archive size mismatch')
(ROOT/'download_report.json').write_text(json.dumps({'dataset':'REFUGE public mirror labelled refuge2','source':'https://www.kaggle.com/datasets/victorlemosml/refuge2','original_source':'https://refuge.grand-challenge.org/','archive_bytes':SIZE,'sha256_local':sha.hexdigest(),'publisher_archive_hash_available':False,'mirror_license':'Unknown','original_license_scope_not_established':True},indent=2),encoding='utf-8')
log({'event':'downloaded','bytes':SIZE})
out=ROOT/'extracted';out.mkdir(exist_ok=True)
r=subprocess.run(['C:/Program Files/7-Zip/7z.exe','x','-y','-o'+str(out),str(archive)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
(ROOT/'extraction.log').write_text(r.stdout,encoding='utf-8')
if r.returncode:raise RuntimeError('7-Zip extraction failed')
log({'event':'extracted','files':sum(p.is_file() for p in out.rglob('*'))})
