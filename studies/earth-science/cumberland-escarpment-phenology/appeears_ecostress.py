"""
Pull ECOSTRESS 70 m night LST over the Domain via the AppEEARS API, for a true
Domain-scale cold-air-pool map (the 1 km MODIS check in script 07 can't resolve the gorges).

Usage:
  python appeears_ecostress.py submit    # log in, submit area task, save task id
  python appeears_ecostress.py fetch     # check status; download GeoTIFFs when 'done'

Credentials: reads EARTHDATA_USER / EARTHDATA_PASS from ../.env (NASA Earthdata Login,
the same account as AppEEARS). Nothing is printed.

We request ECO_L2T_LSTE.002 (LST + QC) over the Domain bbox for all winters (Dec-Mar)
2018-2024. Day/night split and cloud-air analysis happen later in 09_ecostress_coldpool.py.
"""
import sys, os, json, re, time
import requests
import eslib as L

HERE = L.HERE
OUT = L.A('data/ecostress')
os.makedirs(OUT, exist_ok=True)
API = 'https://appeears.earthdatacloud.nasa.gov/api'
TASKFILE = os.path.join(OUT, 'task.json')

def creds():
    env = {}
    for line in open(L.A('../.env')):
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.strip().split('=', 1); env[k] = v
    u, p = env.get('EARTHDATA_USER'), env.get('EARTHDATA_PASS')
    if not u or not p:
        sys.exit('Missing EARTHDATA_USER / EARTHDATA_PASS in ../.env')
    return u, p

def login():
    u, p = creds()
    r = requests.post(API + '/login', auth=(u, p))
    r.raise_for_status()
    return r.json()['token']

def domain_bbox_geojson():
    import ee
    L.init()
    b = L.domain().bounds().buffer(500).bounds().getInfo()['coordinates']
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": b}}]}

def submit():
    token = login()
    geo = domain_bbox_geojson()
    task = {
        "task_type": "area", "task_name": "sewanee_ecostress_nightLST",
        "params": {
            # recurring ranges cannot cross New Year, so winter = (Jan-Mar) + (Dec)
            "dates": [{"startDate": "01-01", "endDate": "03-31", "recurring": True, "yearRange": [2019, 2024]},
                      {"startDate": "12-01", "endDate": "12-31", "recurring": True, "yearRange": [2018, 2023]}],
            "layers": [{"product": "ECO_L2T_LSTE.002", "layer": "LST"},
                       {"product": "ECO_L2T_LSTE.002", "layer": "QC"},
                       {"product": "ECO_L2T_LSTE.002", "layer": "cloud"}],
            "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
            "geo": geo,
        },
    }
    r = requests.post(API + '/task', json=task, headers={'Authorization': 'Bearer ' + token})
    r.raise_for_status()
    tid = r.json()['task_id']
    json.dump({'task_id': tid, 'token_hint': 'expired-relogin'}, open(TASKFILE, 'w'))
    print('submitted task', tid, '-> saved', TASKFILE)
    print('AppEEARS area tasks queue server-side; run "fetch" later (minutes to hours).')

def fetch():
    if not os.path.exists(TASKFILE):
        sys.exit('no task.json — run submit first')
    tid = json.load(open(TASKFILE))['task_id']
    token = login()
    H = {'Authorization': 'Bearer ' + token}
    st = requests.get(API + '/task/' + tid, headers=H).json()
    state = st.get('status', st.get('state'))
    print('task', tid, 'status:', state)
    if state != 'done':
        prog = st.get('progress', {})
        print('  progress:', prog.get('summary', prog))
        return
    bundle = requests.get(API + '/bundle/' + tid, headers=H).json()
    files = bundle['files']
    tifs = [f for f in files if f['file_name'].lower().endswith('.tif')]
    print('bundle ready: %d files (%d GeoTIFFs)' % (len(files), len(tifs)))
    got = 0
    for f in tifs:
        fid, fname = f['file_id'], os.path.basename(f['file_name'])
        dest = os.path.join(OUT, fname)
        if os.path.exists(dest):
            got += 1; continue
        with requests.get(API + '/bundle/%s/%s' % (tid, fid), headers=H, stream=True) as resp:
            resp.raise_for_status()
            with open(dest, 'wb') as fh:
                for chunk in resp.iter_content(8192):
                    fh.write(chunk)
        got += 1
        if got % 20 == 0:
            print('  downloaded %d/%d' % (got, len(tifs)))
    print('downloaded %d GeoTIFFs -> %s' % (got, OUT))

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fetch'
    {'submit': submit, 'fetch': fetch}.get(cmd, lambda: sys.exit('submit|fetch'))()
