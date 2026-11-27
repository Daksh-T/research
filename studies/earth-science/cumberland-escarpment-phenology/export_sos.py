import os
"""Export the SOS climatology raster to a GEE asset (heavy graph -> static asset),
then everything downstream samples the cheap asset. Run once; polls to completion."""
import ee, time, sys
import eslib as L

L.init()
dom = L.domain()
ASSET = os.environ.get('EE_SOS_ASSET', f"projects/{os.environ.get('EE_PROJECT', '')}/assets/escarpment_sos")

# wipe any partial asset
try:
    ee.data.deleteAsset(ASSET); print('deleted existing', ASSET)
except Exception:
    pass

sos = L.build_sos(dom).toFloat()
task = ee.batch.Export.image.toAsset(
    image=sos, description='escarpment_sos', assetId=ASSET,
    region=dom.bounds(), scale=20, maxPixels=1e9)
task.start()
print('started export task', task.id)
while True:
    st = task.status()
    s = st['state']
    print('  state:', s)
    if s in ('COMPLETED', 'FAILED', 'CANCELLED'):
        if s != 'COMPLETED':
            print('ERROR:', st.get('error_message')); sys.exit(1)
        break
    time.sleep(20)
print('DONE ->', ASSET)
