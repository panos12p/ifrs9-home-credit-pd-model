import glob, os

for p in sorted(glob.glob(r"E:\ifrs9-home-credit-pd-model\data\raw\*.csv")):
    print('=== ' + os.path.basename(p))
    try:
        with open(p, encoding='latin1') as f:
            print(f.readline().rstrip('\n'))
            print(f.readline().rstrip('\n'))
    except Exception as e:
        print('ERROR:', e)
