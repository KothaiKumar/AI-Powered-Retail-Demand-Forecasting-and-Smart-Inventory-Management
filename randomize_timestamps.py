import os, fnmatch, random
from datetime import datetime, timedelta
import argparse, time

def find_images(root):
    out=[]
    for r,_,files in os.walk(root):
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            for f in fnmatch.filter(files, ext):
                out.append(os.path.join(r,f))
    return sorted(out)

def random_dates(n,start,end):
    s=start.timestamp()
    e=end.timestamp()
    return [random.uniform(s,e) for _ in range(n)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="data/slash-dataset-480p")
    p.add_argument("--start","-s",default=None)
    p.add_argument("--end","-e",default=None)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--preview",action="store_true")
    args=p.parse_args()
    if args.start is None:
        sd=datetime.now()-timedelta(days=90)
    else:
        sd=datetime.fromisoformat(args.start)
    if args.end is None:
        ed=datetime.now()-timedelta(days=1)
    else:
        ed=datetime.fromisoformat(args.end)
    random.seed(args.seed)
    imgs=find_images(args.root)
    if not imgs:
        print("NO_IMAGES_FOUND")
        return
    timestamps=random_dates(len(imgs),sd,ed)
    if args.preview:
        for i,pth in enumerate(imgs[:10]):
            print(pth, datetime.fromtimestamp(timestamps[i]).isoformat())
        return
    for pth,ts in zip(imgs,timestamps):
        atime=ts
        mtime=ts
        try:
            os.utime(pth,(atime,mtime))
        except Exception as ex:
            print("ERROR",pth,ex)
    print("DONE",len(imgs))

if __name__=="__main__":
    main()
