import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img

def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()

def resize_worker(img_files, sizes, resample):
    lr_img = Image.open(img_files)
    lr_img = lr_img.convert('RGB')
    sr_img = resize_and_convert(lr_img, sizes[1], resample)
    #
    out = [lr_img, sr_img, sr_img]

    return img_files[0].name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, lr_subset):
    for lr in lr_subset:
        # while True:
        #     try:
        i, imgs = wctx.resize_fn(lr)
        lr_img, hr_img, sr_img = imgs
        if not wctx.lmdb_save:
            lr_img.save(
                '{}/lr/{}.png'.format(wctx.out_path, i.zfill(5)))
            hr_img.save(
                '{}/hr/{}.png'.format(wctx.out_path, i.zfill(5)))
            sr_img.save(
                '{}/sr/{}.png'.format(wctx.out_path, i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}'.format(i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}'.format(i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}'.format(i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))
            #     break
            # except Exception as err:
            #     print(err)
            #     print("restart")
            #     continue

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    lr_files = [p for p in Path('{}'.format(img_path)).glob(f'lr/*')] #[19522:]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr'.format(out_path), exist_ok=True)
        os.makedirs('{}/hr'.format(out_path), exist_ok=True)
        os.makedirs('{}/sr'.format(out_path), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    print(n_worker)
    print("ready to prepare")

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        lr_file_subsets = np.array_split(lr_files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, lr_file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(lr_files))
        while not all_threads_inactive(worker_threads):
            print("{}/{} images processed".format(wctx.value(), total_count))
            time.sleep(0.1)
    else:
        print("you have to set num_worker more than 1.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/celebahq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}'.format(args.out)
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)
