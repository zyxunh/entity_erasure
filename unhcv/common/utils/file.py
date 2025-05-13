import glob
import os
import os.path as osp
import time

import PIL.Image
import numpy as np
import shutil
import pickle
import json
import lmdb
import yaml
import io
import cv2

from .custom_logging import get_logger

try:
    import bson
except:
    bson = None
try:
    import msgpack
except:
    msgpack = None
from base64 import b64decode
import PIL.Image as Image
# import unhcv.datasets.common_datasets as common_datasets
# from unhcv.datasets.common_datasets import LmdbBackend, KvBackend, Dataset
Image.MAX_IMAGE_PIXELS = 1000000000
HOME_ROOT = os.environ["HOME"]
logger = get_logger(__name__)

def walk_all_files_with_suffix(dir,
                               suffixs=('.jpg', '.png', '.jpeg'),
                               sort=True):
    if os.path.isdir(dir) == False:
        return [dir]
    if isinstance(suffixs, str):
        suffixs = [suffixs]
    fnames = glob.glob(osp.join(dir, '**/*.*'), recursive=True)
    if suffixs is not None:
        if suffixs[0][0] != ".":
            fnames_ = []
            for fname in fnames:
                for suffix in suffixs:
                    if fname.lower().endswith(suffix.lower()):
                        fnames_.append(fname)
                        break
            fnames = fnames_
        else:
            fnames = [
                var for var in fnames if os.path.splitext(var)[1].lower() in suffixs
            ]
    if sort:
        fnames.sort()

    return fnames

def read_im(path, *args, **kwargs):
    if os.path.exists(path):
        return cv2.imread(path, *args, **kwargs)
    else:
        print(f'not exists {path}')
        return None

def write_im(path, im):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(im, Image.Image):
        return im.save(path)
    return cv2.imwrite(path, im)

def copy_file(path1, path2, keep_src=True):
    if not os.path.exists(os.path.dirname(path2)):
        os.makedirs(os.path.dirname(path2), exist_ok=True)
    if keep_src:
        return shutil.copyfile(path1, path2)
    else:
        return shutil.move(path1, path2)

def get_related_path(input_path, input_root, out_root, suffixs=None):
    if input_root.endswith('/'):
        input_root = input_root[:-1]
    if out_root.endswith('/'):
        out_root = out_root[:-1]
    assert input_path.startswith(input_root)
    path2 = out_root + input_path[len(input_root):]
    if suffixs is not None:
        path2 = os.path.splitext(path2)[0] + suffixs
    return path2

def replace_str(str_x, str_1, str_2, position=None):
    if position is None:
        assert str_1 in str_x
        str_x = str_x.replace(str_1, str_2)
    elif position == "l":
        assert str_x.startswith(str_1)
        str_x = str_2 + str_x[len(str_1):]
    elif position == "r":
        assert str_x.endswith(str_1)
        str_x = str_x[:-len(str_1)] + str_2
    else:
        raise ValueError
    return str_x

def attach_home_root(path: str):
    if path.startswith(HOME_ROOT):
        return path
    else:
        return osp.join(HOME_ROOT, path)

def find_path(path):
    if not os.path.exists(path):
        path = attach_home_root(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

def remove_dir(path, only_file=False, sleep_seconds=None):
    logger.info("to remove {}".format(path))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    if path.startswith('hdfs'):
        os.system("hdfs dfs -rm -r {}".format(path))
    elif osp.isfile(path):
        os.remove(path)
    elif only_file == False:
        shutil.rmtree(path, True)


def get_base_name(input_path, input_root, keep_suffix=True):
    if input_root is not None:
        assert input_path.startswith(input_root)
        input_path = input_path[len(input_root):]
    if input_path.startswith('/') or input_path.startswith("\\"):
        input_path = input_path[1:]
    if keep_suffix == False:
        input_path = os.path.splitext(input_path)[0]
    return input_path

def obj_dump(file: str, obj, mode_json='w', mode_img=None, return_buffer=False):
    if not return_buffer:
        dirname = os.path.dirname(file)
        if len(dirname) and os.path.exists(dirname) == False:
            os.makedirs(os.path.dirname(file), exist_ok=True)
    suffix = osp.splitext(file)[1].lower()
    if isinstance(obj, bytes):
        with open(file, "wb") as f:
            f.write(obj)
    elif suffix == '.json':
        with open(file, mode_json) as f:
            if mode_json == 'a':
                f.write('\n')
            json.dump(obj, f)
    elif suffix == '.pkl':
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
    elif suffix in ('.jpg', '.png', '.jpeg'):
        if mode_img is None:
            mode_img = suffix[1:]
        if mode_img == 'jpg':
            mode_img = 'JPEG'
        elif mode_img == 'png':
            mode_img = 'PNG'
        if isinstance(obj, np.ndarray):
            cv2.imwrite(file, obj)
        else:
            if return_buffer:
                file_fake = io.BytesIO()
                obj.save(file_fake, format=mode_img)
                return file_fake.getvalue()
            obj.save(file, format=mode_img)
    elif suffix == '.yml':
        with open(file, 'w') as f:
            yaml.safe_dump(obj, f, sort_keys=False)
    elif suffix == ".bson":
        if return_buffer:
            return bson.dumps(obj)
        with open(file, "wb") as f:
            f.write(bson.dumps(obj))
    elif suffix == ".txt":
        if return_buffer:
            return obj.encode()
        if isinstance(obj, str):
            obj = [obj]
        with open(file, "w") as f:
            f.writelines(obj)
    elif suffix == ".text":
        if return_buffer:
            return obj.encode()
        raise NotImplementedError
    elif suffix == ".msgpack":
        if file.startswith("hdfs"):
            from unhcv.datasets.common_datasets import LmdbBackend, KvBackend
            kv_backend = KvBackend(file, mode="w")
            for key, value in obj.items():
                kv_backend.write(key, msgpack.packb(value))
            kv_backend.end_write()
            return
        with open(file, "wb") as f:
                packed = msgpack.packb(obj)
                f.write(packed)
    else:
        raise NotImplementedError


def config_load(file):
    from .mmcv_utils import Config
    config = Config.fromfile(file)
    return config


def obj_load(file, buffer=None, load_config=False):
    suffix = osp.splitext(file)[1]
    if suffix == '.json':
        if buffer is None:
            with open(file, 'r') as f:
                obj = json.load(f)
        else:
            obj = json.loads(buffer)
    elif suffix == '.pkl':
        if buffer is None:
            with open(file, 'rb') as f:
                obj = pickle.load(f)
        else:
            obj = pickle.loads(buffer)
    elif suffix.lower() in ['.jpg', '.png', '.jpeg']:
        if buffer is None:
            obj = Image.open(file)
        else:
            obj = Image.open(io.BytesIO(buffer))
    elif suffix == '.yml' or suffix == '.yaml':
        if load_config:
            from .mmcv_utils import Config
            obj = Config.fromfile(file)
        else:
            with open(file, 'r') as f:
                obj = yaml.safe_load(f)
    elif suffix == ".bson":
        if buffer is None:
            with open(file, 'rb') as f:
                obj = bson.loads(f.read())
        else:
            obj = bson.loads(buffer)
    elif suffix == ".txt":
        if buffer is not None:
            obj = buffer.decode()
            return obj
        with open(file, "r") as f:
            obj = f.readlines()
    elif suffix == ".text":
        if buffer is not None:
            obj = buffer.decode()
            return obj
        raise NotImplementedError
    elif suffix == ".msgpack":
        with open(file, "rb") as f:
            obj = msgpack.loads(f.read())
    elif suffix == ".py":
        from .mmcv_utils import Config
        obj = Config.fromfile(file)
    elif buffer is not None:
        obj = bson.loads(buffer)
    else:
        raise NotImplementedError(f"file: {file}")
    return obj

class BufferTool:
    @staticmethod
    def pil_decode(buffer, image_mode=None):
        if isinstance(buffer, bytes):
            pass
        elif isinstance(buffer, str):
            buffer = b64decode(buffer)
        else:
            raise NotImplementedError
        with Image.open(io.BytesIO(buffer)) as image:
            if image_mode == "RGB":
                if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                    breakpoint()
                    image = image.convert("RGBA")
                    white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                    white.paste(image, mask=image.split()[3])
                    image = white
                else:
                    image = image.convert("RGB")
            elif image_mode is None:
                image = image.copy()
            else:
                raise NotImplementedError
        return image
    
    @staticmethod
    def pil_encode(image):
        file_fake = io.BytesIO()
        image.save(file_fake)
        return file_fake.getvalue()

class LmdbFile:
    
    @staticmethod
    def build_lmdb_reader(file):
        env = lmdb.open(
                file,
                max_readers=4,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
        txn = env.begin(write=False)
        return txn
    
    @staticmethod
    def read_img(txn, name):
        img_bin = txn.get(name.encode())
        img_bin = np.fromstring(img_bin, dtype=np.uint8)
        img = cv2.imdecode(img_bin, cv2.IMREAD_UNCHANGED)
        return img

def write_txt(path, txts):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(txts)


if __name__ == '__main__':
    pass
