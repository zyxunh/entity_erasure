from pycocotools import mask as maskUtils
import numpy as np

def annToRLE(segm, shape):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    h, w = shape[:2]
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def coco_ann_to_mask(segm, shape=None):
    """
    shape: hw
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    if shape is None:
        shape = segm['size']
    rle = annToRLE(segm, shape)
    m = maskUtils.decode(rle)
    return m

def mask_to_coco_ann(segm):
    segm = np.asfortranarray(segm)
    rle = maskUtils.encode(segm)
    return rle