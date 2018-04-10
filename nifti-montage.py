#!/usr/bin/env python

import os
import json
import math
import dcmstack
import logging
import datetime
import zipfile
import cStringIO
import numpy as np
import nibabel as nb
from PIL import Image
logging.basicConfig()
log = logging.getLogger('nifti-montage')

Image.MAX_IMAGE_PIXELS = 1000000000

def generate_montage(imagedata, timepoints=[], bits16=False):
    """Generate a montage."""
    # Figure out the image dimensions and make an appropriate montage.
    # NIfTI images can have up to 7 dimensions. The fourth dimension is
    # by convention always supposed to be time, so some images (RGB, vector, tensor)
    # will have 5 dimensions with a single 4th dimension. For our purposes, we
    # can usually just collapse all dimensions above the 3rd.
    # TODO: we should handle data_type = RGB as a special case.
    # TODO: should we use the scaled data (getScaledData())? (We do some auto-windowing below)

    # This transpose (usually) makes the resulting images come out in a more standard orientation.
    # TODO: we could look at the qto_xyz to infer the optimal transpose for any dataset.
    data = imagedata.transpose(np.concatenate(([1, 0], range(2, imagedata.ndim))))
    num_images = np.prod(data.shape[2:])

    if data.ndim < 2:
        raise ValueError('NIfTI file must have at least 2 dimensions')
    elif data.ndim == 2:
        # a single slice: no need to do anything
        num_cols = 1
        data = np.atleast_3d(data)
    elif data.ndim == 3:
        # a simple (x, y, z) volume- set num_cols to produce a square(ish) montage.
        rows_to_cols_ratio = float(data.shape[0])/float(data.shape[1])
        num_cols = int(math.ceil(math.sqrt(float(num_images)) * math.sqrt(rows_to_cols_ratio)))
    elif data.ndim >= 4:
        # timeseries (x, y, z, t) or more
        num_cols = data.shape[2]
        data = data.transpose(np.concatenate(([0, 1, 3, 2], range(4, data.ndim)))).reshape(data.shape[0], data.shape[1], num_images)
        if len(timepoints) > 0:
            data = data[..., timepoints]

    num_rows = int(np.ceil(float(data.shape[2])/float(num_cols)))
    montage = np.zeros((data.shape[0] * num_rows, data.shape[1] * num_cols), dtype=data.dtype)
    for im_num in range(data.shape[2]):
        slice_r, slice_c = im_num / num_cols * data.shape[0], im_num % num_cols * data.shape[1]
        montage[slice_r:slice_r + data.shape[0], slice_c:slice_c + data.shape[1]] = data[:, :, im_num]

    # montage = montage.copy()        # is this necessary? need a deep copy?
    if montage.dtype == np.uint8 and bits16:
        montage = np.cast['uint16'](data)
    elif montage.dtype != np.uint8 or (montage.dtype != np.uint16 and bits16):
        montage = montage.astype(np.float32)  # do scaling/clipping with floats
        clip_vals = np.percentile(montage, (20.0, 99.0))   # auto-window the data by clipping
        montage = montage.clip(clip_vals[0], clip_vals[1]) - clip_vals[0]
        if bits16:
            montage = np.cast['uint16'](np.round(montage/(clip_vals[1]-clip_vals[0])*65535))
        else:
            montage = np.cast['uint8'](np.round(montage/(clip_vals[1]-clip_vals[0])*255.0))
    return montage


def generate_pyramid(montage, tile_size):
    """
    Slice up a NIfTI file into a multi-res pyramid of tiles.
    We use the file name convention suitable for d3tiles
    The zoom level (z) is an integer between 0 and n, where 0 is fully zoomed out and n is zoomed in.
    E.g., z=0 is for 1 tile covering the whole world, z=1 is for 2x2=4 tiles, ... z=n is the original resolution.
    """
    montage_image = Image.fromarray(montage, 'L')
    montage_image = montage_image.crop(montage_image.getbbox())  # crop away edges that contain only zeros
    sx, sy = montage_image.size
    if sx * sy < 1:
        raise ValueError('degenerate image size (%d, %d): no tiles will be created' % (sx, sy))
    if sx < tile_size and sy < tile_size:  # Panojs chokes if the lowest res image is smaller than the tile size.
        tile_size = max(sx, sy)

    pyramid = {}
    pyramid_meta = {
        'tile_size': tile_size,
        'mimetype': 'image/jpeg',
        'real_size': montage_image.size,
        'zoom_levels': {},
    }
    divs = max(1, int(np.ceil(np.log2(float(max(sx, sy))/tile_size))) + 1)
    for z in range(divs):
        # flip the z label to be d3 friendly
        level = (divs - 1) - z
        ysize = int(round(float(sy)/pow(2, z)))
        xsize = int(round(float(ysize)/sy*sx))
        xpieces = int(math.ceil(float(xsize)/tile_size))
        ypieces = int(math.ceil(float(ysize)/tile_size))
        log.debug('level %s, size %dx%d, splits %d,%d' % (level, xsize, ysize, xpieces, ypieces))
        # TODO: we don't need to use 'thumbnail' here. This function always returns a square
        # image of the requested size, padding and scaling as needed. Instead, we should resize
        # and chop the image up, with no padding, ever. panojs can handle non-square images
        # at the edges, so the padding is unnecessary and, in fact, a little wrong.
        im = montage_image.copy()
        im.thumbnail([xsize, ysize], Image.ANTIALIAS)
        im = im.convert('L')    # convert to grayscale
        for x in range(xpieces):
            for y in range(ypieces):
                tile = im.copy().crop((x*tile_size, y*tile_size, min((x+1)*tile_size, xsize), min((y+1)*tile_size, ysize)))
                log.debug(tile.size)
                if tile.size != (tile_size, tile_size):
                    log.debug('tile is not square...padding')
                    background = Image.new('L', (tile_size, tile_size), 'white')  # what to pad with? default black
                    background.paste(tile, (0, 0))
                    tile = background
                buf = cStringIO.StringIO()
                tile.save(buf, 'JPEG', quality=85)
                pyramid[(level, x, y)] = buf
        pyramid_meta['zoom_levels'][level] = (xpieces, ypieces)
    return pyramid, montage_image.size, pyramid_meta


def generate_dir_pyr(imagedata, outbase, tile_size=256):
    """Generate a panojs image pyramid directory."""
    montage = generate_montage(imagedata)
    pyramid, pyramid_size, pyramid_meta = generate_pyramid(montage, tile_size)

    # write directory pyramid
    image_path = os.path.join(outbase, 'images')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
        for idx, tile_buf in pyramid.iteritems():
            with open(os.path.join(image_path, ('%03d_%03d_%03d.jpg' % idx)), 'wb') as fp:
                fp.write(tile_buf.getvalue())

    # check for one image, pyramid file
    if not os.path.exists(os.path.join(outbase, 'images', '000_000_000.jpg')):
        raise ValueError('montage (flat png) not generated')
    else:
        log.debug('generated %s' % outbase)
        return outbase

def generate_zip_pyr(imagedata, outbase, tile_size=256):
    montage = generate_montage(imagedata)
    pyramid, pyramid_size, pyramid_meta = generate_pyramid(montage, tile_size)
    zip_name = outbase + '.zip'
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED) as zf:
        pyramid_meta['dirname'] = os.path.basename(outbase)
        zf.comment = json.dumps(pyramid_meta)
        montage_jpeg = os.path.join(os.path.basename(outbase), 'montage.jpeg')
        buf = cStringIO.StringIO()
        Image.fromarray(montage).convert('L').save(buf, format='JPEG', optimize=True)
        zf.writestr(montage_jpeg, buf.getvalue())
        for idx, tile_buf in pyramid.iteritems():
            tilename = 'z%03d/x%03d_y%03d.jpg' % idx
            arcname = os.path.join(os.path.basename(outbase), tilename)
            zf.writestr(arcname, tile_buf.getvalue())

    return zip_name

def generate_flat(imagedata, filepath):
    """Generate a flat png montage."""
    montage = generate_montage(imagedata)
    Image.fromarray(montage).convert('L').save(filepath, optimize=True)

    if not os.path.exists(filepath):
        raise ValueError('montage (flat png) not generated')
    else:
        log.debug('generated %s' % os.path.basename(filepath))
        return filepath


def nifti_montage(fp, outbase=None, voxel_order='LPS', tile_size=256, montage_type='zip'):
    """
    Write imagedata to image montage pyramid.

    Parameters
    ----------
    fp : str
        full path to nifti file
    outbase : [default None]
        output name prefix.
    voxel_order : str [default LPS]
        three character string indicating the voxel order, ex. 'LPS'.
    montage_type : str [default 'zip']
        type of montage to create. can be 'zip' or 'png'.
    tile_size : int [default 256]
        tile_size for generated zip or directory pyramid. Has no affect on montage_type 'png'.

    Returns
    -------
    results : list
        list of files written.

    """

    if not os.path.exists(fp):
        print 'could not find %s' % fp
        print 'checking input directory ...'
        if os.path.exists(os.path.join('/flywheel/v0/input', fp)):
            fp = os.path.join('/flywheel/v0/input', fp)
            print 'found %s' % fp

    if not outbase:
        fn = os.path.basename(fp)
        outbase = os.path.join('/flywheel/v0/output', fn.split('.nii')[0] + '.montage')
        log.info('setting outbase to %s' % outbase)


    # Load data
    log.info('loading %s' % fp)
    nii = nb.load(fp)
    data = nii.get_data()
    if data is None:
        raise ValueError('Data array is None')
    else:
        log.info('data loaded')


    # Montage
    result = []
    if voxel_order:
        log.info('reodering voxels to %s' % voxel_order)
        data, _, _, _ = dcmstack.reorder_voxels(data, nii.get_affine(), voxel_order)
    if montage_type == 'png':
        log.info('generating montage type: flat png')
        result = generate_flat(data, outbase + '.png')
    elif montage_type == 'zip':
        log.info('generating montage type: zip of tiles')
        result = generate_zip_pyr(data, outbase, tile_size)
    else:
        raise ValueError('montage type must be zip or png. not %s' % montage_type)

    # Write metadata file
    output_files = os.listdir(os.path.dirname(outbase))
    files = []
    if len(output_files) > 0:
        for f in output_files:

            fdict = {}
            fdict['name'] = f

            if f.endswith('montage.zip'):
                ftype = 'montage'
            else:
                ftype = 'None'

            fdict['type'] = ftype
            files.append(fdict)

        metadata = {}
        metadata['acquisition'] = {}
        metadata['acquisition']['files'] = files

        with open(os.path.join(os.path.dirname(outbase),'.metadata.json'), 'w') as metafile:
            json.dump(metadata, metafile)

    return result

if __name__ == '__main__':

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('nifti', help='path to nifti file')
    ap.add_argument('outbase', nargs='?', help='outfile name prefix')
    ap.add_argument('--log_level', help='logging level', default='info')
    args = ap.parse_args()

    log.setLevel(getattr(logging, args.log_level.upper()))
    logging.getLogger('sctran.data').setLevel(logging.INFO)

    # CONFIG: If there is a config file then load that, else load the manifest and read the default values.
    if os.path.exists('/flywheel/v0/config.json'):
        config_file = '/flywheel/v0/config.json'
        MANIFEST=False
    else:
        config_file = '/flywheel/v0/manifest.json'
        MANIFEST=True

    with open(config_file, 'r') as jsonfile:
        config = json.load(jsonfile)
    config = config.pop('config')

    if MANIFEST:
        voxel_order = config['voxel_order']['default']
        tile_size = config['tile_size']['default']
        montage_type = config['montage_type']['default']
    else:
        voxel_order = config['voxel_order']
        tile_size = config['tile_size']
        montage_type = config['montage_type']

    # Generate montage
    log.info('job start: %s' % datetime.datetime.utcnow())
    montage_file = nifti_montage(args.nifti, args.outbase, voxel_order, tile_size, montage_type)
    log.info('job stop: %s' % datetime.datetime.utcnow())

    if montage_file:
        log.info('generated %s' % montage_file)
        os.sys.exit(0)
    else:
        log.info('Failed.')
        os.sys.exit(1)
