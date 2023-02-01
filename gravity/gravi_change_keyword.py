# Change header keywords

import argparse
from astropy.io import fits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the keyword of the FITS file')
    parser.add_argument('filename', type=str, help='The filename to be modified.')
    parser.add_argument('--key', '-k', type=str, help='The keyword to modify.')
    parser.add_argument('--value', '-v', type=str, help='The new value.')
    parser.add_argument('--extension', '-e', type=int, default=0, help='The extension. [0]')
    parser.add_argument('--name', type=str, default=None, help='Save to a new filename. [NONE]')
    args = parser.parse_args()

    if args.name is None:
        hdul = fits.open(args.filename, mode='update')
    else:
        hdul = fits.open(args.filename)

    header = hdul[args.extension].header

    if header.get(args.key, None) is None:
        raise KeyError('Cannot find {0} in extension {1}!'.format(args.key, args.extension))
    
    header[args.key] = args.value

    if args.name is None:
        hdul.flush()
    else:
        hdul.writeto(args.name, overwrite=True)
