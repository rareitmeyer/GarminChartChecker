# Tool to fetch Coast Guard "Local Notice To Mariners" updates
# that impact NOAA charts. Will create two CSV files, a "lnms.csv"
# with all the data, and "lnms_best.csv" with the ones I think
# are easiest to see on a chart.
#
# Charts are hardcoded for San Francisco Bay, but change the
# list of charts at the bottom of the file if interested in
# someplace else.
#
# Requires python3.x with 3rd party packages:
# requests, for getting the data
# numpy and scipy, for computing nearest locations
#
# Copyright 2017, R. A. Reitmeyer
# Released under BSD license

import csv
import math
import re
import time

import requests


meters_per_nautical_mile = 1852
semi_major_axis = 6378137.0 # radius at equator, meters
flattening = 1/298.257223563
semi_minor_axis = (1-flattening)*semi_major_axis # radius to pole, meters...


def get_nm(chart_num, filter=True):
    url = 'https://ocsdata.ncd.noaa.gov/ntm/Listing_Text.aspx?Chart={chart_num}&DateSince=20000914'.format(chart_num=chart_num)
    resp = requests.get(url)
    lines = resp.text.split('\r\n')
    data = []
    header = lines[1].split('\t')  # head 0 is a note about the latest chart.
    # sanity check some columns
    assert(header[0] == 'Chart')
    assert(header[1] == 'Action')
    assert(header[2] == 'Item Name')
    assert(header[3] == 'Charting Label')
    assert(header[6] == 'LatDD')
    assert(header[7] == 'LongDD')
    assert(header[8] == 'Published Document')
    cur_year = time.strftime('%y')
    for i,l in enumerate(lines[2:]):
        fields = [f.strip() for f in l.split('\t')]
        if len(fields) < len(header):
            continue
        rec = {
            'id': i,
            'chart': fields[0],
            'action': fields[1].lower(),
            'item': fields[2],
            'label': fields[3],
            'lat': fields[6],
            'lng': fields[7],
            'doc': fields[8],
            }
        m = re.match('LNM (?P<ww>[0-9]{2})/(?P<yy>[0-9]{2})', rec['doc'])
        if m:
            yy = m.group('yy')
            ww = m.group('ww')
            if yy <= cur_year:
                # conveiently, data starts at 2000-09-14, so all years 2000+
                rec['effective'] = '20{yy}w{ww}'.format(yy=yy, ww=ww)

        # For purposes of checking charts, it's best to filter out
        # things that won't be obvious, date-specfic, labeled changes
        # on a chart. Depth tabulation changes, submarine cables, anchorages
        # and the like might not be easy to spot.
        if filter:
            if not (('effective' in rec)
                    and (rec['action'] in ['add', 'delete'])
                    and (not re.search('(anchorage)|(black double dashed line)|(black label)|(channel limit)|(dashed magenta line)|(depth legend)|(double solid lines)|(ferry maneuvering area)|(ferry route)|(legend)|(note)|(obstruction)|(pipeline)|(prohibited area)|(restricted area)|(shoaling)|(submarine cable)|(security zone)|(\(see note)|(sound )|(sounding)|(tabulation)', rec['item'], re.IGNORECASE))
                    and (not re.match('none', rec['label'], re.IGNORECASE))):
                rec['use'] = 'n'
            else:
                rec['use'] = 'y'
        data.append(rec)
    field_names = [
        'id',
        'chart',
        'action',
        'item',
        'label',
        'lat',
        'lng',
        'doc',
        'effective',
        'use'
    ]
    return (field_names, data)


def _radius_at_latitude(lambda0):
    # From gis.stackexchange.com/questions/20200/how-do-you-compute-the-earths-radius-at-a-given-geodetic-latitude
    lambda0 = math.pi/180*lambda0
    return math.sqrt((semi_major_axis**4*math.cos(lambda0)**2
        +semi_minor_axis**4*math.sin(lambda0)**2)/(semi_major_axis**2
        *math.cos(lambda0)**2+semi_minor_axis**2*math.sin(lambda0)**2))


def angle_fixup(angle, lower=0, upper=360):
    if angle < lower:
        angle += upper-lower
    elif angle >= upper:
        angle -= upper-lower
    return angle


def gc_dist_dir(lambda1, phi1, lambda2, phi2):
    """Great circle distance and azimuths, using formulas from Wikipedia's
    page for "Solution of Triangles." Approximates radius as the
    mid-latitude radius.

    TODO: Should write a vectorized version of this optimized for scipy.
    """

    a = (90.0-phi2)*math.pi/180
    b = (90.0-phi1)*math.pi/180
    gamma = (lambda2-lambda1)*math.pi/180
    c = math.atan2(math.sqrt((math.sin(a)*math.cos(b)-math.cos(a)*math.sin(b)*math.cos(gamma))**2+(math.sin(b)*math.sin(gamma))**2), math.cos(a)*math.cos(b)+math.sin(a)*math.sin(b)*math.cos(gamma))
    alpha = math.atan2(math.sin(a)*math.sin(gamma), math.sin(b)*math.cos(a)-math.cos(b)*math.sin(a)*math.cos(gamma))
    beta = math.atan2(math.sin(b)*math.sin(gamma), math.sin(a)*math.cos(b)-math.cos(a)*math.sin(b)*math.cos(gamma))

    r = _radius_at_latitude((phi1+phi2)/2)
    dist = c*r/meters_per_nautical_mile
    a1_to_2 = angle_fixup(360+alpha*180/math.pi, 0, 360)
    a2_to_1 = angle_fixup(360-beta*180/math.pi, 0, 360)
    retval = (dist, a1_to_2, a2_to_1)
    return retval


def add_nearest_neighbor(field_names, data, prefix=''):
    import numpy as np
    import scipy.spatial

    rows = len(data)
    as_array = np.ndarray(shape=(rows,4), dtype=float)
    as_array[0:rows,0:2] = [(float(r['lat']), float(r['lng'])) for r in data]
    # should really project into web mercator and search on x and y,
    zoom = 2**10 # fairly arbitrary
    as_array[:,2] = 128/np.pi*zoom*(as_array[:,1]+np.pi)
    as_array[:,3] = 128/np.pi*zoom*(np.pi-np.log(np.tan(np.pi/4+as_array[:,0]/2)))
    tree = scipy.spatial.cKDTree(as_array[:,2:4])
    dd, ii = tree.query(as_array[:,2:4], k=2) # 2nd nearest, assuming self is nearest
    new_fields = ['nearest_id', 'dist_nmi', 'a1_to_2']
    if prefix != '':
        new_fields = [prefix+"_"+x for x in new_fields]
    for i,d in enumerate(data):
        other = ii[i,1]
        if other == i:
            other = ii[i,0]
        args = [float(x) for x in [d['lng'],d['lat'],data[other]['lng'], data[other]['lat']]]
        dist_m, a1_to_2, a2_to_1 = gc_dist_dir(*args)
        d[new_fields[0]] = data[other]['id']
        d[new_fields[1]] = dist_m
        d[new_fields[2]] = a1_to_2
    field_names += new_fields
    return field_names, data


def save_csv(filename, field_names, data):
    with open(filename, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(field_names)
        for rec in data:
            writer.writerow([str(rec.get(f,'')) for f in field_names])

def main(charts):
    field_names = None
    data = []
    for c in charts:
        field_names, d = get_nm(c)
        data += d
    field_names, data = add_nearest_neighbor(field_names, data, prefix='all')
    nouse_data = [d for d in data if d['use']=='n']
    field_names, use_data = add_nearest_neighbor(field_names, [d for d in data if d['use']=='y'], prefix='use')
    data = nouse_data + data
    data.sort(key=lambda r: r.get('effective', ''))
    save_csv('lnms.csv', field_names, data)

    save_csv('lnms_best.csv', field_names, [d for d in data if d['use']=='y' and d['use_dist_nmi'] > 10/meters_per_nautical_mile])


if __name__ == '__main__':
    charts = ['18650', '18651', '18653', '18654']
    main(charts)
