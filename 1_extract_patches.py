import sys
import os
from glob import glob
import shutil
import configparser
from collections import defaultdict
from math import ceil, atan2, degrees, sqrt, cos, sin, radians
import logging

import numpy as np
import openslide
import cv2
import pyvips
from affine import Affine

from PIL import Image
import matplotlib.pyplot as plt


def rm_n_mkdir(dir_path):
    '''
    Remove (if was present) and create new directory.

    Parameters:
    - dir_path (str):       full path of the directory

    Returns: None
    '''
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def print_slide_info(slide_path, ds=1, display_prop=False):
    '''
    Parse slide metadata and display information.

    Parameters:
    - slide_path (str):     full path to the slide file
    - ds (int):             preferred downsample level
    - display_prop (bool):  whether to display additional slide properties

    Returns: None
    '''
    slide = openslide.OpenSlide(slide_path)
    print (f"Format: {slide.detect_format(slide_path)}")
    print ("-----")
    print (f"1. Num of levels: {slide.level_count}")
    print (f"2. Slide shape: {slide.dimensions}")
    print (f"3. Level downsamples: {slide.level_downsamples}")
    print (f"4. Level dimensions: {slide.level_dimensions}")
    print ("-----")
    # print (f"Best level for ds={ds}: {slide.get_best_level_for_downsample(ds)}")
    if display_prop:
        print ("-----")
        print (f"Properties:")
        print (slide.properties)


def get_polypons(config_polygon):
    '''
    Get polygon information in a dictionary from parsed .itn file.

    Parameters:
    - config_polygon (configparser.SectionProxy):   config['Polygon'] object

    Returns: 
    - poly_points (dict):   polygon information, like 
        {
            0: [(x0,y0), (x1,y1),...],
            1: [(x0,y0), (x1,y1),...],
            ...
        }
    '''
    poly_points = defaultdict(list)
    l_it = list(((el.split('_')[2:4]) for el in (list(config_polygon))))
    for i in range(0, len(l_it), 2):
        poly, num = l_it[i]
        poly_points[int(poly)].append(
            (float(config_polygon[f"poly_x_{poly}_{num}"]), 
             float(config_polygon[f"poly_y_{poly}_{num}"]))
        )
    return poly_points


def get_n_points_polygons(config_polygon):
    '''
    Get number of points for each polygon.

    Parameters:
    - config_polygon (configparser.SectionProxy):   config['Polygon'] object

    Returns: 
    - n_poly_points (dict):   number of points in each polygon, like 
        {
            0: n_0,
            1: n_1,
            ...
        }
    '''
    n_poly_points = {}
    polygons = get_polypons(config_polygon)
    for p in polygons.keys():
        n_poly_points[p] = len(polygons[p])
    return n_poly_points


def scale_coords(image_shape, scaled_image_shape, original_coords):
    '''
    Scale original polygon coordinates.

    Parameters:
    - image_shape (tuple):          (img.width, img.height, 3) 
    - scaled_image_shape (tuple):   (scaled_img.width, scaled_img.height, 3)
    - original_coords (dict):       poligons with coordinates, like:
        {
        0: [(x0,y0), (x1,y1),...],
        1: [(x0,y0), (x1,y1),...],
        ...
        }

    Returns: 
    - transformed_coords (dict): scaled coordinates in the same format as 'original_coords' 
    '''
    ow, oh, _ = image_shape    
    w, h, _ = scaled_image_shape
    M = cv2.getAffineTransform(np.float32([[ow, 0], [ow, oh], [0, oh]]), np.float32([[w, 0], [w, h], [0, h]]))
    transformed_coords = {}
    for k in original_coords.keys():
        npcoords = np.array([original_coords[k]], np.float32) # np.int32?
        transformed_coords[k] = cv2.transform(npcoords, M).squeeze().tolist()
    return transformed_coords

def draw_polys(image, coords, fill=False, color=(0, 255, 0), lt=5):
    '''
    Draw polygons on the image.

    Parameters:
    - image (np.array):   convert PIL image using np.asarray()
    - corods: (dict):     poligons with coordinates, like:
        {
        0: [(x0,y0), (x1,y1),...],
        1: [(x0,y0), (x1,y1),...],
        ...
        }
    - fill (bool):        whether to fill polygon with color 
    - color (tuple):      color of line and fill (RGB), e.g. (0,255,0) - green 
    - lt (int):           line width in pixels 

    Returns: 
    - image (np.array): cv2 image with marked polygons

    '''
    for k in coords.keys():
        npcoords = np.array([coords[k]], np.int32)
        if fill:
            cv2.fillPoly(image, npcoords, color)
        else:
            cv2.polylines(image, npcoords, True, color, lt)
    return image


def get_annotated_image(slide_path, scaled_annotations, level, fill=False, color=(0, 255, 0), lt=5, verbose=False):
    '''
    Draw polygons on the image and return it as numpy image

    Parameters:
    - slide_path (string): path to slide
    - scaled_annotations (dict): points, obtained from annotation `.itn` file and transformed with `scale_coords`
    - level (int): downsampling level from 1 to 3 usually, 0 - original, maps to downsampling factor in `slide.level_downsamples`
    - fill (bool):        whether to fill polygon with color 
    - color (tuple):      color of line and fill (RGB), e.g. (0,255,0) - green 
    - lt (int):           line width in pixels
    - verbose (bool):     whether to print debug information

    Returns: 
    - image (np.array): cv2 image with polygons
    '''
    slide = pyvips.Image.new_from_file(slide_path, level=level)
    np_img = cv2.cvtColor(slide.numpy(), cv2.COLOR_BGRA2RGB)

    if verbose:
        print(f"Shape of slide: {np.shape(np_img)}")
    
    image = draw_polys(np_img, scaled_annotations, fill=fill, color=color, lt=lt)
    return image

def extract_patches_from_point(slide, slide_id, poly_point_coords, polygon_n, save_dir, patch_size=128, verbose=False):
    '''
    Extract patch from point on slide.

    Parameters:
    - slide (pyvips object): pyvips image from slide with specified downsample level
    - slide_id (string): name of the slide
    - poly_point_coords (tuple): one-polygon points, obtained from annotation `.itn` file and transformed with `scale_coords` and `squash_poly_points`
    - polygon_n (int): polygon number
    - save_dir (string): save path of the patch
    - patch_size (int): size of the patch  
    - verbose (bool): whether to print debug information

    Returns: None
    '''
    npcoords = np.squeeze(np.array([poly_point_coords], np.int32))
    for n, (x,y) in enumerate(npcoords):
        try:
            crop_img = slide.crop(x - patch_size, y - patch_size, patch_size * 2, patch_size * 2)
            crop_img = crop_img.copy_memory()
            crop_img.write_to_file(os.path.join(save_path, f"{slide_id}_{polygon_n}_{n}_{int(x)}_{int(y)}.png"))
        except:
            sys.exit(f"Not valid patch size extraction: crop - ({x-patch_size, y-patch_size, patch_size*2, patch_size*2})")

def get_distance_btw_points(p1, p2):
    '''
    Get distance between 2 points.

    Parameters:
    - p1 (tuple): first point of type (x, y)
    - p2 (tuple): second point of type (x, y)

    Returns: 
    - distance (int): cv2 image with polygons
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = sqrt((dx**2) + (dy**2))
    return distance

def rotate(img_size, rotated_size, point, angle):
    '''
    Get new coordinates for point on a rotated image.

    Parameters:
    - img_size (tuple): size of original image (w, h, _)
    - rotated_size (tuple): size of rotated image (w, h, _) by `angle`
    - point (tuple): point of type (x, y) that should be rotated
    - angle (int): angle of the rotation in degrees

    Returns: 
    - rotated_point (tuple): rotated point of type (x, y)
    '''
    org_center = np.array((img_size[0], img_size[1])) / 2.
    rot_center = np.array((rotated_size[0], rotated_size[1])) / 2.
    org = point - org_center
    angle_rad = np.deg2rad(angle)
    new_coord = np.array([
        (org[0] * np.cos(angle_rad)) + (org[1] * np.sin(angle_rad)),
        (-org[0] * np.sin(angle_rad)) + (org[1] * np.cos(angle_rad)) 
    ])
    rotated_point = new_coord + rot_center
    return rotated_point

def squash_poly_points(points, min_size, close=True):
    '''
    Reduce number of points in polygon, based on minimal distance between points.

    Parameters:
    - points (list): list of points of type [(x0, y0), (x1, y1), ..., (xn, yn)]
    - min_size (int): minimum size threshold used to squash list of points
    - close (bool): whether to close polygon: append last point for the polygon to close it

    Returns: 
    - out_list (list): polygon, with reduced number of points of type [(x0, y0), (x1, y1), ..., (xn, yn)]
    '''
    out_list = []
    for i in range(len(points)):
        if i > (len(points)-1):
            break
        start_elem = points[i]
        dist = 0
        
        while dist < min_size and i < (len(points)-1):
            next_elem = points[i+1]
            dist = get_distance_btw_points(start_elem, next_elem)
            if dist < min_size:
                del points[i+1]
        out_list.append(start_elem)

    if close: out_list.append(out_list[0])
    return out_list


def check_rot_points(rotated_image, rot_points, patch_size):
    width = rotated_image.width
    height = rotated_image.height
    
    x1 = int(rot_points[0][0])
    y1_inv = int(rot_points[0][1])
    
    x2 = int(rot_points[1][0])
    y2_inv = int(rot_points[1][1])
    
    # print (f"Rotated_points: [{x1},{y1_inv}] [{x2},{y2_inv}]")
    # print (f"Rotated_img: [{width}, {height}]")
    
    if ((x1 < width and x2 < width and y1_inv < height and y2_inv < height) and
        (x1 > 0 and x2 > 0 and y1_inv > 0 and y2_inv > 0) and
        (y1_inv + int(patch_size/2) < height and y1_inv - int(patch_size/2 > 0)) and
        (y2_inv + int(patch_size/2) < height and y2_inv - int(patch_size/2 > 0))
       ):
        return True
    else:
        return False

def extract_patches_btw_points(slide, slide_id, pts, polygon_n, patch_size, save_dir, errors_dir, verbose=False):
    '''
    Extract patch from point on slide.

    Parameters:
    - slide (pyvips object): pyvips image from slide with specified downsample level
    - slide_id (string): name of the slide
    - pts (tuple): 2 points of type (x,y)
    - polygon_n (int): polygon number, from which points were taken
    - patch_size (int): size of the patch 
    - save_dir (string): save path of the patch
    - verbose (bool): whether to print debug information

    Returns: None
    '''
    points = [pts[0], pts[1]]
    points.sort(key=lambda x: x[0], reverse=True)

    height = slide.height
    width = slide.width
    # center = ((width/2), (height/2))

    dx = (points[0][0] - points[1][0])
    dy = (points[0][1] - points[1][1])
    
    ### If 2 points are equal (small round area)
    if sqrt(dx**2 + dy**2) < (patch_size // 4):
        try: 
            x = points[0][0]
            y = points[0][1]
            crop_img = slide.crop(x, y, patch_size, patch_size)
            crop_img = crop_img.copy_memory()
            if verbose: print(crop_img.width, crop_img.height)
            crop_img.write_to_file(os.path.join(save_dir, f"{slide_id}_{polygon_n}_{int(x)}_{int(y)}.png"))
            if verbose: print(f"One-point crop: {slide_id}_{polygon_n}_{int(x)}_{int(y)}.png")
        except:
            print (f"Skipping one-point crop: {slide_id}_{polygon_n}_{int(x)}_{int(y)}.png")
        return

    angle = atan2(dy,dx)
    
    affine_transform = Affine.identity()
    affine_transform = affine_transform.rotation(degrees(-angle)) #### NB: -angle
    vp_matrix = (affine_transform.a, affine_transform.b, affine_transform.d, affine_transform.e)    
    rotated_image = slide.affine(vp_matrix)
    
    r_height = rotated_image.height
    r_width = rotated_image.width
    
    p1_r = rotate((width, height), (r_width, r_height), points[0], degrees(angle))
    p2_r = rotate((width, height), (r_width, r_height), points[1], degrees(angle))
    
    dx = p2_r[0] - p1_r[0]
    dy = p2_r[1] - p1_r[1]
    distance_after = sqrt(dx**2 + dy**2)

    n_full_patches = int(distance_after // patch_size)
    residual = int(distance_after % patch_size)
    
    rot_points = [p1_r, p2_r]
    if rot_points[0][0] != rot_points[1][0]:
        rot_points.sort(key=lambda x: x[0], reverse=False)
    else:
        rot_points.sort(key=lambda x: x[1], reverse=False)
    if check_rot_points(rotated_image, rot_points, patch_size):
        line_stops = [(rot_points[0][0], rot_points[0][1])]
        for num in range(1, n_full_patches):
            line_stops.append((rot_points[0][0] + (patch_size * num), rot_points[0][1]))

        for idx, line_stop in enumerate(line_stops):
            x, y = line_stop
            y = y - int(patch_size / 2)

            if n_full_patches > 0:
                if idx == (len(line_stops) - 1) and residual != 0:           
                    try:
                        crop_img = rotated_image.crop(x, y, (patch_size + residual), patch_size)
                    except:
                        print("An exception occurred while cropping:")
                        print(f"Rotated image: {rotated_image.width} x {rotated_image.height}; Angle: {degrees(angle)}; Top-left point: ({x},{y})")

                        rotated_image = rotated_image.draw_circle([0,255,0,255], rot_points[0][0], rot_points[0][1], 500, fill=True) # RGBA
                        rotated_image = rotated_image.draw_circle([255,0,0,255], rot_points[1][0], rot_points[1][1], 500, fill=True) # RGBA
                        orig_image = slide.draw_circle([0,255,255,255], pts[0][0], pts[0][1], 500, fill=True) # RGBA
                        orig_image = orig_image.draw_circle([255,0,255,255], pts[1][0], pts[1][1], 500, fill=True) # RGBA

                        rotated_image = rotated_image.thumbnail_image(4096)
                        orig_image = orig_image.thumbnail_image(4096)

                        rotated_image = rotated_image.copy_memory()
                        rotated_image.write_to_file(os.path.join(errors_dir, f'418_rotated_error_{slide_id}_{polygon_n}.png'))
                        orig_image = orig_image.copy_memory()
                        orig_image.write_to_file(os.path.join(errors_dir, f'418_original_error_{slide_id}_{polygon_n}.png'))
                        return
                        # sys.exit("Exit endpoint 418")

                    crop_img = crop_img.copy_memory()
                    if verbose: print (f"Crop size: {crop_img.width} x {crop_img.height}")
                    crop_img.write_to_file(os.path.join(save_dir, f"{slide_id}_{polygon_n}_{int(x)}_{int(y)}_ext{int(residual)}.png"))
                    if verbose: print(f"Extended crop (full + residual): {slide_id}_{polygon_n}_{int(x)}_{int(y)}_ext{int(residual)}.png")

                else:
                    try:
                        crop_img = rotated_image.crop(x, y, patch_size, patch_size)   
                    except:
                        print("An exception occurred while cropping:")
                        print(f"Rotated image: {rotated_image.width} x {rotated_image.height}; Angle: {degrees(angle)}; Top-left point: ({x},{y})")

                        rotated_image = rotated_image.draw_circle([0,255,0,255], rot_points[0][0], rot_points[0][1], 500, fill=True) # RGBA
                        rotated_image = rotated_image.draw_circle([255,0,0,255], rot_points[1][0], rot_points[1][1], 500, fill=True) # RGBA
                        orig_image = slide.draw_circle([0,255,255,255], pts[0][0], pts[0][1], 500, fill=True) # RGBA
                        orig_image = orig_image.draw_circle([255,0,255,255], pts[1][0], pts[1][1], 500, fill=True) # RGBA

                        rotated_image = rotated_image.thumbnail_image(4096)
                        orig_image = orig_image.thumbnail_image(4096)

                        rotated_image = rotated_image.copy_memory()
                        rotated_image.write_to_file(os.path.join(errors_dir, f'445_rotated_error_{slide_id}_{polygon_n}.png'))
                        orig_image = orig_image.copy_memory()
                        orig_image.write_to_file(os.path.join(errors_dir, f'445_original_error_{slide_id}_{polygon_n}.png'))
                        return
                        # sys.exit("Exit endpoint 445")

                    crop_img = crop_img.copy_memory()
                    if verbose: print (crop_img.width, crop_img.height)
                    crop_img.write_to_file(os.path.join(save_dir, f"{slide_id}_{polygon_n}_{int(x)}_{int(y)}.png"))
                    if verbose: print (f"Full crop: {slide_id}_{polygon_n}_{int(x)}_{int(y)}.png")

            else:
                try:
                    crop_img = rotated_image.crop(x, y, residual, patch_size)
                except:
                    print("An exception occurred while cropping:")
                    print(f"Rotated image: {rotated_image.width} x {rotated_image.height}; Angle: {degrees(angle)}; Top-left point: ({x},{y}); Residual: {residual}")

                    rotated_image = rotated_image.draw_circle([0,255,0,255], rot_points[0][0], rot_points[0][1], 500, fill=True) # RGBA
                    rotated_image = rotated_image.draw_circle([255,0,0,255], rot_points[1][0], rot_points[1][1], 500, fill=True) # RGBA
                    orig_image = slide.draw_circle([0,255,255,255], pts[0][0], pts[0][1], 500, fill=True) # RGBA
                    orig_image = orig_image.draw_circle([255,0,255,255], pts[1][0], pts[1][1], 500, fill=True) # RGBA

                    rotated_image = rotated_image.thumbnail_image(4096)
                    orig_image = orig_image.thumbnail_image(4096)

                    rotated_image = rotated_image.copy_memory()
                    rotated_image.write_to_file(os.path.join(errors_dir, f'472_rotated_error_{slide_id}_{polygon_n}.png'))
                    orig_image = orig_image.copy_memory()
                    orig_image.write_to_file(os.path.join(errors_dir, f'472_original_error_{slide_id}_{polygon_n}.png'))
                    return
                    # sys.exit("Exit endpoint 472")

                crop_img = crop_img.copy_memory()
                crop_img.write_to_file(os.path.join(save_dir, f"{slide_id}_{polygon_n}_{int(x)}_{int(y)}_res{int(residual)}.png"))
                if verbose: print(f"Residual crop: {slide_id}_{polygon_n}_{int(x)}_{int(y)}_res{int(residual)}.png")
    else:
        if verbose: print(f"Crop is not possible with the following setup: \n rotated points: [{rot_points[0][0]} {rot_points[0][1]}] [{rot_points[1][0]} {rot_points[1][1]}] \n rotated img dimensions: {r_width}x{r_height}")

if __name__ == "__main__":
    ### Slide name: (polygon_number_n)
    problematic = {
        # 'slide_name_wo_ext': (poly_n,),
    }

    aperio_slides_fp = sorted(glob("path_to_aperio_slides/*.svs"))
    itn_path = "path_to_itn_annotations/"
    aperio_slides = [os.path.basename(aps) for aps in aperio_slides_fp]
    itn_fp = sorted([os.path.join(itn_path, ap.replace('.svs', '.itn')) for ap in aperio_slides])
    result_fp = {i: (aperio_slides_fp[i], itn_fp[i]) for i in range(len(aperio_slides_fp))}
    ###################
    # Params:
    level = 0
    patch_size = 1000
    
    errors_dir = f'./errors_{patch_size}/'
    savedir = f'path_to_save_folder/preproc_{patch_size}'

    ### logging.DEBUG for more verbose output from pyvips
    logging.basicConfig(filename=f"log_{patch_size}.txt", level=logging.INFO, format="%(asctime)s %(message)s")
    ###################

    for idx in range(len(result_fp.keys())):
        logging.info(f"(516) New slide processing")

        slide_path = result_fp[idx][0] # [key][slide, itn]
        slide = pyvips.Image.new_from_file(slide_path, level=level)
        slide_id = slide.get('aperio.ImageID')
        slide_name = slide.get('aperio.Filename')
        save_path = os.path.join(savedir, slide_name)

        if not (os.path.exists(save_path)):
            rm_n_mkdir(save_path)

            itn_path = result_fp[idx][1]
            config = configparser.ConfigParser()
            config.read(itn_path)
            polys = get_polypons(config['Polygon'])
            
            logging.info(f"(532) Start processing: {slide_path}")
            original_size = (int(slide.get(f'openslide.level[{0}].width')), int(slide.get(f'openslide.level[{0}].height')), 3) ### level 0 - original size
            scaled_size = (slide.width, slide.height, 3)
            logging.info(f"(535) Size scaling: {original_size} -> {scaled_size}")

            logging.info(f"(537) Scaling polygons: {slide_path}")
            scaled_polys = scale_coords(original_size, scaled_size, polys)
            
            logging.info(f"(540) Squashing polygons: {slide_path}")
            squashed_polys = {}
            for k, points in scaled_polys.items():
                print (f"For {k} polygon: {len(points)} points")
                squashed_polys[k] = squash_poly_points(points, patch_size)
                logging.info(f"{k} polygon: {len(points)} -> {len(squashed_polys[k])} points")

            logging.info(f"(548) Cutting patches: {slide_path}")
            
            for k, points in squashed_polys.items():
                if (slide_name in problematic.keys()):
                    if (k in problematic[slide_name]):
                        continue

                # if not (os.path.exists(os.path.join(save_path, str(k)))):
                rm_n_mkdir(os.path.join(save_path, str(k)))
                logging.info(f"(557) {k} polygon: {len(points)} points")
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    logging.info(f"(561) i = {i}/{len(points) - 1}: p1 - {p1}, p2 - {p2} in polygon {k} of {slide_name}")
                    extract_patches_btw_points(slide, slide_id, (p1, p2), k, patch_size, os.path.join(save_path, str(k)), errors_dir)
                    logging.info(f"(563) Extracted patch for {k} polygon in {slide_name}")
            logging.info(f"(564) Finished patch cutting for {slide_name}")
        else:
            logging.info(f"(567) Slide folder <{slide_name}> already exists, skipping...")
