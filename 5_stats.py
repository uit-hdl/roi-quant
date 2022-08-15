from glob import glob
import os
import pandas as pd


if __name__ == "__main__":
    PSIZE = 1000
    INF_DIR = 'hover_inference_directory_with_log_files/'
    count_name = 'Lymphocyte'

    logs = [os.path.basename(x) for x in glob(f"{INF_DIR}/*.log")]
    slides = sorted(list(set([x.split('~')[0] for x in logs])))

    ### Get polygon names and respective patches for each slide ###
    slide_patches = {}
    for slide in slides:
        patches = sorted(list(filter(lambda x: slide in x, logs)))
        polys = sorted(list(set([x.split('~')[1] for x in patches])))
        poly_patches = {}
        for poly in polys:
            p_patches = sorted(list(filter(lambda x: f"~{poly}~" in x, patches)))
            poly_patches[poly] = p_patches        
        slide_patches[slide] = poly_patches

    ### Extract counts for each slide for each polygon ###
    counts = {}
    for slide_name in slide_patches.keys():
        # print (f"Processing: {slide_name}")
        slide_polygons = slide_patches[slide_name]
        polygons_w_counts = {}
        for poly_name in slide_polygons.keys():
            polygon_patches = slide_polygons[poly_name]
            polygon_patches = [os.path.join(INF_DIR, x) for x in polygon_patches]
            poly_counts = 0
            for patch in polygon_patches:
                with open(patch) as f:
                    for line in f:
                        counts_dict = eval(line.split(' : ')[1])
                        if count_name in counts_dict.keys():
                            poly_counts += counts_dict[count_name]
            # Normilize by number of patches in poloygon
            polygons_w_counts[poly_name] = poly_counts / len(polygon_patches)
        counts[slide_name] = polygons_w_counts

    ### Sum all counts from polygons for each slide ###
    counts_summary = {}
    for slide_name in counts.keys():
        slide_sum_counts = 0
        poly_counts_dict = counts[slide_name]
        for polygon in poly_counts_dict.keys():
            slide_sum_counts += poly_counts_dict[polygon]
        counts_summary[slide_name] = slide_sum_counts / len(poly_counts_dict.keys())

    df = pd.DataFrame.from_dict(counts_summary, orient='index', columns=[f'norm_counts_{count_name}'])
    df.to_csv(f'{PSIZE}_norm_counts_{count_name}.csv')

