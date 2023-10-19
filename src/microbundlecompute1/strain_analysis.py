import glob
# import imageio.v2 as imageio
from matplotlib import path as mpl_path
import matplotlib.patheffects as pe
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from microbundlecompute import image_analysis as ia
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
# from PIL import Image  


def box_to_mask(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Given a mask (for dimensions) and a box. Will return a mask of the inside of the box."""
    p = mpl_path.Path([(box[0, 0], box[0, 1]), (box[1, 0], box[1, 1]), (box[2, 0], box[2, 1]), (box[3, 0], box[3, 1])])
    new_mask = np.zeros(mask.shape)
    for rr in range(0, mask.shape[0]):
        for cc in range(0, mask.shape[1]):
            new_mask[rr, cc] = p.contains_points([(rr, cc)])
    return new_mask


def shrink_box(box: np.ndarray, shrink_row: float = 0.1, shrink_col: float = 0.1) -> np.ndarray:
    """Given a box and shrink factors. Will make the box smaller by the given fraction."""
    r0, r1, c0, c1 = ia.box_to_bound(box)
    r0_new, r1_new = ia.shrink_pair(r0, r1, shrink_row)
    c0_new, c1_new = ia.shrink_pair(c0, c1, shrink_col)
    box = ia.bound_to_box(r0_new, r1_new, c0_new, c1_new)
    return box


def create_sub_domains(
    mask: np.ndarray,
    *,
    pillar_clip_fraction: float = 0.5,
    shrink_row: float = 0.1,
    shrink_col: float = 0.1,
    tile_dim_pix: int = 35,
    num_tile_row: int = 5,
    num_tile_col: int = 3,
    tile_style: int = 1,
    clip_columns: bool = True,
    clip_rows: bool = False,
    manual_sub: bool = True,
    sub_extents: List = [53, 195, 18, 397]
) -> List:
    """Given a mask and sub-domain parameters. Will return a list of sub-domains define by 4 box coordinates.
    tile_style = 1 will fit as many square tiles of the given tile_dim_pix size in a grid
    tyle_style = 2 will create a num_tile_row x num_tile_col sized grid and adjust the tile_dim_pix as need
    """
    # clip pillars from the maskS
    mask_removed_pillars = ia.remove_pillar_region(mask, pillar_clip_fraction,clip_columns, clip_rows)

    if True:
        r0_user, r1_user, c0_user, c1_user = [53, 195, 18, 397]
        user_box = ia.bound_to_box(r0_user, r1_user, c0_user, c1_user)
        first_mask = box_to_mask(mask_removed_pillars, user_box)
        first_box_mask = ia.mask_to_box(first_mask)
        box_mask = shrink_box(first_box_mask, shrink_row, shrink_col)
    else:    
        # compute overall box
        first_box_mask = ia.mask_to_box(mask_removed_pillars)
        box_mask = shrink_box(first_box_mask, shrink_row, shrink_col)
    # tile sub-domains
    r0, r1, c0, c1 = ia.box_to_bound(box_mask)

    if tile_style == 1:
        num_tile_row = int(np.floor((r1 - r0) / tile_dim_pix))
        num_tile_col = int(np.floor((c1 - c0) / tile_dim_pix))
    elif tile_style == 2:
        tile_dim_pix = np.min([np.floor((r1 - r0) / num_tile_row), np.floor((c1 - c0) / num_tile_col)])
    # sub-divide into sub-domains
    shrink_row = 1.0 - num_tile_row * tile_dim_pix / (r1 - r0)
    shrink_col = 1.0 - num_tile_col * tile_dim_pix / (c1 - c0)
    box_mask_grid = shrink_box(box_mask, shrink_row, shrink_col)
    r0_box, _, c0_box, _ = ia.box_to_bound(box_mask_grid)
    tile_box_list = []
    for rr in range(0, num_tile_row):
        for cc in range(0, num_tile_col):
            tile_box = ia.bound_to_box(r0_box + rr * tile_dim_pix, r0_box + (rr + 1) * tile_dim_pix, c0_box + cc * tile_dim_pix, c0_box + (cc + 1) * tile_dim_pix)
            tile_box_list.append(tile_box)
    return tile_box_list, tile_dim_pix, num_tile_row, num_tile_col


def isolate_sub_domain_markers(tracker_row_all: List, tracker_col_all: List, sd_box: np.ndarray) -> List:
    """Given tracker row and column arrays and sub-domain box. Will return markers inside sub-domain."""
    sd_tracker_row_all = []
    sd_tracker_col_all = []
    for kk in range(0, len(tracker_row_all)):
        tracker_row = tracker_row_all[kk]
        tracker_col = tracker_col_all[kk]
        sd_tracker_row = []
        sd_tracker_col = []
        for jj in range(0, tracker_row.shape[0]):
            rr = tracker_row[jj, 0]
            cc = tracker_col[jj, 0]
            if ia.is_in_box(sd_box, rr, cc):
                sd_tracker_row.append(tracker_row[jj, :])
                sd_tracker_col.append(tracker_col[jj, :])
        sd_tracker_row = np.asarray(sd_tracker_row)
        sd_tracker_col = np.asarray(sd_tracker_col)
        sd_tracker_row_all.append(sd_tracker_row)
        sd_tracker_col_all.append(sd_tracker_col)
    return sd_tracker_row_all, sd_tracker_col_all


def compute_F_from_Lambda_mat(Lambda_0: np.ndarray, Lambda_t: np.ndarray) -> np.ndarray:
    """Given Lambda (i.e., vectors connecting fiducial marker positions) matricies in positions 0 and t.
    Lambda matricies are units 2 x number of points.
    Will return the average deformation gradient F."""
    term_1 = np.dot(Lambda_t, np.transpose(Lambda_0))
    term_2 = np.linalg.inv(np.dot(Lambda_0, np.transpose(Lambda_0)))
    F = np.dot(term_1, term_2)
    # add in error handling if issues arise here --
    return F


def compute_Lambda_from_pts(row_pos: np.ndarray, col_pos: np.ndarray) -> np.ndarray:
    num_pts = row_pos.shape[0]
    pts_row_col = np.array([row_pos,col_pos])
    ii, jj = np.triu_indices(num_pts, k=1)
    Lambda_mat = pts_row_col[:,ii] - pts_row_col[:,jj] 
    return Lambda_mat

def compute_Lambda_from_newCentroid(row_pos: np.ndarray, col_pos: np.ndarray) -> np.ndarray:
    num_pts = row_pos.shape[0]
    # print("row pos" , row_pos.shape[1])
    pts_row_col = np.array([row_pos,col_pos])
    
    avgRow = np.mean(row_pos)
    avgCol = np.mean(col_pos)
    ii, jj = np.triu_indices(num_pts, k=1)
    # print ("ii", ii.shape)
    # print("jj", jj.shape)
    # print("ptsrowcol", pts_row_col.shape)
    centroidMean = np.array([[avgRow] * np.prod(ii.shape), [avgCol] *  np.prod(ii.shape)])
    # print("centroid mean", centroidMean[:,1])
    # print("first", pts_row_col[:,1])
    Lambda_mat = pts_row_col[:,ii] - centroidMean[:,ii]
    return Lambda_mat

def compute_sub_domain_strain(sd_tracker_row: np.ndarray, sd_tracker_col: np.ndarray) -> np.ndarray:
    """Given tracking point positions. Will return F at every frame with the first frame as the reference."""
    sd_F_list = []
    num_frames = sd_tracker_row.shape[1]
    Lambda_0 = compute_Lambda_from_newCentroid(sd_tracker_row[:, 0], sd_tracker_col[:, 0])
    for kk in range(0, num_frames):
        Lambda_t = compute_Lambda_from_newCentroid(sd_tracker_row[:, kk], sd_tracker_col[:, kk])
        F = compute_F_from_Lambda_mat(Lambda_0, Lambda_t)
        sd_F_list.append(F.reshape((-1, 1)))
    sd_F_arr = np.asarray(sd_F_list)[:, :, 0]
    return sd_F_arr


def get_box_center(box: np.ndarray) -> Union[float, int]:
    """Given a box. Will return the center."""
    center_row = np.mean(box[:, 0])
    center_col = np.mean(box[:, 1])
    return center_row, center_col


def compute_sub_domain_position(sd_tracker_row: List, sd_tracker_col: List, sd_box: np.ndarray) -> List:
    """Given sub-domain displacements and sub-domain box. Will return the box center displacement at every step."""
    sd_row = []
    sd_col = []
    center_row, center_col = get_box_center(sd_box)
    num_frames = sd_tracker_row.shape[1]
    for kk in range(0, num_frames):
        row_pos = center_row + np.mean(sd_tracker_row[:, kk] - sd_tracker_row[:, 0])
        col_pos = center_col + np.mean(sd_tracker_col[:, kk] - sd_tracker_col[:, 0])
        sd_row.append(row_pos)
        sd_col.append(col_pos)
    return np.asarray(sd_row), np.asarray(sd_col)


def compute_sub_domain_position_strain_all(tracker_row_all: List, tracker_col_all: List, sd_box_list: List) -> List:
    """Given all tracker rows, columns, and sub-domain box domains. Will return the markers within each sub-domain box."""
    num_beats = len(tracker_col_all)
    sub_domain_F_all = []
    sub_domain_row_all = []
    sub_domain_col_all = []
    for sd_box in sd_box_list:
        sd_tracker_row_all, sd_tracker_col_all = isolate_sub_domain_markers(tracker_row_all, tracker_col_all, sd_box)
        F_list = []
        sd_row = []
        sd_col = []
        for kk in range(0, num_beats):
            sd_tracker_row = sd_tracker_row_all[kk]
            sd_tracker_col = sd_tracker_col_all[kk]
            # print(sd_tracker_row.shape)
            sd_F_array = compute_sub_domain_strain(sd_tracker_row, sd_tracker_col)
            F_list.append(sd_F_array)
            sd_row_kk, sd_col_kk = compute_sub_domain_position(sd_tracker_row, sd_tracker_col, sd_box)
            sd_row.append(sd_row_kk)
            sd_col.append(sd_col_kk)
        sub_domain_F_all.append(F_list)
        sub_domain_row_all.append(sd_row)
        sub_domain_col_all.append(sd_col)
    return sub_domain_F_all, sub_domain_row_all, sub_domain_col_all


def format_F_for_save(sub_domain_F_all: List) -> List:
    """Given sub_domain_F_all. Will return in a format where F_rr, F_rc, F_cr, and F_cc are formatted for saving."""
    """Subscript r refers to row and c to column: For example F_rr refers to F in the row-row direction."""
    sub_domain_F_rr_all = []
    sub_domain_F_rc_all = []
    sub_domain_F_cr_all = []
    sub_domain_F_cc_all = []
    num_beats = len(sub_domain_F_all[0])
    num_subdomains = len(sub_domain_F_all)
    for kk in range(0, num_beats):
        num_frames = sub_domain_F_all[0][kk].shape[0]
        sub_domain_F_rr = np.zeros((num_subdomains, num_frames))
        sub_domain_F_rc = np.zeros((num_subdomains, num_frames))
        sub_domain_F_cr = np.zeros((num_subdomains, num_frames))
        sub_domain_F_cc = np.zeros((num_subdomains, num_frames))
        for jj in range(0, num_subdomains):
            sub_domain_F_rr[jj, :] = sub_domain_F_all[jj][kk][:, 0]
            sub_domain_F_rc[jj, :] = sub_domain_F_all[jj][kk][:, 1]
            sub_domain_F_cr[jj, :] = sub_domain_F_all[jj][kk][:, 2]
            sub_domain_F_cc[jj, :] = sub_domain_F_all[jj][kk][:, 3]
        sub_domain_F_rr_all.append(sub_domain_F_rr)
        sub_domain_F_rc_all.append(sub_domain_F_rc)
        sub_domain_F_cr_all.append(sub_domain_F_cr)
        sub_domain_F_cc_all.append(sub_domain_F_cc)
    return sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all


def format_sd_row_col_for_save(sub_domain_row_all: List, sub_domain_col_all: List) -> List:
    """Given sub_domain_row_all and sub_domain_col_all. Will return in a format where sub_domain_row and sub_domain_col are formatted for saving."""
    sub_domain_row_all_new = []
    sub_domain_col_all_new = []
    num_beats = len(sub_domain_row_all[0])
    num_subdomains = len(sub_domain_row_all)
    for kk in range(0, num_beats):
        num_frames = sub_domain_row_all[0][kk].shape[0]
        sub_domain_row = np.zeros((num_subdomains, num_frames))
        sub_domain_col = np.zeros((num_subdomains, num_frames))
        for jj in range(0, num_subdomains):
            sub_domain_row[jj, :] = sub_domain_row_all[jj][kk]
            sub_domain_col[jj, :] = sub_domain_col_all[jj][kk]
        sub_domain_row_all_new.append(sub_domain_row)
        sub_domain_col_all_new.append(sub_domain_col)
    return sub_domain_row_all_new, sub_domain_col_all_new


def save_sub_domain_strain(folder_path: Path, sub_domain_F_all: List, sub_domain_row_all: List, sub_domain_col_all: List, strain_sub_domain_info: np.ndarray, *, fname: str = "") -> List:
    """Given results of sub domain strain computation. Will save the strain results."""
    new_path = ia.create_folder(folder_path, "results")
    num_beats = len(sub_domain_row_all[0])
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all = format_F_for_save(sub_domain_F_all)
    sub_domain_row_all, sub_domain_col_all = format_sd_row_col_for_save(sub_domain_row_all, sub_domain_col_all)
    saved_paths = []
    for kk in range(0, num_beats):
        # save the sub domain positions
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_row.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_row_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_col.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_col_all[kk])
        # save the components of the sub domain deformation gradient
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Frr.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_rr_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Frc.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_rc_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Fcr.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_cr_all[kk])
        file_path = new_path.joinpath("strain_" + fname + "_beat%i_Fcc.txt" % (kk)).resolve()
        saved_paths.append(file_path)
        np.savetxt(str(file_path), sub_domain_F_cc_all[kk])
    # save the information about the sub domains (specifically: num_row_tile, num_col_tile, tile_pix_dim)
    file_path = new_path.joinpath("strain_" + fname + "_sub_domain_info.txt").resolve()
    np.savetxt(str(file_path), strain_sub_domain_info)
    saved_paths.append(file_path)
    return saved_paths


def load_sub_domain_strain(folder_path: Path, fname="") -> List:
    """Given folder path. Will load strain results. If there are none, will return an error."""
    res_folder_path = folder_path.joinpath("results").resolve()
    if res_folder_path.exists() is False:
        raise FileNotFoundError("tracking results are not present -- therefore strain results must not be present either")
    file_list = glob.glob(str(res_folder_path) + "/*strain*")
    if len(file_list) == 0:
        raise FileNotFoundError("strain results are not present")
    num_files = len(glob.glob(str(res_folder_path) + "/*strain*beat*.txt"))
    num_beats = int((num_files) / 6)
    sub_domain_F_rr_all = []
    sub_domain_F_rc_all = []
    sub_domain_F_cr_all = []
    sub_domain_F_cc_all = []
    sub_domain_row_all = []
    sub_domain_col_all = []
    for kk in range(0, num_beats):
        sub_domain_F_rr_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Frr.txt" % (kk)))
        sub_domain_F_rc_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Frc.txt" % (kk)))
        sub_domain_F_cr_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Fcr.txt" % (kk)))
        sub_domain_F_cc_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_Fcc.txt" % (kk)))
        sub_domain_row_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_row.txt" % (kk)))
        sub_domain_col_all.append(np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_beat%i_col.txt" % (kk)))
    strain_info = np.loadtxt(str(res_folder_path) + "/strain_" + fname + "_sub_domain_info.txt")
    info = np.loadtxt(str(res_folder_path) + "/info.txt")
    info_reshape = np.reshape(info, (-1, 3))
    return sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info_reshape, strain_info


def F_to_E(F_rr: float, F_rc: float, F_cr: float, F_cc: float) -> float:
    """Given F, will compute E_cc (E_column_column) for visualization."""
    F_arr = np.asarray([[F_rr, F_rc], [F_cr, F_cc]])
    C = np.dot(F_arr.T, F_arr)
    E = 0.5 * (C - np.eye(2))
    return E[1, 1], E[1,0], E[0,0]


def F_to_E_all(sub_domain_F_rr_all: List, sub_domain_F_rc_all: List, sub_domain_F_cr_all: List, sub_domain_F_cc_all: List) -> List:
    sub_domain_Ecc_all = []
    sub_domain_Ecr_all = []
    sub_domain_Err_all = []
    num_beats = len(sub_domain_F_rr_all)
    num_sub_domains = sub_domain_F_rr_all[0].shape[0]
    for kk in range(0, num_beats):
        num_frames = sub_domain_F_rr_all[kk].shape[1]
        Ecc_arr = np.zeros((num_sub_domains, num_frames))
        Ecr_arr = np.zeros((num_sub_domains, num_frames))
        Err_arr = np.zeros((num_sub_domains, num_frames))
        for jj in range(0, num_sub_domains):
            for ii in range(0, num_frames):
                F_rr = sub_domain_F_rr_all[kk][jj, ii]
                F_rc = sub_domain_F_rc_all[kk][jj, ii]
                F_cr = sub_domain_F_cr_all[kk][jj, ii]
                F_cc = sub_domain_F_cc_all[kk][jj, ii]
                Ecc_arr[jj, ii], Ecr_arr[jj, ii], Err_arr[jj, ii]= F_to_E(F_rr, F_rc, F_cr, F_cc)
        sub_domain_Ecc_all.append(Ecc_arr)
        sub_domain_Ecr_all.append(Ecr_arr)
        sub_domain_Err_all.append(Err_arr)
    return sub_domain_Ecc_all, sub_domain_Ecr_all, sub_domain_Err_all


def get_text_str(row: int, col: int) -> str:
    """Given the row and the column of a location. Will return the row and column position string."""
    test_str = "%s%i" % (chr(row + 65), col + 1)
    return test_str


def png_sub_domains_numbered(
    folder_path: Path,
    example_tiff: np.ndarray,
    sub_domain_row: List,
    sub_domain_col: List,
    sub_domain_side: Union[float, int],
    num_sd_row: int,
    num_sd_col: int,
    fname: str = "strain_",
    col_map: object = plt.cm.rainbow
) -> List:
    """Given information to visualize the sub-domains. Will plot the subdomains and label them.
    Rows are labeled A, B, C, etc. -- columns are labeled 1, 2, 3, etc. """
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    img_path = pngs_folder_path.joinpath(fname + "sub_domain_key.pdf").resolve()
    plt.figure()
    plt.imshow(example_tiff, cmap=plt.cm.gray)
    sds = sub_domain_side / 2.0
    for cc in range(0, num_sd_col):
        for rr in range(0, num_sd_row):
            idx = rr * num_sd_col + cc
            center_row = sub_domain_row[idx, 0]
            center_col = sub_domain_col[idx, 0]
            text_str = get_text_str(rr, cc)
            plt.text(center_col, center_row, text_str, color=col_map(idx / (num_sd_col * num_sd_row)), fontsize= int(sub_domain_side/3) ,horizontalalignment="center", verticalalignment="center",path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()])
            corners_rr = [center_row - sds, center_row - sds, center_row + sds, center_row + sds, center_row - sds]
            corners_cc = [center_col - sds, center_col + sds, center_col + sds, center_col - sds, center_col - sds]
            plt.plot(corners_cc, corners_rr, "k-", linewidth=1)
    plt.axis("off")
    plt.savefig(str(img_path),format='pdf')
    plt.close()
    return img_path


def png_sub_domain_strain_timeseries_all(
    folder_path: Path,
    sub_domain_strain_all: List,
    num_sd_row: int,
    num_sd_col: int,
    output: str,
    col_map: object = plt.cm.rainbow,
    *,
    fname: str = "strain_timeseries_Ecc",
    xlabel: str = "frame",
    ylabel: str = "strain Ecc"
) -> List:
    """Given strain timeseries. Will plot all timeseries on the same axis."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err")           
    num_beats = len(sub_domain_strain_all)
    path_list = []
    for kk in range(0, num_beats):
        plt.figure()
        ax = plt.subplot(111)
        sub_domain_strain = sub_domain_strain_all[kk]
        for cc in range(0, num_sd_col):
            for rr in range(0, num_sd_row):
                lab = get_text_str(rr, cc)
                idx = rr * num_sd_col + cc
                ax.plot(sub_domain_strain[idx], label=lab, color=col_map(idx/(num_sd_col * num_sd_row)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("beat %i" % (kk))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=num_sd_col)
        img_path = pngs_folder_path.joinpath(fname + "_beat%i.pdf" % (kk)).resolve()
        plt.savefig(str(img_path), format='pdf')
        plt.close()
        path_list.append(img_path)
    return path_list

def png_sub_domain_strain_timeseries_all_altered(
    folder_path: Path,
    sub_domain_strain_all: List,
    sub_domain_strain_analyt: List,
    sub_domain_strain_elastix: List,
    num_sd_row: int,
    num_sd_col: int,
    output: str,
    col_map: object = plt.cm.rainbow,
    *,
    fname: str = "strain_timeseries_Ecc",
    xlabel: str = "frame",
    ylabel: str = "strain Ecc"
) -> List:
    """Given strain timeseries. Will plot all timeseries on the same axis."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err")           
    num_beats = len(sub_domain_strain_all)
    path_list = []
    sd_ecc_analyticalFinite = np.load("files/files_run_code/Subdomain_ECC_analytical_finite.npy")
    
    for kk in range(0, num_beats):
        plt.figure()
        # fig, axs = plt.subplots(2, 2, sharey = "all", figsize=[9, 9])
     
        # idx1 = 0 * num_sd_col + 3
        # idx2 = 2 * num_sd_col + 4
        # idx3 = 1 * num_sd_col + 2
        # idx4 = 2 * num_sd_col + 6
        
        sub_domain_strain_OF = sub_domain_strain_all[kk]
        sub_domain_strain_OF = sub_domain_strain_OF[:, :-1]
        sub_domain_strain_Analytical = sub_domain_strain_analyt[kk]
        sub_domain_strain_Analytical = sub_domain_strain_Analytical[:, 2:-10]
        sub_domain_strain_Elastix = sub_domain_strain_elastix[kk]
        sub_domain_strain_Elastix = sub_domain_strain_Elastix[:, 1:-10]
        sd_ecc_analyticalFinite = sd_ecc_analyticalFinite[:, 1:-10]

        
        print(sub_domain_strain_Analytical[20,10])
        # print("OF", sub_domain_strain_OF.shape)
        # print("analyt", sub_domain_strain_Analytical.shape)
        # print("elastix", sub_domain_strain_Elastix.shape)
        
        fig, axs = plt.subplots(3, 8, sharex=True, sharey=True, num=1, clear=True)
        for i in range(len(sub_domain_strain_OF)):
            r, c = np.unravel_index(i, (3,8))
            axs[r,c].plot(sub_domain_strain_OF[i], 'b:', linewidth = 0.5, label = "Optical Flow")
            axs[r,c].plot(sub_domain_strain_Analytical[i], 'r-', linewidth = 0.5, label = "Analytical")
            axs[r,c].plot(sub_domain_strain_Elastix[i], 'c--', linewidth = 0.5,label = "Elastix" )
            axs[r,c].plot(sd_ecc_analyticalFinite[i], 'm-.', linewidth = 0.5, label = "Analytical Finite")

        # axs[0, 0].plot(sub_domain_strain_OF[idx1, :-1], label="Optical Flow", color=col_map(idx1/(num_sd_col * num_sd_row)), linestyle='dotted')
        # axs[0, 0].plot(sub_domain_strain_Analytical[idx1, 2:-10], label="Analytical", color=col_map(idx2/(num_sd_col * num_sd_row)))
        # axs[0, 0].plot(sub_domain_strain_Elastix[idx1, 1:-10], label="Elastix", color=col_map(idx3/(num_sd_col * num_sd_row)), ls='--')
        # axs[0, 0].plot(sd_ecc_analyticalFinite[idx1, 1:-10], label="Analytical Finite", color=col_map(idx4/(num_sd_col * num_sd_row)), ls='-.')

        # axs[0, 0].set_xlabel(xlabel)
        # axs[0, 0].set_ylabel(ylabel)
        # axs[0, 0].set_title("a4")
        # #########################
        # axs[0, 1].plot(sub_domain_strain_OF[idx2, :], color=col_map(idx1/(num_sd_col * num_sd_row)), linestyle='dotted')
        # axs[0, 1].plot(sub_domain_strain_Analytical[idx2, 2:-10], color=col_map(idx2/(num_sd_col * num_sd_row)))
        # axs[0, 1].plot(sub_domain_strain_Elastix[idx2, 1:-10], color=col_map(idx3/(num_sd_col * num_sd_row)), ls='--')
        # axs[0, 1].plot(sd_ecc_analyticalFinite[idx2, 1:-10], color=col_map(idx4/(num_sd_col * num_sd_row)), ls='-.')

        
        # axs[0, 1].set_xlabel(xlabel)
        # axs[0, 1].set_ylabel(ylabel)
        # axs[0, 1].set_title("c5")
        # #######################
        # axs[1, 0].plot(sub_domain_strain_OF[idx3, :-1], color=col_map(idx1/(num_sd_col * num_sd_row)), linestyle='dotted')
        # axs[1, 0].plot(sub_domain_strain_Analytical[idx3, 2:-10],color=col_map(idx2/(num_sd_col * num_sd_row)))
        # axs[1, 0].plot(sub_domain_strain_Elastix[idx3, 1:-10], color=col_map(idx3/(num_sd_col * num_sd_row)), ls='--')
        # axs[1, 0].plot(sd_ecc_analyticalFinite[idx3, 1:-10], color=col_map(idx4/(num_sd_col * num_sd_row)), ls='-.')

        # axs[1, 0].set_xlabel(xlabel)
        # axs[1, 0].set_ylabel(ylabel)
        # axs[1, 0].set_title("b3")
        # ###############################
        # axs[1, 1].plot(sub_domain_strain_OF[idx4, :-1], color=col_map(idx1/(num_sd_col * num_sd_row)), linestyle='dotted')
        # axs[1, 1].plot(sub_domain_strain_Analytical[idx4, 2:-10], color=col_map(idx2/(num_sd_col * num_sd_row)))
        # axs[1, 1].plot(sub_domain_strain_Elastix[idx4, 1:-10], color=col_map(idx3/(num_sd_col * num_sd_row)), ls='--')
        # axs[1, 1].plot(sd_ecc_analyticalFinite[idx4, 1:-10], color=col_map(idx4/(num_sd_col * num_sd_row)), ls='-.')

        # axs[1, 1].set_xlabel(xlabel)
        # axs[1, 1].set_ylabel(ylabel)
        # axs[1, 1].set_title("c7")
        # # box = axs[0, 0].get_position()
        # # axs[0, 0].set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])
        # ax.legend(loc='upper center', ncol=num_sd_col)
        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)        
        img_path = pngs_folder_path.joinpath(fname + "_beat%i.png" % (kk)).resolve()
        plt.savefig(str(img_path), format='png', dpi = 180)
        plt.close()
        path_list.append(img_path)
    return path_list

def compute_min_max_strain (
    sub_domain_E_all,
    info: np.ndarray
) -> float: 
    """Given tracking results. Will find the minimum and maximum displacement over all beats."""
    num_beats = info.shape[0]
    min_E_all,max_E_all = [],[]
    for beat in range(0, num_beats):
        E = sub_domain_E_all[beat]
        min_E_all.append(np.min(E))
        max_E_all.append(np.max(E))
    min_E_c = np.min(min_E_all)
    max_E_c = np.max(max_E_all)
    
    return min_E_c, max_E_c
   

def pngs_sub_domain_strain(
    folder_path: Path,
    tiff_list: List,
    sub_domain_row_all: List,
    sub_domain_col_all: List,
    sub_domain_E_all: List,
    sub_domain_side: Union[float, int],
    info: np.ndarray,
    output: str,
    col_min: Union[float, int] = -0.025,
    col_max: Union[float, int] = 0.025,
    col_map: object = plt.cm.RdBu,
    fname: str = "strain",
    save_eps: bool = False
) -> List:
    """Given sub domain strain results. Will create pngs."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
        E_label = "Ecc"
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
        E_label = "Ecr"
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err") 
        E_label = "Err"
        
    path_list = []
    num_beats = info.shape[0]
    for beat in range(0, num_beats):
        tracker_row = sub_domain_row_all[beat]
        tracker_col = sub_domain_col_all[beat]
        E = sub_domain_E_all[beat]
        start_idx = int(info[beat, 1])
        end_idx = int(info[beat, 2])
        for kk in range(start_idx, end_idx):
            plt.figure()
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk - start_idx
            plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=E[:, jj], s=50, cmap=col_map, vmin=-0.09, vmax=0.0)
            # illustrate sub-domain borders
            sds = sub_domain_side / 2.0
            for ii in range(0, tracker_col.shape[0]):
                corners_rr = [tracker_row[ii, jj] - sds, tracker_row[ii, jj] - sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] - sds]
                corners_cc = [tracker_col[ii, jj] - sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] - sds, tracker_col[ii, jj] - sds]
                plt.plot(corners_cc, corners_rr, "k-", linewidth=0.1)
                
            plt.title("frame %i, beat %i, %s" % (kk, beat, E_label))
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label("%s strain (image column axis)"%(E_label), rotation=270)
            plt.axis("off")
            path = pngs_folder_path.joinpath("%04d_" % (kk) + fname + ".png").resolve()
            plt.savefig(str(path))
            if save_eps:
                plt.savefig(str(path)[0:-4]+'.eps',format='eps')
            plt.close()
            path_list.append(path)
    return path_list


def pngs_sub_domain_error(
    folder_path: Path,
    tiff_list: List,
    sub_domain_row_all: List,
    sub_domain_col_all: List,
    sub_domain_E_all: List,
    sub_domain_side: Union[float, int],
    info: np.ndarray,
    output: str,
    col_min: Union[float, int] = -0.025,
    col_max: Union[float, int] = 0.025,
    col_map: object = plt.cm.RdBu,
    fname: str = "strain",
    save_eps: bool = False
) -> List:
    """Given sub domain strain results. Will create pngs."""
    vis_folder_path = ia.create_folder(folder_path, "visualizations")
    main_pngs_folder_path = ia.create_folder(vis_folder_path, "strain_pngs")
    
    if output == "Ecc":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecc")
        E_label = "Ecc"
    elif output == "Ecr":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Ecr")  
        E_label = "Ecr"
    elif output == "Err":
        pngs_folder_path = ia.create_folder(main_pngs_folder_path, "Err") 
        E_label = "Err"
        
    path_list = []
    num_beats = info.shape[0]
    for beat in range(0, num_beats):
        tracker_row = sub_domain_row_all[beat]
        tracker_col = sub_domain_col_all[beat]
        E = sub_domain_E_all
        start_idx = 25
        end_idx = 26
        for kk in range(start_idx, end_idx):
            ax = plt.figure()
            # ax.text(5,10, "Average Error: %d" % np.mean(E), verticalalignment='center')
            # print(E)
            print(E_label, np.mean(np.abs(E)))
            plt.imshow(tiff_list[kk], cmap=plt.cm.gray)
            jj = kk
            plt.scatter(tracker_col[:, jj], tracker_row[:, jj], c=E[:], s=50, cmap=col_map, vmin = -0.2, vmax = 0.2)
            # illustrate sub-domain borders
            sds = sub_domain_side / 2.0
            for ii in range(0, tracker_col.shape[0]):
                corners_rr = [tracker_row[ii, jj] - sds, tracker_row[ii, jj] - sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] + sds, tracker_row[ii, jj] - sds]
                corners_cc = [tracker_col[ii, jj] - sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] + sds, tracker_col[ii, jj] - sds, tracker_col[ii, jj] - sds]
                plt.plot(corners_cc, corners_rr, "k-", linewidth=0.1)
                
            plt.title("frame %i, beat %i, %s" % (kk, beat, E_label))
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label("%s error (image column axis)"%(E_label), rotation=270)
            plt.axis("off")
            path = pngs_folder_path.joinpath("%04d_" % (kk) + fname + ".png").resolve()
            plt.savefig(str(path), dpi = 180)
            if save_eps:
                plt.savefig(str(path)[0:-4]+'.eps',format='eps')
            plt.close()
            path_list.append(path)
    return path_list

def create_gif(folder_path: Path, png_path_list: List, fname="sub_domain_strain") -> Path:
    """Given the pngs path list. Will create a gif."""
    img_list = []
    img = plt.imread(png_path_list[0])
    img_r, img_c,_ = img.shape
    fig, ax = plt.subplots(figsize=(img_c/100,img_r/100))
    plt.axis('off') 
    plt.tight_layout(pad=0.08, h_pad=None, w_pad=None, rect=None)
    for pa in png_path_list:
        img = ax.imshow(plt.imread(pa),animated=True)
        img_list.append([img])
    fn_gif = fname + ".gif"
    gif_path = folder_path.joinpath("visualizations").resolve().joinpath(fn_gif).resolve()
    ani = animation.ArtistAnimation(fig, img_list,interval=100)
    ani.save(gif_path,dpi=100)
    return gif_path


def run_sub_domain_strain_analysis(
    folder_path: Path,
    pillar_clip_fraction: float = 0.5,
    shrink_row: float = 0.1,
    shrink_col: float = 0.1,
    tile_dim_pix: int = 40,
    num_tile_row: int = 5,
    num_tile_col: int = 3,
    tile_style: int = 1,
    is_rotated: bool = True,
    clip_columns: bool = True,
    clip_rows: bool = False,
    manual_sub: bool = False,
    sub_extents: List = None,
    *,
    save_fname: str = ""
) -> List:
    """Given a folder path. Will perform strain analysis and save results as text files.
    Note that this function assumes that we have already run the tracking portion of the code."""
    # read images and mask file
    mask_file_path = folder_path.joinpath("masks").resolve().joinpath("tissue_mask.txt").resolve()
    mask = ia.read_txt_as_mask(mask_file_path)

    # load tracking results
    tracker_row_all, tracker_col_all, _, _ = ia.load_tracking_results(folder_path=folder_path)
    
    if is_rotated:
        # rotate tracking results
        (center_row, center_col, rot_mat, ang, vec) = ia.get_rotation_info(mask=mask)
        rot_tracker_row_all, rot_tracker_col_all = ia.rotate_pts_all(tracker_row_all, tracker_col_all, rot_mat, center_row, center_col)
        if abs(ang) < 0.96*np.pi:
            square = ia.check_square_image(mask)
            if square == False:
                # pad mask
                padded_mask, translate_r, translate_c = ia.pad_img_to_square(mask)
                # translate center of rotation 
                trans_center_row, trans_center_col = ia.translate_points(center_row,center_col,translate_r,translate_c)
                # rotate mask
                rot_mask  = ia.rot_image(padded_mask, trans_center_row, trans_center_col, ang)
                # translate rotated results 
                rot_tracker_row_all_pad, rot_tracker_col_all_pad = ia.translate_pts_all(rot_tracker_row_all,rot_tracker_col_all,translate_r,translate_c)
            else:
                rot_mask  = ia.rot_image(mask, center_row, center_col, ang)
                rot_tracker_row_all_pad, rot_tracker_col_all_pad = rot_tracker_row_all, rot_tracker_col_all        
        else:
            rot_mask = mask
            rot_tracker_row_all_pad, rot_tracker_col_all_pad = tracker_row_all, tracker_col_all
        # create sub-domains
        sub_domain_box_list, tile_dim_pix, num_tile_row, num_tile_col = create_sub_domains(rot_mask, pillar_clip_fraction=pillar_clip_fraction, shrink_row=shrink_row, shrink_col=shrink_col, tile_dim_pix=tile_dim_pix, num_tile_row=num_tile_row, num_tile_col=num_tile_col, tile_style=tile_style,clip_columns=clip_columns,clip_rows=clip_rows,manual_sub= manual_sub,sub_extents=sub_extents)
        np.save("subdomainBoxes.npy", sub_domain_box_list)
        # print(len(sub_domain_box_list))

        # compute strain in each sub-domain
        # sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = compute_sub_domain_position_strain_all(rot_tracker_row_all_pad, rot_tracker_col_all_pad, sub_domain_box_list) 
        # save the sub-domain strains
        # strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix], [center_row, center_col], [vec[0], vec[1]]])
    else:
        # create sub-domains
        sub_domain_box_list, tile_dim_pix, num_tile_row, num_tile_col = create_sub_domains(mask, pillar_clip_fraction=pillar_clip_fraction, shrink_row=shrink_row, shrink_col=shrink_col, tile_dim_pix=tile_dim_pix, num_tile_row=num_tile_row, num_tile_col=num_tile_col, tile_style=tile_style,clip_columns=clip_columns,clip_rows=clip_rows,manual_sub= manual_sub,sub_extents=sub_extents)
        np.save("subdomainBoxes.npy", sub_domain_box_list)

        # print(len(sub_domain_box_list))
        # print(sub_domain_box_list[0].shape)


        # compute strain in each sub-domain
        # print(tracker_row_all[0].shape)
        # print(sub_domain_box_list)
        # sub_domain_F_all, sub_domain_row_all, sub_domain_col_all = compute_sub_domain_position_strain_all(tracker_row_all, tracker_col_all, sub_domain_box_list)
        # save the sub-domain strains
        # strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix]])
   
    # saved_paths = save_sub_domain_strain(folder_path, sub_domain_F_all, sub_domain_row_all, sub_domain_col_all, strain_sub_domain_info, fname=save_fname)
    # return saved_paths


def visualize_sub_domain_strain(
    folder_path: Path,
    automatic_color_constraint: bool = True,
    col_min: Union[float, int] = -0.025,
    col_max: Union[float, int] = 0.025,
    col_map: object = plt.cm.RdBu,
    fname: str = "strain",
    is_rotated: bool = True,
) -> List:
    """Given a folder path where strain analysis has already been run. Will save visualizations."""
    # read image files
    movie_folder_path = folder_path.joinpath("movie").resolve()
    name_list_path = ia.image_folder_to_path_list(movie_folder_path)
    tiff_list = ia.read_all_tiff(name_list_path)
    # load the strain results
    sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all, sub_domain_row_all, sub_domain_col_all, info, strain_info = load_sub_domain_strain(folder_path)
    
    num_sd_row = int(strain_info[0, 0])
    num_sd_col = int(strain_info[0, 1])
    sub_domain_side = strain_info[1, 0]
    
    if is_rotated:
        # rotate the background tiff to match the strain results
        center_row = strain_info[2, 0]
        center_col = strain_info[2, 1]
        vec = strain_info[3, :]
        # strain_sub_domain_info = np.asarray([[num_tile_row, num_tile_col], [tile_dim_pix, tile_dim_pix], [center_row, center_col], [vec[0], vec[1]]])
        (_, ang) = ia.rot_vec_to_rot_mat_and_angle(vec)
        if abs(ang) < 0.96*np.pi:
            square = ia.check_square_image(tiff_list[0])
            if square == False:
                # pad image list
                padded_tiff_list, translate_r, translate_c = ia.pad_all_imgs_to_square(tiff_list)
                # translate center of rotation
                trans_center_row, trans_center_col = ia.translate_points(center_row,center_col,translate_r,translate_c)
                # rotate all images in list
                tiff_list = ia.rotate_imgs_all(padded_tiff_list, ang, trans_center_row, trans_center_col)
            else:
                tiff_list = ia.rotate_imgs_all(tiff_list, ang, center_row, center_col)
        else:
            tiff_list = tiff_list
            
    maxFrame = 24;
    # convert the strain results to Ecc for plotting
    sub_domain_Ecc_all, sub_domain_Ecr_all, sub_domain_Err_all = F_to_E_all(sub_domain_F_rr_all, sub_domain_F_rc_all, sub_domain_F_cr_all, sub_domain_F_cc_all)
    
####### SAVE STRAIN FROM ELSEWHERE ##########################################################################
#Dont need to edit anything here
    # np.save("files/files_run_code/Subdomain_ECC_elastix", sub_domain_Ecc_all)
    # np.save("files/files_run_code/Subdomain_ECR_elastix", sub_domain_Ecr_all)
    # np.save("files/files_run_code/Subdomain_ERR_elastix", sub_domain_Err_all)
    
    # np.save("files/files_run_code/Subdomain_ECC_analytical", sub_domain_Ecc_all)
    # np.save("files/files_run_code/Subdomain_ECR_analytical", sub_domain_Ecr_all)
    # np.save("files/files_run_code/Subdomain_ERR_analytical", sub_domain_Err_all)


######### LOAD STRAIN ########################################################################################
    sd_ecc_elastix = np.load("files/files_run_code/Subdomain_ECC_elastix.npy")
    sd_ecr_elastix = np.load("files/files_run_code/Subdomain_ECR_elastix.npy")
    sd_err_elastix = np.load("files/files_run_code/Subdomain_ERR_elastix.npy")

    sd_ecc_analytical = np.load("files/files_run_code/Subdomain_ECC_analytical.npy")
    sd_ecr_analytical = np.load("files/files_run_code/Subdomain_ECR_analytical.npy")
    sd_err_analytical = np.load("files/files_run_code/Subdomain_ERR_analytical.npy")

###### REGULAR VISUALIZATION ############################################################################################
    if automatic_color_constraint:
        # find limits of colormap
        clim_Ecc_min, clim_Ecc_max = compute_min_max_strain(sub_domain_Ecc_all,info)
        clim_Ecr_min, clim_Ecr_max = compute_min_max_strain(sub_domain_Ecr_all,info)
        clim_Err_min, clim_Err_max = compute_min_max_strain(sub_domain_Err_all,info)
        
    # png_path_list_Ecc = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Ecc_all, sub_domain_side, info, "Ecc", clim_Ecc_min, clim_Ecc_max, col_map, fname, save_eps=False)
    # png_path_list_Ecr = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Ecr_all, sub_domain_side, info, "Ecr", clim_Ecr_min, clim_Ecr_max, col_map, fname, save_eps=False)
    # png_path_list_Err = pngs_sub_domain_strain(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, sub_domain_Err_all, sub_domain_side, info, "Err", clim_Err_min, clim_Err_max, col_map, fname, save_eps=False)    
    
    # # create gif
    # gif_path_Ecc = create_gif(folder_path, png_path_list_Ecc, fname="sub_domain_strain_Ecc")
    # gif_path_Ecr = create_gif(folder_path, png_path_list_Ecr, fname="sub_domain_strain_Ecr")
    # gif_path_Err = create_gif(folder_path, png_path_list_Err, fname="sub_domain_strain_Err")
    
    # # create subdomain locations legend
    # loc_legend_path = png_sub_domains_numbered(folder_path, tiff_list[0], sub_domain_row_all[0], sub_domain_col_all[0], sub_domain_side, num_sd_row, num_sd_col)
    
    # # create subdomain strain timeseries plots edited
    timeseries_path_list_Ecc = png_sub_domain_strain_timeseries_all_altered(folder_path, sub_domain_Ecc_all, sd_ecc_analytical, sd_ecc_elastix, num_sd_row, num_sd_col, output="Ecc", fname="strain_timeseries_Ecc", ylabel="strain Ecc")
    timeseries_path_list_Ecr = png_sub_domain_strain_timeseries_all_altered(folder_path, sub_domain_Ecr_all, sd_ecr_analytical, sd_ecr_elastix, num_sd_row, num_sd_col, output="Ecr", fname="strain_timeseries_Ecr", ylabel="strain Ecr")
    timeseries_path_list_Err = png_sub_domain_strain_timeseries_all_altered(folder_path, sub_domain_Err_all, sd_err_analytical, sd_err_elastix, num_sd_row, num_sd_col, output="Err", fname="strain_timeseries_Err", ylabel="strain Err")

    # # Regular time series plots
    # timeseries_path_list_Ecc = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Ecc_all, num_sd_row, num_sd_col, output="Ecc", fname="strain_timeseries_Ecc", ylabel="strain Ecc")
    # timeseries_path_list_Ecr = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Ecr_all, num_sd_row, num_sd_col, output="Ecr", fname="strain_timeseries_Ecr", ylabel="strain Ecr")
    # timeseries_path_list_Err = png_sub_domain_strain_timeseries_all(folder_path, sub_domain_Err_all, num_sd_row, num_sd_col, output="Err", fname="strain_timeseries_Err", ylabel="strain Err")


###### ERROR CALC AND VISUALIZATION BELOW #####################################################################################
    
    # Elastix error
    # elastix_error_ecc = (sd_ecc_elastix[0][:,maxFrame] - sd_ecc_analytical[0][:,maxFrame])/sd_ecc_analytical[0][:,maxFrame]
    # elastix_error_ecr = (sd_ecr_elastix[0][:,maxFrame] - sd_ecr_analytical[0][:,maxFrame])/sd_ecr_analytical[0][:,maxFrame]
    # elastix_error_err = (sd_err_elastix[0][:,maxFrame] - sd_err_analytical[0][:,maxFrame])/sd_err_analytical[0][:,maxFrame]
    
    # # Optical Flow error
    # of_error_ecc = (sub_domain_Ecc_all[0][:,maxFrame] - sd_ecc_analytical[0][:,maxFrame])/sd_ecc_analytical[0][:,maxFrame]
    # of_error_ecr = (sub_domain_Ecr_all[0][:,maxFrame] - sd_ecr_analytical[0][:,maxFrame])/sd_ecr_analytical[0][:,maxFrame]
    # of_error_err = (sub_domain_Err_all[0][:,maxFrame] - sd_err_analytical[0][:,maxFrame])/sd_err_analytical[0][:,maxFrame]
    
     
    # # Elastix visualization
    # png_path_list_Ecc = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, elastix_error_ecc, sub_domain_side, info, "Ecc", clim_Ecc_min, clim_Ecc_max, col_map, fname = "Strain_error_elastix", save_eps=False)
    # png_path_list_Ecr = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, elastix_error_ecr, sub_domain_side, info, "Ecr", clim_Ecr_min, clim_Ecr_max, col_map, fname = "Strain_error_elastix", save_eps=False)
    # png_path_list_Err = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, elastix_error_err, sub_domain_side, info, "Err", clim_Err_min, clim_Err_max, col_map, fname = "Strain_error_elastix", save_eps=False)    

    # # Optical Flow visualization
    # png_path_list_Ecc = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, of_error_ecc, sub_domain_side, info, "Ecc", clim_Ecc_min, clim_Ecc_max, col_map, fname = "Strain_error_OF", save_eps=False)
    # png_path_list_Ecr = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, of_error_ecr, sub_domain_side, info, "Ecr", clim_Ecr_min, clim_Ecr_max, col_map, fname = "Strain_error_OF", save_eps=False)
    # png_path_list_Err = pngs_sub_domain_error(folder_path, tiff_list, sub_domain_row_all, sub_domain_col_all, of_error_err, sub_domain_side, info, "Err", clim_Err_min, clim_Err_max, col_map, fname = "Strain_error_OF", save_eps=False)    
   
    # return png_path_list_Ecc, png_path_list_Ecr, png_path_list_Err, gif_path_Ecc, gif_path_Ecr, gif_path_Err, loc_legend_path, timeseries_path_list_Ecc, timeseries_path_list_Ecr, timeseries_path_list_Err



