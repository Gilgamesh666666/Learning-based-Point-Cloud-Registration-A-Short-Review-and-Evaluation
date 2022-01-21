import os
import numpy as np
import open3d as o3d
import time
def make_o3d_PointCloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd
def make_o3d_Feature(feat):
    # [n, c]
    feature = o3d.registration.Feature()
    feature.data = feat.T
    return feature

def apply_trans(pts:np.array, trans:np.array, with_translate=True):
    from scipy.spatial.transform import Rotation as Rotation
    # (n, 3) (4, 4)
    R = trans[:3, :3]
    t = trans[:3, 3]
    r = Rotation.from_matrix(R)
    if with_translate:
        return r.apply(pts) + t[np.newaxis,:]
    else:
        return r.apply(pts)

def compute_overlap_ratio(source, target, trans, voxel_size):
    # [n1, 3], [n2, 3], [4, 4]
    from scipy.spatial import cKDTree
    align_source = apply_trans(source, trans)
    tree = cKDTree(align_source)
    dist, ind = tree.query(target, k=1)
    mask = dist < voxel_size
    overlap_ratio = ind[mask].shape[0]/max(source.shape[0], target.shape[0])
    return overlap_ratio

def find_correspondence_one_pair(feat1, feat2):
    # [n1, c], [n2, c]
    # [n1, n2]
    diff = np.linalg.norm(feat1, axis=1, keepdims=True) + np.linalg.norm(feat2, axis=1, keepdims=True).T - 2 * np.dot(feat1, feat2.T)
    corr_idx1 = np.argmin(diff, axis=1) # [n1]
    corr_idx2 = np.argmin(diff, axis=0) # [n2]
    mask = (corr_idx2[corr_idx1] == np.arange(corr_idx1.shape[0])) # [n1]
    
    idx2 = corr_idx1[mask] # 2
    idx1 = np.arange(corr_idx1.shape[0])[mask] # 1
    return idx1, idx2

def est_trans_one_pair(xyz, xyz_corr, feat, feat_corr, voxel_size, func='ransac'): # ransac/teaserpp
    # [n1, 3], [n1, 3], [n1, c], [n1, c]
    # record reg_time
    # xyz1和xyz2的点数可能不一样，但是每个点一定要有feat
    # source -> target
    if func=='ransac':
        assert voxel_size > 0
        source = make_o3d_PointCloud(xyz)
        target = make_o3d_PointCloud(xyz_corr)
        feature_source = make_o3d_Feature(feat)
        feature_target = make_o3d_Feature(feat_corr)
        start = time.time()
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source, target, feature_source, feature_target, voxel_size,
            o3d.registration.TransformationEstimationPointToPoint(False),3,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(voxel_size)],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000)) # max_iter, max_val:只有过了check才会去validation
        end = time.time()
        trans = result.transformation
        reg_time = end - start
    elif func=='fgr':
        print('use FGR')
        source = make_o3d_PointCloud(xyz)
        target = make_o3d_PointCloud(xyz_corr)
        feature_source = make_o3d_Feature(feat)
        feature_target = make_o3d_Feature(feat_corr)
        start = time.time()
        reg = o3d.registration.registration_fast_based_on_feature_matching(source, target, feature_source, feature_target, o3d.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel_size))
        end = time.time()
        trans = reg.transformation
        reg_time = end - start
    elif func=='teaser':
        NOISE_BOUND = 0.05
        try:
            import teaserpp_python
        except:
            print('please install TEASER++')
            exit(-1)
        def compose_mat4_from_teaserpp_solution(solution):
            """
            Compose a 4-by-4 matrix from teaserpp solution
            """
            s = solution.scale
            rotR = solution.rotation
            t = solution.translation
            T = np.eye(4)
            T[0:3, 3] = t
            R = np.eye(4)
            R[0:3, 0:3] = rotR
            M = T.dot(R)

            if s == 1:
                M = T.dot(R)
            else:
                S = np.eye(4)
                S[0:3, 0:3] = np.diag([s, s, s])
                M = T.dot(R).dot(S)

            return M
        source = xyz.T
        target = xyz_corr.T
        """
        Use TEASER++ to perform global registration
        """
        # Prepare TEASER++ Solver
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = NOISE_BOUND
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        print("TEASER++ Parameters are:", solver_params)
        teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        # Solve with TEASER++
        start = time.time()
        teaserpp_solver.solve(source, target)
        end = time.time()
        est_solution = teaserpp_solver.getSolution()
        est_mat = compose_mat4_from_teaserpp_solution(est_solution)
        max_clique = teaserpp_solver.getTranslationInliersMap()
        print("Max clique size:", len(max_clique))
        final_inliers = teaserpp_solver.getTranslationInliers()
        trans = est_mat
        reg_time = end - start
    return trans, reg_time

def feature_matching_one_pair(xyz1, xyz2, feat1, feat2, gt_trans, corr_distance=0.1):
    idx1, idx2 = find_correspondence_one_pair(feat1, feat2)
    xyz1 = apply_trans(xyz1, gt_trans)
    inlier_num = np.sum((np.linalg.norm((xyz1[idx1] - xyz2[idx2]), axis=1) < corr_distance).astype(np.int))
    inlier_ratio = inlier_num/idx1.shape[0]
    return inlier_ratio, inlier_num

def registration_recall_one_pair(est_trans, gt_trans, gt_info, rmse_thresh=0.2):
    DCM = np.dot(np.linalg.inv(gt_trans), est_trans)
    qout = np.zeros(4)
    qout[0] = 0.5 * np.sqrt(1 + DCM[0, 0] + DCM[1, 1] + DCM[2, 2])
    qout[1] = - (DCM[2, 1] - DCM[1, 2]) / ( 4 * qout[0] )
    qout[2] = - (DCM[0, 2] - DCM[2, 0]) / ( 4 * qout[0] )
    qout[3] = - (DCM[1, 0] - DCM[0, 1]) / ( 4 * qout[0] )
    kc = np.concatenate((DCM[:3, 3], -qout[1:]), axis=0)[:, np.newaxis] #[6, 1]
    rmse = np.dot(np.dot(kc.T, gt_info), kc)/gt_info[0, 0]
    if rmse < rmse_thresh*rmse_thresh:
        return 1
    else:
        return 0
def RE_TE_one_pair(gt, est):
    import math
    # np [4, 4], [4, 4]
    gt_R = gt[:3, :3] # [3, 3]
    est_R = est[:3, :3] # [3, 3]
    A = (np.trace(np.dot(gt_R.T, est_R)) - 1)/2
    if A > 1:
        A = 1
    elif A < -1:
        A = -1
    rotError = math.degrees(math.fabs(math.acos(A))) # degree
    translateError = np.linalg.norm(gt[:3, 3] - est[:3, 3]) # norm
    return rotError, translateError

class Threedmatch_Log_Info:
    def __init__(self, idx1, idx2, trans, info):
        self.idx1 = idx1
        self.idx2 = idx2
        self.trans = trans # np array [4,4]
        self.info = info # np array [6,6]
    def __str__(self):
        informationstr = f'idx1:{self.idx1}, idx2:{self.idx2}\ntrans:{np.array_str(self.trans)}\ninfo:{np.array_str(self.info)}'
        return informationstr
    def __repr__(self):
        informationstr = f'idx1:{self.idx1}, idx2:{self.idx2}\ntrans:{np.array_str(self.trans)}\ninfo:{np.array_str(self.info)}'
        return informationstr
# read gt log
def read_gt_log_info(root, scene_list, suffix=''):
    log_info = {}
    for scene in scene_list:
        log_info_one_scene = {}
        scenePath = os.path.join(root, scene + suffix)
        logfile = os.path.join(scenePath, 'gt.log')
        infofile = os.path.join(scenePath, 'gt.info')
        with open(logfile, 'r') as f:
            gtlog = f.readlines()
        with open(infofile, 'r') as f:
            gtinfo = f.readlines()
        log_info_one_scene['num_point_cloud'] = int(gtlog[0].strip().split('\t')[2])
        log_info_one_scene['data'] = {}
        i, j = 0, 0
        while i < len(gtlog):
            logline = gtlog[i].strip().split('\t')
            assert len(logline)==3
            head = [int(s) for s in logline]
            trans = np.array([[float(s) for s in gtlog[i+1].strip().split('\t')],
                            [float(s) for s in gtlog[i+2].strip().split('\t')],
                            [float(s) for s in gtlog[i+3].strip().split('\t')],
                            [float(s) for s in gtlog[i+4].strip().split('\t')]])
            log_info_one_scene['data'][f'{head[0]}_{head[1]}'] = Threedmatch_Log_Info(head[0], head[1], trans, None)
            i += 5
        while j < len(gtinfo):
            infoline = gtinfo[j].strip().split('\t')
            assert len(infoline)==3
            head = [int(s) for s in infoline]
            info = np.array([[float(s) for s in gtinfo[j+1].strip().split('\t')],
                            [float(s) for s in gtinfo[j+2].strip().split('\t')],
                            [float(s) for s in gtinfo[j+3].strip().split('\t')],
                            [float(s) for s in gtinfo[j+4].strip().split('\t')],
                            [float(s) for s in gtinfo[j+5].strip().split('\t')],
                            [float(s) for s in gtinfo[j+6].strip().split('\t')]])
            log_info_one_scene['data'][f'{head[0]}_{head[1]}'].info = info
            j += 7
        log_info[scene] = log_info_one_scene
    return log_info

def evaluate_one_scene(args, log_info, scene):
    saveDir = os.path.join(args.saveRoot+f'{args.estimation_pose_func}_{args.voxel_size}_{args.num_key_points}', scene)
    os.makedirs(saveDir, exist_ok=True)
    scene_data = log_info[scene]
    num_point_cloud = scene_data['num_point_cloud']
    log_infoes = scene_data['data']
    gt_num = len(log_infoes)
    inlier_num_sum = 0
    inlier_ratio_sum = 0
    time_sum = 0
    rotError_sum = 0
    translateError_sum = 0
    registration_true_positive_sum = 0
    feature_matching_num_sum = 0
    rot_translate_positive_sum = 0
    registration_gt_num = 0
    #registration_pred_positive_sum = 0
    for i in range(num_point_cloud):
        for j in range(i+1, num_point_cloud):
            key = f'{i}_{j}'
            savePath = os.path.join(saveDir, f'cloud_{i}_{j}.result.txt')
            if os.path.exists(savePath):
                #print(f'{scene}_{i}_{j} file exists')
                with open(savePath, 'r') as f:
                    content = f.readlines()
                data = content[0].strip().split('\t')
                _, inlier_ratio, reg_time, rotError, translateError = [float(i) for i in data[:5]]
                inlier_num, registration_gt, registration_true_positive, feature_matching_num, rot_translate_positive = [int(i) for i in data[5:]]#, registration_pred_positive = [int(i) for i in data[5:]]
                # est_trans = np.array([content[1].strip().split('\t')] + [content[2].strip().split('\t')]+\
                #                 [content[3].strip().split('\t')] + [content[4].strip().split('\t')])
                inlier_num_sum += inlier_num
                inlier_ratio_sum += inlier_ratio
                time_sum += reg_time
                rotError_sum += rotError
                translateError_sum += translateError
                registration_gt_num += registration_gt
                registration_true_positive_sum += registration_true_positive
                feature_matching_num_sum += feature_matching_num
                rot_translate_positive_sum += rot_translate_positive
                #registration_pred_positive_sum += registration_pred_positive
                continue
            key_xyz1, key_feat1, all_xyz1, all_feat1 = feature_reader(scene, i, args.num_key_points)
            key_xyz2, key_feat2, all_xyz2, all_feat2 = feature_reader(scene, j, args.num_key_points)
            
            if (all_feat1 is not None) and (all_feat2 is not None):
                xyz, xyz_corr = all_xyz1, all_xyz2
                feat, feat_corr = all_feat1, all_feat2
            else:
                xyz, xyz_corr = key_xyz1, key_xyz2
                feat, feat_corr = key_feat1, key_feat2
            # target -> source
            est_trans, reg_time = est_trans_one_pair(xyz_corr, xyz, feat_corr, feat, args.voxel_size, func=args.estimation_pose_func)
            #est_trans, reg_time = est_trans_one_pair(xyz, xyz_corr, feat, feat_corr, args.voxel_size, func='ransac')
            #est_trans = np.linalg.inv(est_trans)
            #overlap_ratio = compute_overlap_ratio(all_xyz2, all_xyz1, est_trans, args.voxel_size)
            #registration_pred_positive = int(overlap_ratio > args.overlap_ratio_thresh)
            if key in log_infoes.keys():
                gt_flag = 1
                log_info = log_infoes[key]
                gt_trans = log_info.trans
                gt_info = log_info.info
                
                # target -> source
                inlier_ratio, inlier_num = feature_matching_one_pair(key_xyz2, key_xyz1, key_feat2, key_feat1, gt_trans, corr_distance=0.1)
                feature_matching_num = int(inlier_ratio > args.inlier_thresh)
                if inlier_ratio < args.inlier_thresh:
                    print(f'{scene[:5]}:{i}_{j} inlier_ratio = {inlier_ratio}, inlier_num = {inlier_num}')
                rotError, translateError = RE_TE_one_pair(gt_trans, est_trans)
                if (rotError < args.rot_thresh) and (translateError < args.translate_thresh):
                    rot_translate_positive = 1
                else:
                    rot_translate_positive = 0
                if (j - i) >1:
                    registration_gt = 1
                    registration_true_positive = registration_recall_one_pair(est_trans, gt_trans, gt_info)
                else:
                    registration_gt = 0
                    registration_true_positive = 0
                os.makedirs(os.path.join(os.getcwd(), 'log_result', scene+'-evaluation'), exist_ok=True)
                with open(os.path.join(os.getcwd(), 'log_result', scene+'-evaluation', 'D3Feat_evaluate_3rd.log'), 'a+') as f:
                    f.write(f'{i}\t{j}\t{37}\n')#{registration_pred_positive}\n')
                    f.write(f"{est_trans[0,0]}\t{est_trans[0,1]}\t{est_trans[0,2]}\t{est_trans[0,3]}\n")
                    f.write(f"{est_trans[1,0]}\t{est_trans[1,1]}\t{est_trans[1,2]}\t{est_trans[1,3]}\n")
                    f.write(f"{est_trans[2,0]}\t{est_trans[2,1]}\t{est_trans[2,2]}\t{est_trans[2,3]}\n")
                    f.write(f"{est_trans[3,0]}\t{est_trans[3,1]}\t{est_trans[3,2]}\t{est_trans[3,3]}\n")
            else:
                gt_flag = 0
                reg_time = 0
                inlier_ratio, inlier_num, rotError, translateError = 0, 0, 0, 0
                registration_gt, registration_true_positive, feature_matching_num, rot_translate_positive = 0, 0, 0, 0
            # 写文件:inlier_ratio, reg_time, rotError, translateError
            # registration_true_positive, feature_matching_num, registration_pred_positive
            # est_trans
            
            with open(savePath, 'w+') as f:
                f.write(f'{gt_flag}\t{inlier_ratio}\t{reg_time:.4f}\t{rotError:.4f}\t{translateError:.4f}\t{inlier_num}\t{registration_gt}\t{registration_true_positive}\t{feature_matching_num}\t{rot_translate_positive}\n')#{registration_pred_positive}\n')
                f.write(f"{est_trans[0,0]}\t{est_trans[0,1]}\t{est_trans[0,2]}\t{est_trans[0,3]}\n")
                f.write(f"{est_trans[1,0]}\t{est_trans[1,1]}\t{est_trans[1,2]}\t{est_trans[1,3]}\n")
                f.write(f"{est_trans[2,0]}\t{est_trans[2,1]}\t{est_trans[2,2]}\t{est_trans[2,3]}\n")
                f.write(f"{est_trans[3,0]}\t{est_trans[3,1]}\t{est_trans[3,2]}\t{est_trans[3,3]}\n")
            inlier_num_sum += inlier_num
            inlier_ratio_sum += inlier_ratio
            time_sum += reg_time
            rotError_sum += rotError
            translateError_sum += translateError
            registration_gt_num += registration_gt
            registration_true_positive_sum += registration_true_positive
            feature_matching_num_sum += feature_matching_num
            rot_translate_positive_sum += rot_translate_positive
            #registration_pred_positive_sum += registration_pred_positive

    inlier_num_avg = inlier_num_sum/gt_num
    inlier_ratio_avg = inlier_ratio_sum/gt_num
    time_avg = time_sum/gt_num
    rotError_avg = rotError_sum/gt_num
    translateError_avg = translateError_sum/gt_num
    feature_matching_recall = feature_matching_num_sum/gt_num
    #registration_precision = registration_true_positive_sum/registration_pred_positive_sum
    registration_recall = registration_true_positive_sum/registration_gt_num
    rot_translate_recall = rot_translate_positive_sum/gt_num

    return inlier_num_avg, inlier_ratio_avg, time_avg,\
            rotError_avg, translateError_avg, feature_matching_recall,\
            registration_recall, rot_translate_recall#, registration_precision

# TODO: read feature file
# rootdir write in the function
# to be implement to read the features 
# for every point cloud in 3dmatch test set

# if num_key_points is not None:
#     # Like FCGF/D3Feat
#     return key_xyz, key_feature, all_xyz, all_feature
# else:
#     # Like 3DSmoothNet/3DMatch/PPfNet/PPFfoldingNet..
#     return key_xyz, key_feature, None, None ()
def feature_reader(scene, idx, num_key_points=None):
    cloud_bin_s = f'cloud_bin_{idx}'
    keyptspath = f"D3Feat_contralo-54-pred/keypoints/{scene}"
    descpath = f"D3Feat_contralo-54-pred/descriptors/{scene}"
    source_keypts = np.load(os.path.join(keyptspath, cloud_bin_s + '.npy'))
    source_desc = np.load(os.path.join(descpath, cloud_bin_s + '.D3Feat.npy'))
    source_desc = np.nan_to_num(source_desc)
    # Select {num_keypts} points based on the scores. The descriptors and keypts are already sorted based on the detection score.
    key_xyz = source_keypts[-num_key_points:, :]
    key_feature = source_desc[-num_key_points:, :]

    return key_xyz, key_feature, source_keypts, None#source_desc

def main():
    import argparse
    import logging
    from multiprocessing import Pool, cpu_count
    from functools import partial
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_key_points', default=None, type=int)
    parser.add_argument('--voxel_size', default=0.0, type=float, help='distance for ransac to check inlier')
    parser.add_argument('--estimation_pose_func', default='ransac', type=str, choices=['ransac', 'fgr', 'teaser'])
    parser.add_argument('--overlap_ratio_thresh', default=0.3, type=float)
    parser.add_argument('--inlier_thresh', default=0.05, type=float)
    parser.add_argument('--dataRoot', default=None, type=str)
    parser.add_argument('--saveRoot', default=None, type=str)
    parser.add_argument('--logfile', default=None, type=str)
    parser.add_argument('--rot_thresh', default=15, type=float)
    parser.add_argument('--translate_thresh', default=0.3, type=float)
    args = parser.parse_args()
    if args.saveRoot is None:
        args.saveRoot = os.path.join(os.getcwd(), 'evaluate-result')
    
    BASIC_CONFIG = '[%(levelname)s] %(asctime)s:%(message)s'
    DATE_CONFIG = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=args.logfile, format=BASIC_CONFIG, datefmt=DATE_CONFIG, level=logging.DEBUG)

    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    logging.info(f'args = {args}')
    #args.dataRoot
    log_info = read_gt_log_info('3dmatch', scene_list, suffix='-evaluation')
    
    pool = Pool(min(len(scene_list), cpu_count()))
    func = partial(evaluate_one_scene, args, log_info)
    results = pool.map(func, scene_list)
    pool.close()
    pool.join()

    inlier_num_avg_list, inlier_ratio_avg_list, time_avg_list = [], [], []
    rotError_avg_list, translateError_avg_list, feature_matching_recall_list = [], [], []
    registration_recall_list, rot_translate_recall_list = [], []
    #registration_precision_list = []
    for result in results:
        inlier_num_avg, inlier_ratio_avg, time_avg,\
        rotError_avg, translateError_avg, feature_matching_recall,\
        registration_recall, rot_translate_recall = result
        #registration_recall, registration_precision = result
        #logging.info(f'------ {scene} ------')
        logging.info(f"Feature Matching Recall {feature_matching_recall*100:.3f}%")
        logging.info(f"Registration Recall {registration_recall*100:.3f}%")
        logging.info(f'Rot Translate Recall {rot_translate_recall*100:.3f}%')
        #logging.info(f"Registration Precision {registration_precision*100}%")
        logging.info(f"Average Num Inliners: {inlier_num_avg:.3f}")
        logging.info(f"Average Inliner Ratio: {inlier_ratio_avg:.3f}")
        logging.info(f"Average Reg Time: {time_avg:.3f}s")
        logging.info(f"Average degree Error: {rotError_avg:.3f} degree")
        logging.info(f"Average translate Error: {translateError_avg:.3f}")
        feature_matching_recall_list.append(feature_matching_recall)
        registration_recall_list.append(registration_recall)
        rot_translate_recall_list.append(rot_translate_recall)
        #registration_precision_list.append(registration_precision)
        inlier_num_avg_list.append(inlier_num_avg)
        inlier_ratio_avg_list.append(inlier_ratio_avg)
        time_avg_list.append(time_avg)
        rotError_avg_list.append(rotError_avg)
        translateError_avg_list.append(translateError_avg)
    logging.info('-------------------- Summary ---------------------------------------')
    logging.info(f"All {len(scene_list)} Scenes Feature Matching Recall {sum(feature_matching_recall_list)*100/len(feature_matching_recall_list):.3f}%")
    logging.info(f"All {len(scene_list)} Scenes Registration Recall {sum(registration_recall_list)*100/len(registration_recall_list):.3f}%")
    logging.info(f"All {len(scene_list)} Scenes Rot Translate Recall {sum(rot_translate_recall_list)*100/len(rot_translate_recall_list):.3f}%")
    #logging.info(f"All {len(scene_list)} Scenes Registration Precision {sum(registration_precision_list)*100/len(registration_precision_list)}%")
    logging.info(f"All {len(scene_list)} Scenes Average Num Inliners: {sum(inlier_num_avg_list)/len(inlier_num_avg_list):.3f}")
    logging.info(f"All {len(scene_list)} Scenes Average Inliner Ratio: {sum(inlier_ratio_avg_list)/len(inlier_ratio_avg_list):.3f}")
    logging.info(f"All {len(scene_list)} Scenes Average Reg Time: {sum(time_avg_list)/len(time_avg_list):.3f}s")
    logging.info(f"All {len(scene_list)} Scenes Average degree Error: {sum(rotError_avg_list)/len(rotError_avg_list):.3f} degree")
    logging.info(f"All {len(scene_list)} Scenes Average translate Error: {sum(translateError_avg_list)/len(translateError_avg_list):.3f}")

    
    logging.info(f"All {len(scene_list)} Scenes STD Registration Recall {np.std([100*i for i in registration_recall_list]):.3f}%")
    logging.info(f"All {len(scene_list)} Scenes STD Rot Translate Recall {np.std([100*i for i in rot_translate_recall_list]):.3f}%")
    
    logging.info(f"All {len(scene_list)} Scenes STD Average Reg Time: {np.std(time_avg_list):.3f}s")
    logging.info(f"All {len(scene_list)} Scenes STD Average degree Error: {np.std(rotError_avg_list):.3f} degree")
    logging.info(f"All {len(scene_list)} Scenes STD Average translate Error: {np.std(translateError_avg_list):.3f}")

if __name__ == '__main__':
    main()
    '''
    要把all_feat加上
    把estimate source->target然后再inv改成直接target->source
    把log trans改成广泛用的那种，不要写死
    voxel_size改个名字
    '''