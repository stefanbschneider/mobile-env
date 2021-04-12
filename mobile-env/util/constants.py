"""
Utility module holding constants that are used across different places.
Such that this is the single point in the code to adjust these constants.
"""
import os
import pathlib

import svgpath2mpl
import matplotlib.transforms


# logging settings
LOG_ROUND_DIGITS = 3

# use sets for O(1) include checks
CENTRAL_ALGS = {'ppo', 'random', 'fixed', 'brute-force'}
MULTI_ALGS = {'ppo', '3gpp', 'fullcomp', 'dynamic'}
SUPPORTED_ALGS = CENTRAL_ALGS.union(MULTI_ALGS)
SUPPORTED_AGENTS = {'single', 'central', 'multi'}
SUPPORTED_ENVS = {'small', 'medium', 'large', 'custom'}
SUPPORTED_RENDER = {'html', 'gif', 'both', None}
SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'rate-fair', 'proportional-fair'}
SUPPORTED_REWARDS = {'min', 'sum'}

# small epsilon used in denominator to avoid division by zero
EPSILON = 1e-16

# constants to tune "fairness" of proportional-fair sharing
# high alpha --> closer to max cap; high beta --> closer to resource-fair; alpha = beta = 1 is used in 3G
# actually no, alpha=1=beta converges to exactly the same allocation as resource-fair for stationary users!
# https://en.wikipedia.org/wiki/Proportionally_fair#User_prioritization
FAIR_WEIGHT_ALPHA = 1
FAIR_WEIGHT_BETA = 1


# constants regarding result files
def get_result_dirs(result_dir=None):
    """
    Return the path to the result dir, test dir, and video dir.
    If a custom result dir is provided, use that. Otherwise, default to project root/results.
    """
    if result_dir is None:
        # project root (= repo root; where the readme is) for file access
        _this_dir = pathlib.Path(__file__).parent.absolute()
        project_root = _this_dir.parent.parent.absolute()
        result_dir = os.path.join(project_root, 'results')

    train_dir = os.path.join(result_dir, 'train')
    test_dir = os.path.join(result_dir, 'test')
    video_dir = os.path.join(result_dir, 'videos')

    # create dirs
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    return result_dir, train_dir, test_dir, video_dir


# custom symbols for rendering based on SVG paths
# base station icon from noun project: https://thenounproject.com/search/?q=base+station&i=1286474
station_path = "M31.5,19c0-4.1-3.4-7.5-7.5-7.5s-7.5,3.4-7.5,7.5c0,2.9,1.6,5.4,4,6.7l0.9-1.8c-1.7-0.9-2.9-2.8-2.9-4.9 " \
               "c0-3,2.5-5.5,5.5-5.5s5.5,2.5,5.5,5.5c0,2.1-1.2,3.9-2.9,4.9l0.9,1.8C29.9,24.4,31.5,21.9,31.5,19z " \
               "M37,19c0-7.2-5.8-13-13-13s-13,5.8-13,13c0,5.1,2.9,9.4,7.1,11.6l0.9-1.8c-3.6-1.8-6-5.5-6-9.8   " \
               "c0-6.1,4.9-11,11-11s11,4.9,11,11c0,4.3-2.5,8-6,9.8l0.9,1.8C34.1,28.4,37,24.1,37,19z " \
               "M42,19c0-9.9-8.1-18-18-18S6,9.1,6,19c0,7,4.1,13.1,10,16.1l0.9-1.8C11.6,30.7,8,25.2,8,19   " \
               "c0-8.8,7.2-16,16-16s16,7.2,16,16c0,6.2-3.6,11.7-8.8,14.3l0.9,1.8C37.9,32.1,42,26,42,19z " \
               "M24,22c-1.7,0-3-1.3-3-3s1.3-3,3-3s3,1.3,3,3S25.7,22,24,22z M24,18c-0.6,0-1,0.4-1,1s0.4,1,1,1     " \
               "s1-0.4,1-1S24.6,18,24,18z " \
               "M34.2,44.1L24,23.1l-10.2,21c-0.3,0.6-0.3,1.3,0.1,1.9c0.4,0.6,1,0.9,1.7,0.9h16.8c0.7,0,1.3-0.4,1.7-0.9 "\
               "C34.5,45.5,34.5,44.7,34.2,44.1z M25,29.8l2.2,4.4L25,36V29.8z M23,36l-2.2-1.8l2.2-4.4V36z " \
               "M22.5,38.2l-6,5.1l3.5-7.3L22.5,38.2z " \
               "M23,40.4V45h-5.5L23,40.4z M25,40.3l5.5,4.7H25V40.3z M25.6,38.2l2.5-2.1l3.5,7.2L25.6,38.2z"
station_symbol = svgpath2mpl.parse_path(station_path)
station_symbol.vertices -= station_symbol.vertices.mean(axis=0)
# rotate (otherwise up side down): https://stackoverflow.com/a/48231144/2745116
station_symbol = station_symbol.transformed(matplotlib.transforms.Affine2D().rotate_deg(180))
