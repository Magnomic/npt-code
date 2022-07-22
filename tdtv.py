import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
import sys
import numpy as np
import random
import tensorly as tl
from tensorly.decomposition import parafac
from img_utils import normalize, rgb_to_gray, dataloader, full_trace

def calculate_similarity(t_patches, target_patch):
    sim_array = []
    for patch in t_patches:
        dist = np.linalg.norm(target_patch-patch)
        sim_array.append((1/dist, patch))
#     print(sorted(sim_array, key=lambda x: x[0]))
    return [y[1] for y in sorted(sim_array, key=lambda x: x[0])]

random.seed(1)
# random.seed(666)

# for ite, n_st in zip(range(2,6), [70,90,110,130]):

#     for _ in range(1):
for ite in range(0,4,2):

    idx = random.randint(0, 1000)
    import time

    st_time = time.time()

#     selected_patch = np.load('/home/haiwen/noise_reduction/data_final/noise_data/synthetic_multi_{}/s{}.npy'.format(n_st, 30 + ite + 100 * (ite % 2))) #test_for_show_noise
    selected_patch = np.load('./parameter_test/beatrice_content_256/{}.npy'.format(idx))
    matrix_w = np.zeros(selected_patch.shape)

    sliding_step = 4
    patch_size = 12

    patches = []

    for i in range(patch_size // 2, selected_patch.shape[0] - patch_size // 2+1, sliding_step):
        for j in range(patch_size // 2, selected_patch.shape[0] - patch_size // 2+1, sliding_step):
            patches.append(selected_patch[i-patch_size // 2: i+patch_size // 2, j-patch_size // 2: j+patch_size // 2])
            for k0 in range(i-patch_size // 2, i+patch_size // 2):
                for k1 in range(j-patch_size // 2, j+patch_size // 2):
                    matrix_w[k0, k1] += 1

    square_shape = int(math.sqrt(len(patches)))

    matrix_x = np.zeros(selected_patch.shape)

    # get distance and stacked cube
    arg_S = 10
    for k in range(len(patches)):
        if k % 10 == 0:
            print(k)
        patches_window = []
        for i in range(k - arg_S * square_shape, k + arg_S * square_shape, square_shape):
            if i < 0 or i >= len(patches):
                continue
            for j in range(i-arg_S, i+arg_S):
                # if equal to center point or not in same line 'i'
                if j ==k or j // square_shape != i // square_shape:
                    continue
                patches_window.append(patches[j])
        # get distance
        stacked_patches = np.array(calculate_similarity(patches_window, patches[k])).astype(np.float32)
        # CP 鍒嗚В
        factors = parafac(stacked_patches, rank=12)
        full_tensor = tl.kruskal_to_tensor(factors)
        full_tensor = (full_tensor[0]+full_tensor[1]+full_tensor[2]+full_tensor[3]+full_tensor[4])
        line = k // square_shape
        row = k % square_shape
        ite_i = 0
        for x_i in range(patch_size//2 + line * sliding_step - patch_size // 2, patch_size//2 + line * sliding_step + patch_size // 2):
            ite_j = 0
            for x_j in range(patch_size//2 + row * sliding_step - patch_size // 2, patch_size//2 + row * sliding_step + patch_size // 2):
                matrix_x[x_i, x_j] = matrix_x[x_i, x_j] + full_tensor[ite_i, ite_j]
                ite_j += 1
            ite_i += 1

#     np.save('tdtv_patch_{}_{}.npy'.format(ite, n_st), matrix_x / matrix_w)
    np.save('tdtv_patch_{}.npy'.format(idx), matrix_x / matrix_w)
