import numpy as np
from os import listdir

BASE_DIR_THEBE_TRAIN_SEISMIC = '/data/cvfault/thebeData/processedThebe/train/seismic'

BEATRICE_RAW_DATA = './dataset_raw/beatrice.npy'
THEBE_TRAIN_DATA = './dataset_raw/thebe_train.npy'
FAULT_DATA = '/data/cvfault/faultSegData/validation/seis/{}.dat'
# FAULT_SELECTED_DATA = [4,13,14,20,23,24,100,121,130,116,118,111,112,113,105,106,107,109,86,98,59,57,39,62,63,91,183,199,171,191,159,166]
# FAULT_SELECTED_DATA = [1,12,41,42,74,95,97,64,54,49,103,105,113,115,120,121,124,134,145,150,155,161,163,164,170,182,191,195,106,117,128,136,146,148,188,198,199]
# GSB_DATA = '/data/cvfault/gsbData/gsb_{}.npy'
# GSB_SUB_TRACE = ['crl_2600','crl_2700','crl_2800','crl_2900','crl_3000','inl_1791','inl_2011']

def normalize(img, min_val = 0, max_val = 1):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_val - min_val) + min_val

def rgb_to_gray(img):
    if len(img.shape) == 3:
        img = .03*img[:,:,0]+0.59*img[:,:,1]+0.11*img[:,:,2]
    return img

def dataloader(trace_name, seismic):
    result = []
    if trace_name == 'thebe' and seismic:
        files = [file for file in listdir(BASE_DIR_THEBE_TRAIN_SEISMIC) if not file.startswith('.')]
        files = files[:10000]
        for file in files:
            result.append(np.load(BASE_DIR_THEBE_TRAIN_SEISMIC + '/{}'.format(file)))
    return files, np.asarray(result)

def full_trace(trace_name):
    rtn = None
    if trace_name == 'beatrice':
        print("loading data from beatrice")
        rtn = np.rot90(np.load(BEATRICE_RAW_DATA), axes = (1,2))
        print("load success, data shape is {}".format(rtn.shape))
        return rtn
    if trace_name == 'thebe_train':
        print("loading data from thebe train")
        rtn = np.rot90(np.load(THEBE_TRAIN_DATA), 3, axes = (1,2))
        print("load success, data shape is {}".format(rtn.shape))
        return rtn
    if trace_name == 'thebe_val':
        print("loading data from thebe val")
        rtn = np.rot90(np.load(THEBE_VAL_DATA), 3, axes = (1,2))
        print("load success, data shape is {}".format(rtn.shape))
        return rtn
    if trace_name == 'thebe_test':
        print("loading data from thebe test")
        rtn = np.rot90(np.load(THEBE_TEST_DATA), 3, axes = (1,2))
        print("load success, data shape is {}".format(rtn.shape))
        return rtn
    if trace_name == 'faultseg':
        print("loading data from thebe test (only selected)")
        rtn = None
        for num in range(20):
            dat = np.fromfile(FAULT_DATA.format(num),dtype=np.single).reshape(128,128,128)
            dat = np.rot90(dat, axes = (1,2))
            if rtn is None:
                rtn = dat
            else:
                rtn = np.concatenate((rtn,dat), axis=0)
#         rtn = np.array(rtn)      
        print("load success, data shape is {}".format(rtn.shape))  
        return rtn
    if trace_name == 'gsb':
        print("loading data from gsb")
        rtn=[]
        for file_name in GSB_SUB_TRACE:
            gsb_np = np.load(GSB_DATA.format(file_name))[0]
            print(gsb_np.shape)
            max_x,max_y = gsb_np.shape
            if (max_x > max_y):
                gsb_np = np.rot90(gsb_np)
            if gsb_np.shape[1] > 484:
                gsb_np = gsb_np[:,:484]
            rtn.append(gsb_np)
        rtn = np.array(rtn)
        print("load success, data shape is {}".format(rtn.shape))
        return rtn
    print("wrong param, load failed!")
    return rtn

def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out, noise # 这里也会返回噪声，注意返回值