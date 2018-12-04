
#IMAGE SIZE THAT IS PASSED SHOULD BE EQUAL
def dist1(img1,img2):
    return np.abs(img1,img2)

def dist2(img1,img2):
    return np.sqrt(np.sum((img1-img2)**2))

def psnr(img1,img2):
    mse = np.mean((img1-img2)**2)
    return 20 * np.log10(255.0 / np.sqrt(mse))
