import numpy as np
HU_fat = -115.8
HU_muscle = -2029.6
HU_tumor = 21.1
HU_nerve = -2131.6

def cnr(I_object, I_background, sigma):
    """Function for readability to generate CNR."""
    return np.abs(I_object - I_background)/np.abs(sigma)

sigma_muscle = np.sqrt(np.abs(HU_muscle))
cnr_nerve_muscle = cnr(HU_nerve,HU_muscle,sigma_muscle)
cnr_tumor_muscle = cnr(HU_tumor,HU_muscle,sigma_muscle)
cnr_nerve_tumor = cnr(HU_nerve,HU_tumor,sigma_muscle) # or cnr_nerve_muscle - cnr_tumor_muscle


print("$CNR_{tumor,muscle}$: ", cnr_tumor_muscle,"\n$CNR_{nerve,muscle}$:",cnr_nerve_tumor,"\n$CNR_{nerve,tumor}$:",cnr_nerve_tumor)