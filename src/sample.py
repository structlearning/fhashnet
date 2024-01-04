from scipy import interpolate, integrate
import numpy as np
import argparse

def get_pdf_normalization(T, a1, b1):
    N = integrate.quad(pdf, a1, b1, points=0, limit=10000, args=T)[0]
    return N

def pdf(w, T):

    R_G = 2 * (np.sin(w*T/2))**2 / w**2 + T * np.sin(w*T) / w
    I_G = np.sin(w*T) / w**2 - T * np.cos(w*T) / w
    return  (np.abs(R_G) + np.abs(I_G))

def generate_samples(num_samples, T, a1, b1):
    x1 = np.linspace(a1,b1,100000)
    y1 = pdf(x1,T)                       
    cdf_y = np.cumsum(y1)            
    cdf_y = cdf_y/cdf_y.max()      
    inverse_cdf = interpolate.interp1d(cdf_y,x1)    
    N = get_pdf_normalization(T, a1, b1)    
    # num_samples = 1000
    uniform_samples = np.random.uniform(1e-5,1,num_samples) 
    cdf_samples = inverse_cdf(uniform_samples)
    pdfs = np.array([pdf(w, T) / N for w in cdf_samples])
    samples_with_pdf = list(zip(cdf_samples, pdfs))
    all_data = {
            'samples': cdf_samples,
            'pdf': pdfs,
            'samples_with_pdf': samples_with_pdf,
            'a': a1,
            'b': b1,
            'T': T,
            'num': num_samples
        }
    return all_data

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_q",                   type=int,   default=200)
    ap.add_argument("--a",                       type=int,   default=-100)
    ap.add_argument("--b",                       type=int,   default=100)
    ap.add_argument("--T",                       type=int,   default=125)

    av = ap.parse_args()
    np.random.seed(1)
    generate_samples(av.num, av.T, av.a, av.b)
