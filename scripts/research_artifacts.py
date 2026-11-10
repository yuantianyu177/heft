"""Stable experiment identifiers; scalar decimals must never be file suffixes."""
from pathlib import Path

def scalar_tag(value):
    return f'{value:g}'.replace('.','p').replace('-','m')

def response_stem(root,kind,name,codes,epsilon,sigma=.5,directions=2):
    folders={'rademacher':'response_sketch_wan','gaussian':'gaussian_response_sketch_wan','pixel':'pixel_response_sketch_wan',
             'fourier':'fourier_response_sketch_wan'}
    tag=f'{name}_sigma{int(round(sigma*1000)):04d}_k{codes}'
    if kind!='gaussian':tag+=f'_d{directions}'
    tag+=f'_eps{scalar_tag(epsilon)}'
    return Path(root)/'results'/folders[kind]/tag
