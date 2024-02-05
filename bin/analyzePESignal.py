#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:23:52 2023

@author: yuntse
"""

"""python3 bin/analyzePESignal.py -c "cfg_cold_651b9261/one_cfg" -s exttrig_2023_10_03_02_28_06_PDT.h5 -b exttrig_2023_10_03_03_05_54_PDT.h5 -t 'new try3'

"""

import argparse
import os
from glob import glob

import h5py
import numpy as np
# import pandas as pd
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from exttrig import analyzeExttrig1
from geom import load_layout_np
import Target_lib 
from sklearn.cluster import DBSCAN
from DBScan_tracks_ref import find_centers, draw_hits_dbscaned, draw_boundaries, draw_labels
#import QE_analyser
from QE_analyser import center_dif , measured_QE
from Target_lib import *
from Intensity_integral import Gaus, I_integral

SLACube1_targets = Target_lib.SLACube1_targets
locations = Target_lib.locations
beam_center = [-94,-27]
#Initial uniform illumination assumption

#photons = 3.35E13
#area = 3.141592*(17*4.4/2)**2
#photons_per_pixel = photons/area*(4.4**2)

min_energy_sf = 0.29E-6  #uJ/cm^2
max_energy_sf = 0.51E-6  #uJ/cm^2

min_energy_lf = 0.99E-6  #uJ/cm^2
max_energy_lf = 1.73E-6  #uJ/cm^2

photons_per_mm2 = min_energy_lf/(1.6E-19*4.66)/100  #energy/cm to photons/mm
photons_per_pixel = photons_per_mm2*(4.4**2)

I = 500000
dev = 500
tau = 129.28
drift_time = 200



def make_colors(array):
    out = []
    for i in array:
        cluster = i
        #while cluster > 4:
        #    cluster = cluster - 5        
        if cluster == -1:
            out += ['k'] #noise is black
        elif cluster == 0:
            out += ['b']
        elif cluster == 1:
            out += ['g']
        elif cluster == 2:
            out += ['r']
        elif cluster == 3:
            out += ['c']
        elif cluster == 4:
            out += ['m']
    return out


#%%
def getChannelMask(cfgfiles):
    
    chMask = {}
    for cfgfile in cfgfiles:
        with open(cfgfile, 'r') as f:
            cfg = json.load(f)
            
        reg = cfg['register_values']
        for chid in range(0, 64):
            uid = ((reg['chip_id'] - 11) << 6) + chid
            if uid in chMask.keys():
                print( f'Uid {uid} already exists...')
                continue
            chMask[uid] = reg['channel_mask'][chid]
    
    return chMask

#%%
def make_plot(evt, outfile, title):
    mpl.use('agg')
    sns.set_theme('talk', 'white')

    pix_loc = load_layout_np()

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=True, sharey=True)

    kwargs = dict(marker='o', s=5, cmap='viridis')
    
    mask = evt['active']
    
    x, y = pix_loc[mask].T

    ax = axes[0]
    sc = ax.scatter(x, y, c = evt[mask]['mean'], vmin = 90, vmax = 125, **kwargs)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Mean [ADC]')

    ax = axes[1]
    sc = ax.scatter(x, y, c = evt[mask]['std'], vmin = 0, vmax = 20, **kwargs)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='Std [ADC]')

    for ax in axes:
        ax.set_ylabel('y [mm]')
        axes[-1].set_xlabel('x [mm]')

    fig.suptitle(title, fontsize='medium')
    fig.tight_layout()
    fig.show()
    plt.show()

    fig.savefig(outfile)

#%%
#get target types and locations
def get_targets(targets = SLACube1_targets):
    Az_target_xs = []
    Az_target_ys = []
    VD_target_xs = []
    VD_target_ys = []   
    Zn_large_targets_xs = []
    Zn_large_targets_ys = []
    Zn_small_targets_xs = []
    Zn_small_targets_ys = []
    Ag_targets_xs = []
    Ag_targets_ys = []
    
    for target in targets:
        t_type = locations[target]['type']
        target_pos = np.array(locations[target]['pos']) 
        tx = target_pos[0]*(27) - 6*27
        ty = (target_pos[1]*27 - 6*27)*-1
        if  t_type == 'Az':
            Az_target_xs += [tx]
            Az_target_ys += [ty]
        elif t_type == 'VD':
            VD_target_xs += [tx]
            VD_target_ys += [ty]
        elif t_type == 'Zn_large':
            Zn_large_targets_xs += [tx]
            Zn_large_targets_ys += [ty]
        elif t_type == 'Zn_small':
            Zn_small_targets_xs += [tx]
            Zn_small_targets_ys += [ty]
        elif t_type == 'Ag':
            Ag_targets_xs += [tx]
            Ag_targets_ys += [ty]
    return(Az_target_xs,Az_target_ys,VD_target_xs,VD_target_ys,Zn_large_targets_xs,Zn_large_targets_ys,Zn_small_targets_xs,Zn_small_targets_ys,Ag_targets_xs,Ag_targets_ys)

def tar_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_boundaries(ax)
    draw_labels(ax) 
    Az_target_xs,Az_target_ys,VD_target_xs,VD_target_ys,Zn_large_targets_xs,Zn_large_targets_ys,Zn_small_targets_xs,Zn_small_targets_ys,Ag_targets_xs,Ag_targets_ys = get_targets()
    fig.scatter(Az_target_xs,Az_target_ys,s = 400, c='b')
    plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
    plt.scatter(Zn_large_targets_xs,Zn_large_targets_ys,s = 400, c='r')
    plt.scatter(Zn_small_targets_xs,Zn_small_targets_ys,s =40 , c='r')
    plt.scatter(Ag_targets_xs,Ag_targets_ys,s =40 , c='g')

def pix_select(sig, bkg, outfile, title):
    sns.set_theme('talk', 'white')

    pix_loc = load_layout_np()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    kwargs = dict(marker='o', s=5, cmap='viridis')    
    
    x, y = pix_loc.T
    d = []
    xn = []
    yn = []
    for i in range(0,len(sig)):
        out = sig[i][1] - bkg[i][1]
        if out > 0.75:
            d += [out]
            xn += [x[i]]
            yn += [y[i]]
    d = np.array(d)
    xn = np.array(xn)
    yn = np.array(yn)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_boundaries(ax)
    draw_labels(ax)  
    sc = ax.scatter(xn, yn, c = d, vmin = 0, vmax = d.max(), **kwargs,zorder=10)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label= r'$\Delta$ADC')
    #tar_plot()
    ax.set_ylabel('y [mm]')
    ax.set_xlabel('x [mm]')

    fig.suptitle(title, fontsize='medium')
    fig.tight_layout()
    fig.show()

    fig.savefig(outfile)
    plt.show()
    return(xn,yn,d)
    
def db_scan(x,y,d):
    xy_tracks = np.array([x,y]).T
    dist = 5    
    db = DBSCAN(eps = dist, min_samples=4).fit(xy_tracks)
    labels = db.labels_
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_boundaries(ax)
    draw_labels(ax) 
    #core_samples = db.core_sample_indices_
    #n_clusters = len(set(labels)) - (1 if -1 in labels else 0) 
    color_array = make_colors(labels)
    points = plt.scatter(x, y,c=color_array, s = 10)
    
    plt.show()
    return (db)

def QE_est(d_totals,pix_counts):
    
    QE_cluster = []
    for i in range(0,len (d_totals)):
        q_i = d_totals[i]*1000
        q = q_i*np.exp(drift_time/tau)
        QE = q/(photons_per_pixel*pix_counts[i])
        #print ("QE = electron / photons/pix*pix count")
        #print (" QE = " + str(q) + " / " + str(photons_per_pixel) + " * " + str(pix_counts[i]))
        print (QE)
        QE_cluster += [QE]
    return (QE_cluster)
    
def Sum_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) 
    I_cluster = []
    for cluster_number in range(0,n_clusters):
        I_running = 0
        for i in range(0,len(labels)):
            if cluster_number == labels[i]:
                I_running += Gaus(xn[i], yn[i],beam_center,I,dev)*4.4**2 #need to verify units
        I_cluster += [I_running]
    return  I_cluster
    
def QE_w_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels):   
    xi = np.linspace(-150,150, num = 60)
    yi = np.linspace(-150,150, num = 60)
    [X,Y] = np.meshgrid(xi,yi)
    z_intensity = Gaus(X,Y,beam_center,I,dev)
    plt.contour(xi,yi,z_intensity, cmap = 'viridis')
    QE_cluster = []
    for i in range(0,len (d_totals)):
        q_i = d_totals[i]*1000
        q = q_i*np.exp(drift_time/tau)
        #sum intensity on pixels 
        I_clusters =Sum_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels)
        QE = q/I_clusters[i]
        #print ("QE = electron / photons/pix*pix count")
        #print (" QE = " + str(q) + " / " + str(photons_per_pixel) + " * " + str(pix_counts[i]))
        print (QE)
        QE_cluster += [QE]
    return (QE_cluster)


def make_diffplot( sig, bkg, outfile, title):
    #tar_plot()
    sns.set_theme('talk', 'white')

    pix_loc = load_layout_np()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    kwargs = dict(marker='o', s=5, cmap='viridis')

    #mask = sig['active']
    #x, y = pix_loc[mask].T
    x, y = pix_loc.T
    #d = sig[mask]['mean'] - bkg[mask]['mean']
    d = []
    for i in range(0,len(sig)):
        out = sig[i][1] - bkg[i][1]
        d += [out]
    
    d = np.array(d)
    #d = sig[:][1] - bkg[:][1]
    #tar_plot()
    sc = ax.scatter(x, y, c = d, vmin = 0, vmax = d.max(), **kwargs,zorder=10)
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label= r'$\Delta$ADC')
    #tar_plot()
    ax.set_ylabel('y [mm]')
    ax.set_xlabel('x [mm]')

    fig.suptitle(title, fontsize='medium')
    fig.tight_layout()
    fig.show()

    fig.savefig(outfile)
    plt.show()
    return (d)
#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( description = 'Make a PE map' )
    
    parser.add_argument( '-s', dest = 'sigFile', type = str, 
                        help = 'input signal file' )
    parser.add_argument( '-b', dest = 'bkgFile', type = str,
                        help = 'input background file' )
    parser.add_argument( '-o', dest = 'outDir', type = str, default = 'test',
                        help = 'output directory' )
    parser.add_argument( '-c', dest = 'cfgDir', type = str,
                        help = 'configuration directory' )
    parser.add_argument( '-t', dest = 'title', type = str, default = '2023-10-23 Long Fiber',
                        help = 'plot title' )
    parser.add_argument( '-p', dest = 'progress', type = bool, default = False,
                        help = 'show progress' )
    args = parser.parse_args()
    
    
    # try: os.makedirs( args.outDir )
    # except FileExistsError():
    #     pass
    
    outName = os.path.join(args.outDir, f'{args.title.replace(" ", "_")}.png')

    # Load configuration files
    channelMask = {}
    print( 'Obtaining channel masks...Not used at this moment' )
    cfgfiles = glob(os.path.join(args.cfgDir, 'config-*.json'))
    if len(cfgfiles) == 0:
        raise FileNotFoundError('No config file found', args.cfgDir)
    channelMask = getChannelMask(cfgfiles)
    
    print( f'Processing {args.sigFile}' )
    
    with h5py.File( args.sigFile, 'r' ) as sf:
        signal = analyzeExttrig1( sf['packets'], show_progress = args.progress )
        #print(signal)
    # make_plot(signal, outName, '2023-10-03_SLACube_LongFiber')
    
    print( f'Processing {args.bkgFile}' )
    
    with h5py.File( args.bkgFile, 'r' ) as bf:
        bkg = analyzeExttrig1( bf['packets'], show_progress = args.progress )
    
    make_diffplot(signal, bkg, outName, f'SLACube: {args.title}' )
    xn,yn,d = pix_select(signal, bkg, outName, f'SLACube: {args.title}' )
    #cluster charges
    db = db_scan(xn,yn,d)
    x_centers , y_centers , d_totals , pix_counts = find_centers(db,xn,yn,d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_boundaries(ax)
    draw_labels(ax)  
    labels = db.labels_
    #plot cluster centers
    
    color_array= make_colors(labels)
    points = plt.scatter(xn, yn,c=color_array, s = 10)
    plt.scatter(x_centers,y_centers, s=20, c='k')
    
    plt.show()
    target_dists, closests_targets, closest_xs, closest_ys = center_dif(x_centers,y_centers)
    #QEs,Is = measured_QE(closests_targets,beam_center,I,dev,d_totals)
    Az_target_xs,Az_target_ys,VD_target_xs,VD_target_ys,Zn_large_targets_xs,Zn_large_targets_ys,Zn_small_targets_xs,Zn_small_targets_ys,Ag_targets_xs,Ag_targets_ys = get_targets()
    #plot all target centers
    plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
    plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
    for i in range(0,len(x_centers)):
        pairx = [x_centers[i],closest_xs[i]]
        pairy = [y_centers[i],closest_ys[i]]
        plt.plot(pairx, pairy, color='r', linewidth=1.5) 
    plt.show()    
    QEs = QE_est(d_totals,pix_counts)
    #print (QEs)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_boundaries(ax)
    draw_labels(ax)  
    points = plt.scatter(xn, yn,c=color_array, s = 10)
    for i in range(0,len(x_centers)):
            plt.text(x_centers[i],y_centers[i],' QE ~ '+(str('{:.2e}'.format(QEs[i]))), fontsize = 'xx-small')
    plt.show()
    
    file_name = 'out_min_'+str(args.sigFile)+'.txt'
    file_name =  file_name.replace('SigFiles/', '')
    with open(file_name, 'w') as f:
        f.write('signal: '+ str({args.sigFile}))
        f.write('\n')
        f.write('background: '+ str({args.bkgFile}))
        f.write('\n')
        for val in QEs:
            f.write(str(val))
            f.write('\n')
    print('saved QEs')
    
    
    QE_new = QE_w_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels)
    plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
    plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
    points =plt.scatter(xn, yn,c=color_array, s = 10)
    plt.show()
    
    #QE_new = QE_w_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels)
    plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
    plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
    points =plt.scatter(xn, yn,c=color_array, s = 10)
    plt.show()
    
    #QE_new = QE_w_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels)
    plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
    plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
    #points =plt.scatter(xn, yn,c=color_array, s = 10)
    QE_new = QE_w_intensity(beam_center,I,dev,d_totals,pix_counts, xn, yn, labels)
    plt.show()
    
    #print (QE_new)
    points =plt.scatter(xn, yn,c=color_array, s = 10)
    for i in range(0,len(x_centers)):
            plt.text(x_centers[i],y_centers[i],' QE ~ '+ str((QE_new[i]))[0:5], fontsize = 'xx-small')
    
    plt.show()
    
    

    
    
    
