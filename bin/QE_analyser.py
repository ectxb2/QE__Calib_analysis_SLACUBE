import numpy as np
import matplotlib.pyplot as plt
from Intensity_integral import I_integral, Gaus
from DBScan_tracks_ref import find_centers, draw_hits_dbscaned, draw_boundaries, draw_labels
#from Target_centerDist import *
from sklearn.cluster import DBSCAN
import h5py
from Target_lib import *



#example run script :
# python3 QE_analyser.py selftrigger_2022_08_05_00_04_09_PDT_evd.h5 78

#Pass the function these things
#Argument 1: name of selftrigger file, ex: selftrigger_2022_08_05_00_04_09_PDT_evd.h5
#Argument 2: array of tracks to scan arround, but in specific format ex: 3,5,7,9


detector_bounds = [[-150, 150], [-150, 150], [0, 300]] # mm (x, y, z)
v_drift = 1.6 # mm/us (very rough estimate)
clock_interval = 0.1 # us/tick -- 10 MHz clock rate
drift_distance = detector_bounds[2][1] - detector_bounds[2][0] 
drift_window = drift_distance/(v_drift*clock_interval) # maximum drift time
drift_direction = 1 # +/- 1 depending on the direction of the drift in z
beam_centers = {"left_beam": [-119.9,-3.82],
               "right_beam": [5.2,-3.82]}
#beam center to be used in this run 
beam_center = beam_centers["right_beam"]
#Ligth paramiters
I = 500000000
dev = 800

#f = h5py.File('selftrigger_2022_08_05_05_06_01_PDT_evd.h5')

#f = h5py.File(sys.argv[1])

#eventData = f['charge']['events']['data']
#hitData = f['charge']['hits']['data']

# this is a list of pairs of ints
# the first is an event ID, the second is a hit ID
#eventHitRefs = f['charge']['events']['ref']['charge']['hits']['ref'] 
# removing G2
#"G2":{'pos':[2,7],'type':'VD'},

locations = {"A1": {'pos':[1,1],'type':'Az'}, "A2": {'pos':[2,1],'type':'VD'},
          "A3": {'pos':[3,1],'type':'Az'},"A4": {'pos':[4,1],'type':'VD'},
          "A5": {'pos':[5,1],'type':'Az'},"A6": {'pos':[6,1],'type':'VD'},
          "A7": {'pos':[7,1],'type':'Az'},"A8": {'pos':[8,1],'type':'VD'},
          "A9": {'pos':[9,1],'type':'Az'},"A10": {'pos':[10,1],'type':'VD'},
          "A11": {'pos':[11,1],'type':'Az'},
          
          "B1":{'pos':[1,2],'type':'VD'},"B2":{'pos':[2,2],'type':'Az'},
          "B3":{'pos':[3,2],'type':'VD'},"B4":{'pos':[4,2],'type':'Az'},
          "B5":{'pos':[5,2],'type':'VD'},"B6":{'pos':[6,2],'type':'Az'},
          "B7":{'pos':[7,2],'type':'VD'},"B8":{'pos':[8,2],'type':'Zn_large'},
          "B9":{'pos':[9,2],'type':'VD'},"B10":{'pos':[10,2],'type':'Az'},
          "B11":{'pos':[11,2],'type':'VD'},
          
          "C1":{'pos':[1,3],'type':'Az'},"C2":{'pos':[2,3],'type':'VD'},
          "C3":{'pos':[3,3],'type':'Az'},"C4":{'pos':[4,3],'type':'VD'},
          "C5":{'pos':[5,3],'type':'Az'},"C6":{'pos':[6,3],'type':'VD'},
          "C7":{'pos':[7,3],'type':'Az'},"C8":{'pos':[8,3],'type':'VD'},
          "C9":{'pos':[9,3],'type':'Az'},"C10":{'pos':[10,3],'type':'VD'},
          "C11":{'pos':[11,3],'type':'Az'},
          
          "D1":{'pos':[1,4],'type':'VD'},"D2":{'pos':[2,4],'type':'Az'},
          "D3":{'pos':[3,4],'type':'VD'},"D4":{'pos':[4,4],'type':'Az'},
          "D5":{'pos':[5,4],'type':'VD'},"D6":{'pos':[6,4],'type':'Az'},
          "D7":{'pos':[7,4],'type':'VD'},"D8":{'pos':[8,4],'type':'Az'},
          "D9":{'pos':[9,4],'type':'VD'},"D10":{'pos':[10,4],'type':'Az'},
          "D11":{'pos':[11,4],'type':'VD'},
          
          "E1":{'pos':[1,5],'type':'Az'},"E2":{'pos':[2,5],'type':'VD'},
          "E3":{'pos':[3,5],'type':'Az'},"E4":{'pos':[4,5],'type':'Ag'},
          "E5":{'pos':[5,5],'type':'Az'},"E6":{'pos':[6,5],'type':'Zn_small'},
          "E7":{'pos':[7,5],'type':'Az'},"E8":{'pos':[8,5],'type':'VD'},
          "E9":{'pos':[9,5],'type':'Az'},"E10":{'pos':[10,5],'type':'VD'},
          "E11":{'pos':[11,5],'type':'Az'},
          
          "F1":{'pos':[1,6],'type':'VD'},"F2":{'pos':[2,6],'type':'Az'},
          "F3":{'pos':[3,6],'type':'Zn_small'},"F4":{'pos':[4,6],'type':'Az'},
          "F5":{'pos':[5,6],'type':'Ag'},"F6":{'pos':[6,6],'type':'Az'},
          "F7":{'pos':[7,6],'type':'VD'},"F8":{'pos':[8,6],'type':'Az'},
          "F9":{'pos':[9,6],'type':'VD'},"F10":{'pos':[10,6],'type':'Az'},
          "F11":{'pos':[11,6],'type':'VD'},
          
          "G1":{'pos':[1,7],'type':'Az'},
          "G3":{'pos':[3,7],'type':'Az'},"G4":{'pos':[4,7],'type':'VD'},
          "G5":{'pos':[5,7],'type':'Az'},"G6":{'pos':[6,7],'type':'VD'},
          "G7":{'pos':[7,7],'type':'Az'},"G8":{'pos':[8,7],'type':'VD'},
          "G9":{'pos':[9,7],'type':'Az'},"G10":{'pos':[10,7],'type':'VD'},
          "G11":{'pos':[11,7],'type':'Az'},
          
          "H1":{'pos':[1,8],'type':'VD'},"H2":{'pos':[2,8],'type':'Az'},
          "H3":{'pos':[3,8],'type':'VD'},"H4":{'pos':[4,8],'type':'Az'},
          "H5":{'pos':[5,8],'type':'VD'},"H6":{'pos':[6,8],'type':'Az'},
          "H7":{'pos':[7,8],'type':'VD'},"H8":{'pos':[8,8],'type':'Az'},
          "H9":{'pos':[9,8],'type':'VD'},"H10":{'pos':[10,8],'type':'Az'},
          "H11":{'pos':[11,8],'type':'VD'},
          
          "I1":{'pos':[1,9],'type':'Az'},"I2":{'pos':[2,9],'type':'VD'},
          "I3":{'pos':[3,9],'type':'Az'},"I4":{'pos':[4,9],'type':'VD'},
          "I5":{'pos':[5,9],'type':'Az'},"I6":{'pos':[6,9],'type':'VD'},
          "I7":{'pos':[7,9],'type':'Az'},"I8":{'pos':[8,9],'type':'VD'},
          "I9":{'pos':[9,9],'type':'Az'},"I10":{'pos':[10,9],'type':'VD'},
          "I11":{'pos':[11,9],'type':'Az'},
          
          "J1":{'pos':[1,10],'type':'VD'},"J2":{'pos':[2,10],'type':'Az'},
          "J3":{'pos':[3,10],'type':'VD'},"J4":{'pos':[4,10],'type':'Az'},
          "J5":{'pos':[5,10],'type':'VD'},"J6":{'pos':[6,10],'type':'Az'},
          "J7":{'pos':[7,10],'type':'VD'},"J8":{'pos':[8,10],'type':'Az'},
          "J9":{'pos':[9,10],'type':'VD'},"J10":{'pos':[10,10],'type':'Az'},
          "J11":{'pos':[11,10],'type':'VD'},
          
          "K1":{'pos':[1,11],'type':'Az'},"K2":{'pos':[2,11],'type':'VD'},
          "K3":{'pos':[3,11],'type':'Az'},"K4":{'pos':[4,11],'type':'VD'},
          "K5":{'pos':[5,11],'type':'Az'},"K6":{'pos':[6,11],'type':'VD'},
          "K7":{'pos':[7,11],'type':'Az'},"K8":{'pos':[8,11],'type':'VD'},
          "K9":{'pos':[9,11],'type':'Az'},"K10":{'pos':[10,11],'type':'VD'},
          "K11":{'pos':[11,11],'type':'Az'}}

#remove G2 as it is not visible 

SLACube1_targets = ['A2','A4','A8','A10',
                    'B1','B3','B8','B9','B11',
                    'C2','C4','C6','C8','C9','C10',
                    'D1','D3','D5','D7','D9','D11',
                    'E2','E4','E6','E8','E10',
                    'F1','F3','F5','F7','F9','F11',
                    'G4','G6','G8','G10',
                    'H1','H3','H5','H7','H9','H11',
                    'I2','I4','I6','I8','I10',
                    'J1','J3','J5','J7','J9','J11',
                    'K2','K4','K6','K8','K10',
                    'E1','E3','E5','E7',
                    'F2','F4','F6','F8',
                    'G1','G3','G5','G7','G9',
                    'H2','H4','H6','H8',
                    'I3','I7'
                    ]

event_num = 0
t_num = 0

#convert string from input to usable array for tracks of interest
#t = sys.argv[2].split(',')
#t = [eval(i) for i in t]
dist = 18.5 # pixel center to center diagonally asuming 4.4mm pixel pitch 
#Make function to do this too 

 
#Read centers from file and compare to centers from DBScanner
def center_dif(x_centers,y_centers,targets = SLACube1_targets):
    closests_targets = []
    target_dists = []
    closest_xs = []
    closest_ys = []

    for i in range(0,len(x_centers)):
        shortest_dist = 150
        closest_target = 'error'
        for target in targets:
            target_pos = np.array(locations[target]['pos'])         
            tx = target_pos[0]*(27) - 6*27
            ty = (target_pos[1]*27 - 6*27)*-1
            cluster_target_dist = np.sqrt((x_centers[i] - (tx))**2 + (y_centers[i]-(ty))**2) 
            #distance between cluster and target, 27 converts to mm
            
            if cluster_target_dist < shortest_dist:
                shortest_dist = cluster_target_dist
                closest_target = target
                closest_x =tx
                closest_y = ty
        closests_targets += [closest_target]
        target_dists += [shortest_dist]
        closest_xs += [closest_x]
        closest_ys += [closest_y]

    return(target_dists, closests_targets, closest_xs, closest_ys )        
             
#get target types and locations
def get_targets(targets = SLACube1_targets):
    Az_target_xs = []
    Az_target_ys = []
    VD_target_xs = []
    VD_target_ys = []   
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
    return(Az_target_xs, Az_target_ys, VD_target_xs, VD_target_ys)

def measured_QE(closests_targets,beam_center,I,dev,q_totals):
    QEs = []
    Is = []    
    for i in range(0,len(closests_targets)):
        target = closests_targets[i]
        I = I_integral(target,beam_center,I,dev)
        QEs += [q_totals[i]/I]
        Is += [I]
    return(QEs,Is)
   
   
   
"""for event in eventData:
    if event_num == t[t_num]:

        t0 = event['ts_start']
        tf = event['ts_end']
        eventMask = ( t0 <= hitData['ts']) & (hitData['ts'] < tf ) 
        eventHits = hitData[eventMask]
        px = eventHits['px']
        py = eventHits['py']
        ts = eventHits['ts']
        q = eventHits['q'] 
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xy_tracks = np.array([px,py,ts]).T
        db = DBSCAN(eps = dist, min_samples=4).fit(xy_tracks)
        x_centers , y_centers , q_totals = find_centers(db,px,py,q)
        #Draw cluster center, target and write QE
        #ax = draw_hits_dbscaned(event)
        draw_boundaries(ax)
        draw_labels(ax)  
        #plot cluster centers
        plt.scatter(x_centers,y_centers, s=30, c='k') 
        target_dists, closests_targets, closest_xs, closest_ys = center_dif(x_centers,y_centers,targets = Target_lib.SLACube1_targets)
        #plt.scatter(closest_xs, closest_ys, s = 30, c='b')
        QEs,Is = measured_QE(closests_targets,beam_center,I,dev,q_totals)
        Az_target_xs,Az_target_ys,VD_target_xs,VD_target_ys = get_targets()
        #plot all target centers
        plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
        plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')
        for i in range(0,len(x_centers)):
            pairx = [x_centers[i],closest_xs[i]]
            pairy = [y_centers[i],closest_ys[i]]
            plt.plot(pairx, pairy, color='r', linewidth=1.5) 

        plt.title('Target Cantroid Displacement')
        plt.show()
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xy_tracks = np.array([px,py,ts]).T
        db = DBSCAN(eps = dist, min_samples=4).fit(xy_tracks)
        x_centers , y_centers , q_totals = find_centers(db,px,py,q)
        #Draw cluster center, target and write QE
        #ax = draw_hits_dbscaned(event)
        draw_boundaries(ax)
        draw_labels(ax)  
        plt.title('Target Centroid Change')
        #plot cluster centers
        #plt.scatter(x_centers,y_centers, s=30, c='k') 
        #target_dists, closests_targets, closest_xs, closest_ys = center_dif(x_centers,y_centers,targets = all_targets)
        #plot target centers
        #plt.scatter(closest_xs, closest_ys, s = 30, c='b')
        
        #add light contours
        x = np.linspace(-150,150, num = 30)
        y = np.linspace(-150,150, num = 30)
        [X,Y] = np.meshgrid(x,y)
        z_intensity = Gaus(X,Y,beam_center,I,dev)
        plt.contour(x,y,z_intensity)
        
        plt.scatter(Az_target_xs,Az_target_ys,s = 100, c='b')
        plt.scatter(VD_target_xs,VD_target_ys,s = 30, c='b')  
              
        plt.scatter(x_centers,y_centers, s=30, c='k')
        
        for i in range(0,len(x_centers)):
            plt.text(x_centers[i],y_centers[i],' QE ~ '+str((QEs[i])))
            pairx = [x_centers[i],closest_xs[i]]
            pairy = [y_centers[i],closest_ys[i]]
            plt.plot(pairx, pairy, color='r', linewidth=1)
                
               
        plt.show()
        
        QE_fit, b = np.polyfit(Is,q_totals,1)
        QE_line_label = 'Q = '+ str(QE_fit) + '* I + ' + str(b)
        fitline = QE_fit*np.array(Is)+b
        
        plt.scatter(Is,q_totals)
        plt.plot(Is, fitline, label = QE_line_label)
        
        plt.title('Intensity vs Charge Measured')
        plt.ylabel('Total cluster charge')
        plt.xlabel('Light on target')
        plt.legend()
        plt.show()
        
        t_num += 1
        event_num +=1
    else : 
        event_num +=1
        



"""






