#!/usr/bin/env python

import numpy as np
import datetime
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

# to convert the camera geometry
pix_spiral = np.load("pixel_spiral_id.npy")
revert_spiral = np.argsort(pix_spiral)



class States:

    def __init__(
        self, 
        images, label,
        geom = CameraGeometry.from_name("LSTCam-002"),
        vmin = -np.inf, vmax = np.inf
    ):
        
        self._images = images
        self.label = label
        self.geom = geom
        self.vmin = vmin
        self.vmax = vmax
        self._images_clip = np.clip(self._images, self.vmin, self.vmax)



class PixelStates(States):

    def __init__(
        self,
        images, label,
        vmin = -np.inf, vmax = np.inf,
        cmin = None, cmax = None,
        geom = CameraGeometry.from_name("LSTCam-002")
    ):
        
        super().__init__(images, label, geom, vmin, vmax)
        # conversion
        self.images = self._images[revert_spiral, :]
        self.images_clip = np.clip(self.images, vmin, vmax)
        # for plot
        self.cmin = cmin
        self.cmax = cmax
        self.display = None


    def adjust_camera_image(self):

        # self.display.add_frame_name=''
        self.display.axes.set_aspect("equal")
        self.display.axes.spines['top'].set_visible(False)
        self.display.axes.spines['right'].set_visible(False)
        self.display.axes.set_xticks([-1, -0.5, 0, 0.5, 1], ha='center', rotation=0)
        self.display.axes.set_yticks([-1, -0.5, 0, 0.5, 1], va='center', rotation=90)
        self.display.axes.set_xlabel(r'$x$ [m]')
        self.display.axes.set_ylabel(r'$y$ [m]')
        self.display.axes.set_xlim(-1.3, 1.3)
        self.display.axes.set_ylim(-1.3, 1.3)
        self.display.axes.grid(color='black', alpha=0.4, linestyle='dashed')
    

    def show_snapshot(
        self, ax,
        idx = -1,
        adjust_plot = True,
        *args, **kwargs,
    ):

        # ToDo: Replace "image_clip" with something like matplotlib.colors.Normalize
        disp = CameraDisplay(
            geometry = self.geom, 
            image = self.images_clip[:, idx],
            ax = ax, show_frame = False,
            title = self.label,
            *args, **kwargs,
        )
        # colorbar
        disp.pixels.set_clim(self.cmin, self.cmax)
        disp.add_colorbar(ax=ax) # label = self.label
        # class variable
        self.display = disp
        # better visualization
        if adjust_plot: self.adjust_camera_image()

        return disp
    

    def show_time_evolution(
        self, ax
    ):
        raise NotImplementedError
    


class ModuleStates(PixelStates):
    
    def __init__(
        self,
        images, label,
        vmin = -np.inf, vmax = np.inf,
        cmin = None, cmax = None,
        geom = CameraGeometry.from_name("LSTCam-002")
    ):
        
        self.module_images = images
        images = np.repeat(images, 7, axis=0)
        super().__init__(
            images, label,
            vmin, vmax, 
            cmin, cmax, geom,
        )




class ClusCoStates:

    def __init__(
        self,
        high_voltage, anode_current, 
        amp_temp, bp_temp,
        scb_temp, scb_hum, 
        l0_rate, l0_threshold, 
        l1_rate, camera_rate,
        l1_threshold_A, l1_threshold_B, l1_threshold_C,
        hot_pixel,
    ):
        
        self.scb_temp = ModuleStates(
            images=scb_temp, 
            cmin = 20, cmax = 40,
            label="SCB Temperature $[\mathrm{C}^{\circ}]$"
        )
        self.scb_hum = ModuleStates(
            images=scb_hum, 
            label="SCB Humidity [%]"
        )
        self.amp_temp = PixelStates(
            images=amp_temp, 
            cmin = 20, cmax = 40,
            label="Amplifier Temperature $[\mathrm{C}^{\circ}]$"
        )
        self.high_voltage = PixelStates(
            images=high_voltage, 
            cmin = 0, cmax = 1500,
            label="High Voltage [V]"
        )
        self.anode_current = PixelStates(
            images=anode_current, 
            cmin = 0.0,
            label="Anode Current [uA]"
        )
        self.bp_temp = ModuleStates(
            images=bp_temp,
            cmin = 20, cmax = 40,
            label="Back-Plane Temperature $[\mathrm{C}^{\circ}]$"
        )
        self.l0_rate = PixelStates(images=l0_rate, label="L0 Rate [Hz]")
        self.l0_threshold = PixelStates(images=l0_threshold, label="L0 Threshold")
        self.l1_rate = ModuleStates(images=l1_rate, label="L1 Rate [Hz]")
        self.camera_rate = ModuleStates(images=camera_rate, label="Camera Rate [Hz]")
        # self.l1_threshold_A = ModuleStates(images=l1_threshold_A, label="foo")
        # self.l1_threshold_B = ModuleStates(images=l1_threshold_B, label="foo")
        # self.l1_threshold_C = ModuleStates(images=l1_threshold_C, label="foo")
        # self.hot_pixel = ModuleStates(images=hot_pixel, label="foo")
    


class ClusCoLog:

    def __init__(
        self, filename,
        version = 1,
        header = 9, 
        unit = 115 # 90
    ):

        header = 9
        unit = 115
        if version == 0: unit = 90

        self.data_header = 6 # ???
        self.data_unit = 53 # ???
        self.info_col = self.make_info_col(version, header, unit)
        self.filename = filename
        self.data = np.loadtxt(self.filename, unpack=True, usecols=self.info_col)
        self.time = self.read_time()
        self.states = self.read_var()


    def _make_info_col(
        self, version = 1, *args, **kwargs,
    ):
        
        # log info
        if version == 0:
            header = 9
            unit = 90
        elif version == 1:
            header = 9
            unit = 115
        else:
            raise KeyError

        # specify the indices
        base_col = [
            np.arange(6, 6+1), # scb temp
            np.arange(8, 8+1), # scb temp
            np.arange(10, 10+7), # amp temp
            np.arange(60, 60+7), # high voltage
            np.arange(68, 68+7), # anode current
            np.arange(76, 76+1), # bp temp
            np.arange(79, 79+7), # ipr
        ]

        if version == 0:
            base_col = base_col + [
                np.arange(87, 87+1), # L1 local
                np.arange(89, 89+1), # Camera Late
            ]
        elif version == 1:
            base_col = base_col + [
                np.arange(88, 88+7), # L0 Local
                np.arange(96, 96+1), # L1 local
                np.arange(98, 98+1), # Camera Late
                np.arange(101, 101+6), # L1 thresholds
                np.arange(108, 108+7), # hot pixel
            ]
        else:
            raise KeyError
    
        # single array
        base_col = np.concatenate(base_col)

        # header shift
        base_col = base_col + header
        # iterate: module
        nmod = 265
        base_col = np.repeat(base_col.reshape(1,-1), nmod, axis=0)
        ind_mod = (np.arange(0, nmod) * unit).reshape(-1,1)
        info_col = base_col + ind_mod
        info_col = info_col.flatten()

        # time info
        info_col = np.append((2,3,4,5,6,7), info_col)

        return info_col


    def make_info_col(
        self, version = 1, *args, **kwargs,
    ):
        
        header = 9
        unit = 115

        info_col = np.empty(0, int)
        info_col = np.append(info_col, (2,3,4,5,6,7)) # time

        for imod in range(265):
            info_col = np.append(info_col, header+unit*imod+6) # SCB TEMP
            info_col = np.append(info_col, header+unit*imod+8) # SCB HUM
    
            info_col = np.append(info_col, header+unit*imod+10) # AMP TEMP0
            info_col = np.append(info_col, header+unit*imod+11) # AMP TEMP1
            info_col = np.append(info_col, header+unit*imod+12) # AMP TEMP2
            info_col = np.append(info_col, header+unit*imod+13) # AMP TEMP3
            info_col = np.append(info_col, header+unit*imod+14) # AMP TEMP4
            info_col = np.append(info_col, header+unit*imod+15) # AMP TEMP5
            info_col = np.append(info_col, header+unit*imod+16) # AMP TEMP6
            
            info_col = np.append(info_col, header+unit*imod+60) # HV0
            info_col = np.append(info_col, header+unit*imod+61) # HV1
            info_col = np.append(info_col, header+unit*imod+62) # HV2
            info_col = np.append(info_col, header+unit*imod+63) # HV3
            info_col = np.append(info_col, header+unit*imod+64) # HV4
            info_col = np.append(info_col, header+unit*imod+65) # HV5
            info_col = np.append(info_col, header+unit*imod+66) # HV6
            
            
            info_col = np.append(info_col, header+unit*imod+68) # Anode current0
            info_col = np.append(info_col, header+unit*imod+69) # Anode current1
            info_col = np.append(info_col, header+unit*imod+70) # Anode current2
            info_col = np.append(info_col, header+unit*imod+71) # Anode current3
            info_col = np.append(info_col, header+unit*imod+72) # Anode current4
            info_col = np.append(info_col, header+unit*imod+73) # Anode current5
            info_col = np.append(info_col, header+unit*imod+74) # Anode current6
            
            info_col = np.append(info_col, header+unit*imod+76) # BP TEMP
            
            info_col = np.append(info_col, header+unit*imod+79) # IPR0
            info_col = np.append(info_col, header+unit*imod+80) # IPR1
            info_col = np.append(info_col, header+unit*imod+81) # IPR2
            info_col = np.append(info_col, header+unit*imod+82) # IPR3
            info_col = np.append(info_col, header+unit*imod+83) # IPR4
            info_col = np.append(info_col, header+unit*imod+84) # IPR5
            info_col = np.append(info_col, header+unit*imod+85) # IPR6
        
            if(version==0):

                info_col = np.append(info_col, header+unit*imod+87) # L1 local trigger rate
                info_col = np.append(info_col, header+unit*imod+89) # camera trigger rate
            
            elif(version==1):
            
                info_col = np.append(info_col, header+unit*imod+88) # L0 threshold0
                info_col = np.append(info_col, header+unit*imod+89) # L0 threshold1
                info_col = np.append(info_col, header+unit*imod+90) # L0 threshold2
                info_col = np.append(info_col, header+unit*imod+91) # L0 threshold3
                info_col = np.append(info_col, header+unit*imod+92) # L0 threshold4
                info_col = np.append(info_col, header+unit*imod+93) # L0 threshold5
                info_col = np.append(info_col, header+unit*imod+94) # L0 threshold6
                
                info_col = np.append(info_col, header+unit*imod+96) # L1 local trigger rate
                info_col = np.append(info_col, header+unit*imod+98) # camera trigger rate
                
                info_col = np.append(info_col, header+unit*imod+101) # L1 threshold0
                info_col = np.append(info_col, header+unit*imod+102) # L1 threshold1
                info_col = np.append(info_col, header+unit*imod+103) # L1 threshold2
                info_col = np.append(info_col, header+unit*imod+104) # L1 threshold3
                info_col = np.append(info_col, header+unit*imod+105) # L1 threshold4
                info_col = np.append(info_col, header+unit*imod+106) # L1 threshold5
                
                info_col = np.append(info_col, header+unit*imod+108) # hotpix0
                info_col = np.append(info_col, header+unit*imod+109) # hotpix1
                info_col = np.append(info_col, header+unit*imod+110) # hotpix2
                info_col = np.append(info_col, header+unit*imod+111) # hotpix3
                info_col = np.append(info_col, header+unit*imod+112) # hotpix4
                info_col = np.append(info_col, header+unit*imod+113) # hotpix5
                info_col = np.append(info_col, header+unit*imod+114) # hotpix6

        return info_col


    def read_time(self):

        time_=[[] for i in range(6)]
        time=[]

        for i in range(6):
            time_[i] = np.array(self.data[i], dtype='int')
            
        for i in range(np.size(time_[0])):
            datetime_ = datetime.datetime(time_[0][i],time_[1][i],time_[2][i],time_[3][i],time_[4][i],time_[5][i], tzinfo=datetime.timezone.utc)
            time = np.append(time, datetime_)
        
        return time
    

    def read_var(self):

        temp_scb_ = [[] for i in range(265)]
        humi_scb_ = [[] for i in range(265)]
        temp_amp_ = [[] for i in range(1855)]
        hv_ = [[] for i in range(1855)]
        an_cur_ = [[] for i in range(1855)]
        temp_bp_ = [[] for i in range(265)]
        l0_rate_ = [[] for i in range(1855)]
        l0_threshold_ = [[] for i in range(1855)]
        l1_rate_ = [[] for i in range(265)]
        camera_rate_ = [[] for i in range(265)]
        l1_threshold_A_ = [[] for i in range(265)]
        l1_threshold_B_ = [[] for i in range(265)]
        l1_threshold_C_ = [[] for i in range(265)]
        hot_pixel_ = [[] for i in range(1855)]
        
        for imod in range(265):    
            ## -- Temperature (SCB) -- ##
            temp_scb_[imod] = self.data[self.data_header + 0 + self.data_unit * imod] #8-14  
            
            ## -- Humidity (SCB) -- ##
            humi_scb_[imod] = self.data[self.data_header + 1 + self.data_unit * imod] #8-14

            ## -- Temperature(PACTA board) -- ##
            for ipix in range(7):
                temp_amp_[imod*7+ipix] = self.data[self.data_header + 2 + self.data_unit*imod + ipix] #8-14

            ## --  HV --- ##
            for ipix in range(7):
                hv_[imod*7+ipix] = self.data[self.data_header + 9 + self.data_unit * imod + ipix] #15-21 
                
            ## --  anode current --- ##
            for ipix in range(7):
                an_cur_[imod*7+ipix] = self.data[self.data_header + 16 + self.data_unit * imod + ipix]

            ## -- Temperature (SCB) -- ##
            temp_bp_[imod] = self.data[self.data_header + 23 + self.data_unit * imod] #8-14  
    
            ## --  L0 rate --- ##
            for ipix in range(7):
                l0_rate_[imod*7+ipix] = self.data[self.data_header + 24 + self.data_unit * imod + ipix] #30-36    

            ## --  L0 threshold -- ##
            for ipix in range(7):
                l0_threshold_[imod*7+ipix] = self.data[self.data_header + 31 + self.data_unit * imod + ipix] #37-43    

            ## --  L1 rate --- ##
            l1_rate_[imod] = self.data[self.data_header + 38 + self.data_unit * imod] #44  
    
            ## --  camera rate --- ##
            camera_rate_[imod] = self.data[self.data_header + 39 + self.data_unit * imod] #45 
    
            ## --  L1 threshold -- ##             
            l1_threshold_A_[imod] = self.data[self.data_header + 40 + self.data_unit*imod]/4 #46-51   
            l1_threshold_B_[imod] = self.data[self.data_header + 41 + self.data_unit*imod]/4 #46-51
            l1_threshold_C_[imod] = self.data[self.data_header + 42 + self.data_unit*imod]/4 #46-51
    
            ## --  hot_pixel -- ##
            for ipix in range(7):
                hot_pixel_[imod*7+ipix] = self.data[self.data_header + 46 + self.data_unit * imod + ipix] #52-58    
        
    
        scb_temp = np.vstack(temp_scb_)
        scb_hum = np.vstack(humi_scb_)
        amp_temp = np.vstack(temp_amp_)
        high_voltage = np.vstack(hv_)
        anode_current = np.vstack(an_cur_)
        bp_temp = np.vstack(temp_bp_)
        l0_rate = np.vstack(l0_rate_)
        l0_threshold = np.vstack(l0_threshold_)
        l1_rate = np.vstack(l1_rate_)
        camera_rate = np.vstack(camera_rate_)
        l1_threshold_A = np.vstack(l1_threshold_A_)
        l1_threshold_B = np.vstack(l1_threshold_B_)
        l1_threshold_C = np.vstack(l1_threshold_C_)
        hot_pixel = np.vstack(hot_pixel_)


        ####################################
        ### overwritten below by Shotaro ###
        ####################################

        clusco_states = ClusCoStates(
            scb_temp = scb_temp,
            scb_hum = scb_hum,
            amp_temp = amp_temp,
            high_voltage = high_voltage,
            anode_current = anode_current,
            bp_temp = bp_temp,
            l0_rate = l0_rate,
            l0_threshold = l0_threshold,
            l1_rate = l1_rate,
            camera_rate = camera_rate,
            l1_threshold_A = l1_threshold_A,
            l1_threshold_B = l1_threshold_B,
            l1_threshold_C = l1_threshold_C,
            hot_pixel = hot_pixel,
        )
        
        return clusco_states