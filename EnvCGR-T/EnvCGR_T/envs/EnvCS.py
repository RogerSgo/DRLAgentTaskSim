import os
import sys
import cv2 as cv
import gymnasium
import numpy as np
import sim
import time
import random
import math
import random
from matplotlib import pyplot as plt 
from scipy.interpolate import PchipInterpolator, CubicSpline
import torch as th   
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium import spaces, error, utils
from gymnasium.utils import seeding
from typing import Optional
#----------------------------------------------------------------------------------------------------------------------------
class ConTask(gymnasium.Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps":4}
    def __init__(self):      
        # CONEXION AL ENTORNO
        sim.simxFinish(-1) # Cerrar todas las conexiones abiertas
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 3000, 5)
        if self.clientID!=-1:
            print('Conectado al servidor API remoto')
            sim.simxSynchronous(self.clientID,True)
        else:
            print('Conexion no exitosa')
            sys.exit('No se puede conectar')
        # VARIABLES
        self.x, self.y, self.w, self.h = 105, 0, 50, 255   # ROI-Imagen de entrada
        self.image_width, self.image_height = 50, 150   # Ancho y altura espacio de observacion
        self.nw, self.nh = 50, 150
        self.reward = 0.0   # Recompensa
        self.n_actions = 3  # Salida de espacio de accion
        self.step_counter = 0   # Contador de episodio 
        # PUNTOS INICIALES/FINALES DE TRAYECTORIAS
        self.default_pos = np.array([0.475, 0.025, 0.0])   # T1 entrenamiento
        #self.default_pos = [0.475, 0.225, 0.0]   # recta
        #self.default_pos = [0.450, 0.125, 0.0]   # curva con recta
        self.pos_final = np.array([0.775, 0.025, 0.0])   # T1 entrenamiento
        #self.pos_final = [0.775, 0.225, 0.0]   # recta
        #self.pos_final = [0.800, 0.125, 0.0]   # curva con recta
        # MANEJADORES DE ESCENA
        _, self.target = sim.simxGetObjectHandle(self.clientID, 'Target', sim.simx_opmode_blocking)   # EfectorFinal
        _, self.tcp = sim.simxGetObjectHandle(self.clientID, 'Tip', sim.simx_opmode_blocking)   # Punta TCP
        _, self.cam = sim.simxGetObjectHandle(self.clientID, 'CamCen', sim.simx_opmode_blocking)   # Camara central
        _, self.objetivo = sim.simxGetObjectHandle(self.clientID, 'Trayectoria1', sim.simx_opmode_blocking)   # Trayectoria
        # ESPACIO DE OBSERVACION        
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, 1), dtype=np.uint8) # imgs
        # ESPACIO DE ACCION
        self.action_space = spaces.Box(-1., 1, shape = (self.n_actions,), dtype = 'float32') 
#----------------------------------------------------------------------------------------------------------------------------
#                                                  METODOS GYMNASIUM
#----------------------------------------------------------------------------------------------------------------------------
    def step(self, action):
        print('--------------------------------------------------------------------')
        self.step_counter += 1
        print('STEP: ', self.step_counter)
        self.truncated, self.terminated = False, False 
        print('Acciones: ', action)
        op = self.get_orientacion(self.target)   # Orientacion del efector final
        # EJECUTAR ACCION  
        self.control_ef(action)  # Establecer una accion del agente
        # OBTENER INFORMACION ADICIONAL
        self.info = self._get_info()
        self.obs = self._get_obs()   # Imagenes
        c, t, x, y = self.proc_img(self.obs) 
        # RECOMPENSAS y PENALIZACIONES
        rc = self.compute_reward(action, op)
        self.reward = rc
        print(f'--- RECOMPENSA TOTAL: {self.reward:.3f} --- ')    

        if c[0]<20 or c[0]>30 or c[1]>130 or c[1]<20:   # Reinicio cuando se llega a terminacion de estado
            self.terminated = True
            print(' --- REINICIO --- ')            
        #self.mov_tray(self.step_counter)   # desplazar la trayectoria      
        return self.obs, self.reward, self.terminated, self.truncated, self.info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        
        pos_ef = self.default_pos.copy()
        set_posIni = self.set_posicion(pos_ef)
        self.set_orientacion([0.0, 0.0, 0.0])
        observation = self._get_obs()   # Establecer la observacion
        info = self._get_info()
        return observation, info

    def render(self, mode="rgb_array"):
        imagen = self._get_obs()
        center, theta, endx, endy = self.proc_img(imagen)

        if mode == "rgb_array":
            # Mostrar ROI 
            plt.imshow(imagen, cmap='gray')
            plt.scatter(center[0], center[1], marker="o")
            plt.plot([center[0], endx], [center[1], endy])
            plt.show()
        return imagen
    
    def close(self): # no cambiar
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)   # Cerrar conexion a CoppeliaSim
#--------------------------------------------------------------------------------------------------------------------------
#                                                FUNCIONES CONTROL/CONFIGURACION
#--------------------------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        g, p = self.get_imagen()
        return g
    
    def get_imagen(self):
        '''
            Adquisicion de imagen de camara ubicada en el extremo del efector final.
            Retorno
                Variable de imagen: RGB (camara central), Imagen procesada.
        '''        
        # Camara central
        _, re, imgc = sim.simxGetVisionSensorImage(self.clientID, self.cam, False, sim.simx_opmode_blocking)
        imgc = np.array(imgc).astype(np.uint8)
        imgc = np.reshape(imgc, (re[0], re[1], 3))
        imgc = np.flip(imgc, axis=0)
        
        # Imagen Procesada de camara central
        gray_img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY) # Escala en gris
        gray_img = gray_img[self.y:self.y + self.h, self.x:self.x + self.w]   # ROI
        gray_img = cv.resize(gray_img, dsize=(self.nw, self.nh))
        ret, gray_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY_INV) # Establecer umbral
        g_img = np.expand_dims(gray_img, axis=2)
        
        rectified_img = cv.equalizeHist(gray_img) # Rectificar la imagen
        kernel = np.ones((5,5), np.uint8) 
        dilated_img = cv.dilate(rectified_img, kernel, iterations=1) # Dilatar la imagen
        thinned_img = cv.ximgproc.thinning(dilated_img) # Adelgazar la imagen
        thinned_img = np.expand_dims(thinned_img, axis=2) # expandir imagen procesada
        
        #return {'c_image': exp_img}
        return g_img, thinned_img
    
    def proc_img(self, g_img):
        """
            Procesa la img obtenida obteniendo el centro de la linea.
            Retorno:
            Posicion en pixeles de la linea central: centro, x, y.
        """
        fv = False
        img_bin = (g_img > 128).astype(np.uint8)
        contours, _ = cv.findContours(img_bin, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(contours) > 0 :
            # Determine center of gravity and orientation using Moments
            M = cv.moments(contours[0])
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            theta = 0.5*np.arctan2(2*M["mu11"],M["mu20"]-M["mu02"])
            theta_grados = (theta * 180)/math.pi
            endx = 17 * np.cos(theta) + center[0] # largo de linea
            endy = 17 * np.sin(theta) + center[1]
        else:
            center = (50, 150)
            theta_grados = 0
            endx  = 0
            endy = 0
            print("No hay objeto observable")
        
        return center, theta_grados, endx, endy
        
    def get_posicion(self, handle):
        """ 
            Obtener la posicion de un objeto
            Retorno: 
                Array que contiene (X,Y,Z) del objecto.
        """
        _, position = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
        _, quaternion = sim.simxGetObjectQuaternion(self.clientID, handle, -1, sim.simx_opmode_blocking)
        return np.array(position, dtype=np.float32)   #np.r_[position, quaternion]
    
    def get_orientacion(self, handle):
        _, orientacion = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_oneshot)
        og = [180.0 * a / math.pi for a in orientacion]   # de radianes a grados
        return np.array(og, dtype=np.float32)   # angulos Euler
    
    def set_orientacion(self, orAngles):
        or_Angles = [math.pi * a / 180.0 for a in orAngles]   # de grados a radianes
        sim.simxSetObjectOrientation(self.clientID, self.target, -1, or_Angles, sim.simx_opmode_blocking)
    
    def set_posicion(self, pos):
        """
            Establecer la posicion (X, Y, Z) de un objeto en la escena. Manejador Target.
        """
        sim.simxSetObjectPosition(self.clientID, self.target, -1, pos, sim.simx_opmode_blocking)
        
    def mov_tray(self, step):
        v = random.uniform(0, 0.02)
        
        for step in range(0, 50):
            if step % 10 == 0:
                
                sim.simxSetObjectPosition(self.clientID, self.objetivo, -1, [0.625+v, 0.025+v, 0], sim.simx_opmode_blocking)
        
    def med_dist(self):
        """
            Medir la distancia entre el Efector final y el punto final de la trayectoria
            Retorno:
                Distancia[metros]
        """
        pa = self.get_posicion(self.target)
        dist_pi = np.sqrt((self.default_pos[0] - pa[0]) ** 2 + (self.default_pos[1] - pa[1]) ** 2)
        dist_pf = np.sqrt((self.pos_final[0] - pa[0]) ** 2 + (self.pos_final[1] - pa[1]) ** 2)
        return dist_pi, dist_pf
    
    def grafica(self, vx, vy):       
        x = np.array([0.475, 0.490, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.760, 0.775]) # T1
        y = np.array([0.025, 0.025, 0.05, 0.0525, 0.05, 0.025, -0.015, -0.0175, -0.015, 0.025, 0.025])
        
        #x = np.array([0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750, 0.775, 0.800])   # T2
        #y = np.array([0.125, 0.125, 0.125, 0.175, 0.175, 0.175, 0.125, 0.125, 0.125, 0.175, 0.175, 0.175, 0.125, 0.125, 0.125, ])
        
        #x = np.array([0.475, 0.775])   # T3
        #y = np.array([0.225, 0.225])
        
        # Crear una interpolación cúbica
        cs = PchipInterpolator(x, y)

        # Generar puntos para la curva interpolada
        x_interpolated = np.linspace(min(x), max(x), 500)
        y_interpolated = cs(x_interpolated)

        fig, ax1 = plt.subplots(figsize=(8, 2.75), dpi=1000)
  
        color1 = 'tab:blue'
        ax1.plot(x_interpolated, y_interpolated, color=color1, linewidth=15.0)
        ax1.set_ylim(0.03, 0.07)
        ax1.set_xlabel('X axis (m)', fontsize=11)
        ax1.set_ylabel('Y axis (m)', color=color1, fontsize=11)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.plot(vx, vy, color=color2)
        ax2.set_ylim(0.03, 0.07)
        ax2.set_ylabel('End Effector', color=color2, fontsize=11)
        plt.title('Trajectory Tracking', fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        plt.tick_params(axis='both', which='major', labelsize=11)
        
        ax1.grid(color='grey', ls = '-.', lw = 0.25)
        plt.show()
               
        
    def control_ef(self, accion):
        """
            Establecer una accion de movimiento del efector final del robot.
            Movimientos en plano XY y orientacion en Z.
            Comentar set_orientacion() si se va usar para trayectorias rectas. 
        """
        #rango_origen = (-1, 1)
        #rango_destino = (-0.785, 0.785)
        
        pos_act = self.get_posicion(self.target)  # posicion actual del efector
        or_act = self.get_orientacion(self.target)  # orientacion actual del efector
        
        #new_posicion = [pos_act[i] + (accion[i]/200) for i in range(2)]   # Calculo de movimiento XY
        new_posicion = [pos_act[0] + (accion[0]/200), pos_act[1] + (accion[1]/200)]   # valor en 200
        #accionz = np.interp(accion[2], rango_origen, rango_destino)   # cambiar rango de valores 
        new_orientacion = [or_act[0], or_act[1], or_act[2] + (accion[2])]   # Calculo de orientacion Z

        self.set_posicion(new_posicion)  # Establecer posicion X-Y
        self.set_orientacion(new_orientacion)   # Establecer orientacion en Z
        
    def _get_info(self): # verificar
        """
            Medir distancia recorrida por el efector final.
        """
        cal_dis, d, = self.med_dist()
        return {
            'info' : cal_dis
        }  
       
    
    def compute_reward(self, accion, op):
        """
            Funcion de recompensa para seguimiento de trayectoria aleatoria en plano XY.
            Elementos: Accion, posicion, angulo de orientacion.
            Orientacion: Grados
        """
        c, t, x, y = self.proc_img(self.obs)   # Datos de imagen de entrada
        di, df = self.med_dist()   # Medicion distancia desde el inicio o final trayectoria
        oa = self.get_orientacion(self.target)   # Orientacion actual efector final 0.025
        print(f"Angulos: Trayectoria Detectada: {t:.3f}, Efector Final: {oa[2]:.3f}")   
        cy = abs(c[1] - 75) / 75   # Calcular el costo hacia el centro y
        rt = math.exp(-2.25 * cy)   # Recompensa 1 por distancia al centro
        rc = 0.25 * ((180 - abs(t*(-1) - oa[2]))/180)   # Recompensa 2 por Correccion de giro
        dg = 1.0 if t > 0.0 else -1.0   # Dirección del giro (derecha(1), izquierda(-1))
        rg = 2 * accion[2] * dg * (-1)   # Recompensa 3 de giro en la dirección correcta
        ra = 0.25 * np.linalg.norm(accion[0] + accion[1] + accion[2])   # Recompensa 4 por accion de movimiento
        re = 10.0 if df < 0.025 else 0.0   # Recompensa 5 por llegar a final de trayectoria
        self.recompensa_total = rt + rc + rg + ra + re  # funciona
        print(f"RT: {rt:.3f}, RC: {rc:.3f}, RG: {rg:.3f}, RA: {ra:.3f}, RF: {re:.3f}") 
        
        return self.recompensa_total