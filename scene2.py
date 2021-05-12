from manim import *
import imagesc as imagesc
import numdifftools as nd
import numpy as np
import math
from scipy.io import loadmat

import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.cm as cm
import pdb

t=0
omega=3
k=1

Smatrix=[]
Umatrix=[]
Vmatrix=[]
iVal=0

class TravelingSineWave(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": np.pi/2},
                y_axis_config={"tick_frequency": .25},
                **kwargs
            )
    def setup(self):
        GraphScene.setup(self)

    def SineWave(self, x):
        global t, omega, k
        return np.sin(omega*t - k*x)
    
    def construct(self):
        global t, omega, k
        
        self.graph_origin = [-5, -1, 0]
        self.x_axis_width = 6
        self.y_axis_height = 4.5
        
        self.x_min = 0
        self.x_max = 4*np.pi           
        self.y_min = -1   
        self.y_max = 1
        self.x_axis_label = "x"
        self.y_axis_label = "f"
        self.setup_axes(animate=True)

        SinGraph = self.get_graph(self.SineWave).set_color(YELLOW)
        SinRects = self.get_riemann_rectangles(SinGraph, dx=np.pi/16)
        
        tText = Text("t={}".format(t), font='CMU Serif',  color=WHITE).move_to([-6, 2, 0]).scale(0.7)
        eqnText = MathTex(r"f(x,t)=sin(\omega t-kx)").move_to([0, 3, 0])

        self.play(Create(SinGraph), Create(tText), Create(eqnText))
        self.wait(1)

        lastI=60
        deltaX=np.pi/64
        for i in range(1,lastI):
            t=0.05*i

            SinGraph2 = self.get_graph(self.SineWave).set_color(YELLOW)
            SinRects2 = self.get_riemann_rectangles(SinGraph, dx=deltaX)
            
            tText2 = Text("t={}".format(np.round(t*200)/200), font='CMU Serif',  color=WHITE).move_to([-6, 2, 0]).scale(0.7)

            self.play(Transform(SinGraph, SinGraph2), Transform(tText,tText2), run_time=1/60)

        self.play(Create(SinRects2), run_time=1/60)
        for i in range(1,5):
            t=0.05*(i+lastI)

            SinGraph2 = self.get_graph(self.SineWave).set_color(YELLOW)
            SinRects3 = self.get_riemann_rectangles(SinGraph, dx=deltaX)
            
            tText2 = Text("t={}".format(np.round(t*200)/200), font='CMU Serif',  color=WHITE).move_to([-6, 2, 0]).scale(0.7)

            self.play(Transform(SinGraph, SinGraph2), Transform(tText,tText2),Transform(SinRects2, SinRects3), Transform(tText,tText2), run_time=1/60)

        self.play(FadeOut(SinGraph), run_time=1/60)
        for i in range(1,lastI):
            t=0.05*(i+lastI+5)

            SinGraph2 = self.get_graph(self.SineWave).set_color(YELLOW)
            SinRects3 = self.get_riemann_rectangles(SinGraph, dx=deltaX)
            
            tText2 = Text("t={}".format(np.round(t*200)/200), font='CMU Serif',  color=WHITE).move_to([-6, 2, 0]).scale(0.7)

            self.play(Transform(SinRects2, SinRects3), Transform(tText,tText2), run_time=1/60)

        self.wait(1)


        #Creates vectors
        time_length=60
        x_vector=np.arange(0,4*np.pi,deltaX)
        A_Matrix=[]   
        
        pxHei=0.02
        pxWid=0.035 
        
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-1,vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])   

        for i in range(0,time_length):
            
            A_vector=np.sin(omega*t - k*x_vector)
            A_Matrix.append(A_vector)
            
            pixelVec=[]
            for j in range(0,len(x_vector)):
                xVal=3+i*pxWid
                yVal=2-j*pxHei
                clr=cpick.to_rgba(A_vector[j]) 
                pixelVec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            
            Avec=VGroup(*pixelVec)

            self.play(Transform(SinRects2, Avec), lag_ratio=0.4, run_time=1)

            t=t+0.05
            SinGraph2 = self.get_graph(self.SineWave).set_color(YELLOW)
            SinRects2 = self.get_riemann_rectangles(SinGraph, dx=deltaX)
            
            tText2 = Text("t={}".format(np.round(t*200)/200), font='CMU Serif',  color=WHITE).move_to([-6, 2, 0]).scale(0.7)

            self.play(Create(SinRects2), Transform(tText,tText2), run_time=0.5)
            self.wait(0.5)
            
        self.wait(1)
        eqnText2 = MathTex(r"sin(\omega t-kx) = \underbrace{sin(\omega t)cos(-kx)}_{Mode 1} + \underbrace{cos(\omega t)sin(-kx)}_{Mode 2}").move_to([0, 3, 0])

        self.play(Transform(eqnText,eqnText2), FadeOut(self.axes), FadeOut(SinRects2), FadeOut(tText), run_time=1)



class SineWavePOD(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": 5},
                y_axis_config={"tick_frequency": 70},
                **kwargs
            )
    def setup(self):
        GraphScene.setup(self)
    
    def S_Diag_Function(self, x):
        global Smatrix
        xx=math.floor(x)
        return Smatrix[xx]

    def U_Function(self, x):
        global Umatrix, iVal
        xx=math.floor(x)
        return Umatrix[xx, iVal]
    
    def construct(self):
        global t, omega, k
        
        

        lastI=125
        deltaX=np.pi/64
        #Creates vectors
        time_length=61
        x_vector=np.arange(0,4*np.pi,deltaX)
        A_Matrix=[]   
        pixelVec=[]
        
        nRange=time_length 
        #nRange=6 
        
        #Setup axes
        self.graph_origin = [0, -3.5, 0]
        self.x_axis_width = 3
        self.y_axis_height = 2
        self.x_min = 0
        self.x_max = nRange -1        
        self.y_min = 0   
        self.y_max = 100
        self.x_axis_label = "Mode"
        self.y_axis_label = "$\sigma$"
        

        pxHei=0.02
        pxWid=0.035 
        
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-1,vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])  


        for i in range(0,nRange):
            t=t+0.05
            A_vector=np.sin(omega*t - k*x_vector)
            A_Matrix.append(A_vector)
            
            for j in range(0,len(x_vector)):
                xVal=3+i*pxWid
                yVal=2-j*pxHei
                clr=cpick.to_rgba(A_vector[j]) 
                pixelVec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            

        AllAmatrix=VGroup(*pixelVec)
            
        eqnText2 = MathTex(r"sin(\omega t-kx) = \underbrace{sin(\omega t)cos(-kx)}_{Mode 1} + \underbrace{cos(\omega t)sin(-kx)}_{Mode 2}").move_to([0, 3, 0])

        self.add(AllAmatrix,eqnText2)
        self.wait(1)

        #Performs SVD 
        U, S, V = np.linalg.svd(np.transpose(A_Matrix), full_matrices=False)
        #pdb.set_trace()

        global Smatrix        
        global Umatrix
        Smatrix=S
        Umatrix=U

        Uvec=[]
        for i in range(0,nRange):        
            for j in range(0,len(x_vector)):
                xVal=-3+i*pxWid
                yVal=1.5-j*pxHei
                clr=cpick.to_rgba(U[j,i]*10) 
                Uvec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            
        Ugroup=VGroup(*Uvec)
        
        Svec=[]
        Svec2=[]
        for i in range(0,len(S)):            
            for j in range(0,len(S)): 
                xVal=0+i*pxWid
                yVal=1.5-j*pxWid
                if i==j:
                    clr=cpick.to_rgba(S[i]) 
                else:
                    clr=cpick.to_rgba(0) 
                Svec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
                Svec2.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            
        Sgroup=VGroup(*Svec)
        SgroupBkp=VGroup(*Svec2)

        Vvec=[]
        for i in range(0,nRange):        
            for j in range(0,nRange):
                xVal=2.5+i*pxWid
                yVal=1.5-j*pxWid
                clr=cpick.to_rgba(V[j,i]*3) 
                Vvec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            
        V_group=VGroup(*Vvec)

        #Letters
        Atext = MathTex(r"A").move_to([-5, 6, 0])
        Eqtext = MathTex(r"=").move_to([-3.4, 6, 0])
        Utext = MathTex(r"U").move_to([-1.75, 6, 0])
        Stext = MathTex(r"\Sigma").move_to([1, 6, 0])
        Vtext = MathTex(r"V").move_to([3.5, 6, 0])

        self.add(Atext, Eqtext, Utext, Stext, Vtext)

        self.play(ApplyMethod(AllAmatrix.shift,[-9, -0.5, 0]), Create(Ugroup), Create(Sgroup), Create(V_group),
            ApplyMethod(eqnText2.move_to,[0, 6, 0]), ApplyMethod(Atext.move_to,[-5, 2.5, 0]), ApplyMethod(Eqtext.move_to,[-3.4, 2.5, 0]), 
            ApplyMethod(Utext.move_to,[-1.75, 2.5, 0]), ApplyMethod(Stext.move_to,[1, 2.5, 0]), ApplyMethod(Vtext.move_to,[3.5, 2.5, 0]), lag_ratio=0.002, run_time=4)

        self.wait(1)

        #Does the S matrix        
        self.setup_axes(animate=True)

        EigenGraph = self.get_graph(self.S_Diag_Function).set_color(YELLOW)
        EigenRects = self.get_riemann_rectangles(EigenGraph, dx=1)

        self.play(Transform(Sgroup, EigenRects), run_time=2)
        self.wait(1)
        self.play(Transform(Sgroup, SgroupBkp), run_time=2)

        #Creates a matrix that multiplies U and sigma
        

        USvec=[]
        FirstUSvecs=[]
        for i in range(0,nRange):    
            USvecTemp=[]    
            for j in range(0,len(x_vector)):
                xVal=-3+i*pxWid
                yVal=1.5-j*pxHei
                clr=cpick.to_rgba(U[j,i]*S[i]) 
                USvec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
                if i<2:
                    USvecTemp.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            if i<2:
                USvecGroup=VGroup(*USvecTemp)
                FirstUSvecs.append(USvecGroup)
            
        USgroup=VGroup(*USvec)    

        self.wait(1)

        self.play(Transform(Sgroup,USgroup), ApplyMethod(Utext.move_to,[-1.875, 2.5, 0]), ApplyMethod(Stext.move_to,[-1.55, 2.5, 0]), 
            ApplyMethod(Vtext.move_to,[0.95, 2.5, 0]), ApplyMethod(V_group.move_to,[1, 0.4325, 0]),FadeOut(self.axes), run_time=2)

        self.remove(Ugroup)
        self.wait(1)
        

class UsigmaAndTravelingReconstruction(GraphScene,MovingCameraScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": 18},
                y_axis_config={"tick_frequency": 0.25},
                **kwargs
            )
    def setup(self):
        MovingCameraScene.setup(self)
        GraphScene.setup(self)
    

    def U_Function_Mode1(self, x):
        global Umatrix, Smatrix, Vmatrix, iVal
        xx=math.floor(x)
        m=0
        if iVal==-1:
            vVal=0.1784
        else:
            vVal=Vmatrix[m, iVal]
        return Umatrix[xx, m]*Smatrix[m]*vVal
    
    def U_Function_Mode2(self, x):
        global Umatrix, Smatrix, Vmatrix, iVal
        xx=math.floor(x)
        m=1
        if iVal==-1:
            vVal=0.1833
        else:
            vVal=Vmatrix[m, iVal]
        return Umatrix[xx, m]*Smatrix[m]*vVal
    
    def SumModes1And2(self, x):
        global Umatrix, Smatrix, Vmatrix, iVal
        xx=math.floor(x)
        res = Umatrix[xx, 0]*Smatrix[0]*Vmatrix[0, iVal] + Umatrix[xx, 1]*Smatrix[1]*Vmatrix[1, iVal]
        return res

    def construct(self):
        global t, omega, k
                

        lastI=125
        deltaX=np.pi/64
        #Creates vectors
        time_length=61
        x_vector=np.arange(0,4*np.pi,deltaX)
        A_Matrix=[]   
        pixelVec=[]
        
        nRange=time_length 
        #nRange=6 
        
        #Setup axes
        self.graph_origin = [-2, 0, 0]
        self.x_axis_width = 8
        self.y_axis_height = 6
        self.x_min = 0
        self.x_max = len(x_vector) -1        
        self.y_min = -1   
        self.y_max = 1
        self.x_axis_label = "x"
        self.y_axis_label = ""
        

        pxHei=0.02
        pxWid=0.035 
        
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-1,vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])  


        for i in range(0,nRange):
            t=t+0.05
            A_vector=np.sin(omega*t - k*x_vector)
            A_Matrix.append(A_vector)
            
            for j in range(0,len(x_vector)):
                xVal=3+i*pxWid
                yVal=2-j*pxHei
                clr=cpick.to_rgba(A_vector[j]) 
            
            
        #Performs SVD 
        U, S, V = np.linalg.svd(np.transpose(A_Matrix), full_matrices=False)

        global Smatrix        
        global Umatrix      
        global Vmatrix
        global iVal
        Smatrix=S
        Umatrix=U
        Vmatrix=V

        Ugroup=[]
        for i in range(0,2):  
            Uvec=[]      
            for j in range(0,len(x_vector)):
                xVal=-3+i*pxWid
                yVal=1.5-j*pxHei
                clr=cpick.to_rgba(U[j,i]*S[i]) 
                Uvec.append(Rectangle(height=pxWid, width=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
            
            Ugroup.append(VGroup(*Uvec))

        self.add(Ugroup[0])
        self.add(Ugroup[1])
    
        
        self.wait(1)

        #Transforms the modes into charts 
        TextYaxis=TextMobject("$u_1(x) \sigma_i$").set_color(WHITE).move_to([-1.75, 3.5, 0])     
        self.setup_axes(animate=False)
        self.add(TextYaxis)

        iVal=-1
        Mode1Graph = self.get_graph(self.U_Function_Mode1).set_color(YELLOW)
        Mode1Bars = self.get_riemann_rectangles(Mode1Graph, dx=1, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5)        
        Mode2Graph = self.get_graph(self.U_Function_Mode2).set_color(BLUE_A)
        Mode2Bars = self.get_riemann_rectangles(Mode2Graph, dx=1, show_signed_area=False, start_color=BLUE_A, end_color=BLUE_A, fill_opacity=0.5)


        TextMode1=TextMobject("Mode 1 $\sim cos(-kx)$").set_color(YELLOW).scale(0.75).move_to([7.75, 2.5, 0])
        TextMode2=TextMobject("Mode 2 $\sim sin(-kx)$").set_color(BLUE_A).scale(0.75).move_to([6.75, 3.25, 0])

        print("====Creating line charts======")
        self.play(Transform(Ugroup[0], Mode1Bars), run_time=2)
        self.play(Transform(Ugroup[1], Mode2Bars), run_time=2)
        self.wait(1)

        self.play(self.camera.frame.animate.move_to([3,0,0]))

        self.play(Create(TextMode1))
        self.wait(1)        
        self.play(Create(TextMode2))
        self.wait(1)        
        coskx = MathTex(r"cos(-kx)").move_to([3.4, 5.35, 0]).set_color(YELLOW)
        sinkx = MathTex(r"sin(-kx)").move_to([7.56, 5.35, 0]).set_color(BLUE_A)
        eqnText2 = MathTex(r"sin(\omega t-kx) = \underbrace{sin(\omega t)cos(-kx)}_{Mode 1} + \underbrace{cos(\omega t)sin(-kx)}_{Mode 2}").move_to([3, 5, 0])

        self.add(eqnText2)
        self.play(self.camera.frame.animate.move_to([3,2.5,0]), Transform(TextMode1, coskx), Transform(TextMode2, sinkx), run_time=2)

        self.wait(2)        
        self.play(self.camera.frame.animate.move_to([3,0,0]))



        self.wait(2)
        TextYaxis2=TextMobject("$u_i(x) \sigma_i v_i(t)$").set_color(WHITE).move_to([-1.75, 3.5, 0])   

        iVal=0
        Mode1Graph_0 = self.get_graph(self.U_Function_Mode1).set_color(YELLOW)
        Mode1Bars_0 = self.get_riemann_rectangles(Mode1Graph_0, dx=1, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5)        
        Mode2Graph_0 = self.get_graph(self.U_Function_Mode2).set_color(BLUE_A)
        Mode2Bars_0 = self.get_riemann_rectangles(Mode2Graph_0, dx=1, show_signed_area=False, start_color=BLUE_A, end_color=BLUE_A, fill_opacity=0.5)

        self.play(Transform(Ugroup[0], Mode1Bars_0), Transform(Ugroup[1], Mode2Bars_0), Transform(TextYaxis, TextYaxis2), run_time=1.5)  

        self.wait(1)

        for i in range(1, 15):
            iVal=i
            Mode1Graph_0 = self.get_graph(self.U_Function_Mode1).set_color(YELLOW)
            Mode1Bars_0 = self.get_riemann_rectangles(Mode1Graph_0, dx=1, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5)        
            Mode2Graph_0 = self.get_graph(self.U_Function_Mode2).set_color(BLUE_A)
            Mode2Bars_0 = self.get_riemann_rectangles(Mode2Graph_0, dx=1, show_signed_area=False, start_color=BLUE_A, end_color=BLUE_A, fill_opacity=0.5)

            self.play(Transform(Ugroup[0], Mode1Bars_0), Transform(Ugroup[1], Mode2Bars_0), Transform(TextYaxis, TextYaxis2), run_time=2/15)  

        
        self.wait(1)
        
        #Sum the two waves
        TextYaxis3=TextMobject("$u_1(x) \sigma_1 v_1(t) + u_2(x) \sigma_2 v_2(t)$").set_color(WHITE).move_to([.25, 3.5, 0])  
        self.play(Transform(TextYaxis, TextYaxis3), run_time=1.5)  


        for i in range(0,len(Ugroup[0].submobjects)):
            R1=Ugroup[0].submobjects[i].get_vertices()
            R2=Ugroup[1].submobjects[i].get_vertices()
            R1y=np.max(abs(R1[:,1])) * np.sign(np.sum(np.sign(R1[:,1])))
            #R1 is fixed
            if i<10:
                rt=0.5
            else:
                rt=0.15
            self.play(ApplyMethod(Ugroup[1].submobjects[i].shift,[0,R1y,0]),run_time=rt)

        self.wait(1)

        #============================
        #Create the summed wave here
        IndividualGraph = self.get_graph(self.SumModes1And2).set_color(YELLOW)
        IndividualWave = self.get_riemann_rectangles(IndividualGraph, dx=1, show_signed_area=False, start_color=YELLOW, end_color=BLUE_A, fill_opacity=1)

        self.play(FadeOut(Ugroup[0]), FadeOut(Ugroup[1]), FadeIn(IndividualWave))

        for i in range(15, time_length):
            iVal=i
            IndividualGraph2 = self.get_graph(self.SumModes1And2).set_color(YELLOW)
            IndividualWave2 = self.get_riemann_rectangles(IndividualGraph2, dx=1, show_signed_area=False, start_color=YELLOW, end_color=BLUE_A, fill_opacity=1)
            self.play(Transform(IndividualWave,IndividualWave2), run_time=2/15)

        self.wait(1)


class ModesCanOnlyPulsate(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": 18},
                y_axis_config={"tick_frequency": 0.25},
                **kwargs
            )
    def setup(self):
        GraphScene.setup(self)
    

    def U_Function_Mode1(self, x):
        global t, omega, k
        return np.cos(omega*t)*np.sin(- k*x)
    
    def U_Function_Mode2(self, x):
        global t, omega, k
        return np.sin(omega*t)*np.cos(- k*x)
    
    def SumModes1And2(self, x):
        global Umatrix, Smatrix, Vmatrix, iVal
        xx=math.floor(x)
        res = Umatrix[xx, 0]*Smatrix[0]*Vmatrix[0, iVal] + Umatrix[xx, 1]*Smatrix[1]*Vmatrix[1, iVal]
        return res

    def construct(self):
        global t, omega, k
                

        lastI=125
        deltaX=np.pi/64
        #Creates vectors
        time_length=61
        x_vector=np.arange(0,4*np.pi,deltaX)
        A_Matrix=[]   
        pixelVec=[]
        
        nRange=time_length 
        #nRange=6 
        
        #Setup axes
        self.graph_origin = [-4, 0, 0]
        self.x_axis_width = 8
        self.y_axis_height = 6
        self.x_min = 0
        self.x_max = 4*np.pi        
        self.y_min = -1   
        self.y_max = 1
        self.x_axis_label = "x"
        self.y_axis_label = ""

        self.setup_axes(animate=False)
        

        t=0
        Mode1Graph_0 = self.get_graph(self.U_Function_Mode1).set_color(YELLOW)
        Mode1Bars_0 = self.get_riemann_rectangles(Mode1Graph_0, dx=deltaX, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5) 
        Mode2Graph_0 = self.get_graph(self.U_Function_Mode2).set_color(YELLOW)
        Mode2Bars_0 = self.get_riemann_rectangles(Mode2Graph_0, dx=deltaX, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5) 

        self.add(Mode1Bars_0, Mode2Bars_0)  
        self.wait(.1)

        for i in range(1, 90):
            t=0.05*i
            Mode1Graph_1 = self.get_graph(self.U_Function_Mode1).set_color(YELLOW)
            Mode1Bars_1 = self.get_riemann_rectangles(Mode1Graph_1, dx=deltaX, show_signed_area=False, start_color=YELLOW, end_color=YELLOW, fill_opacity=0.5)        
            Mode2Graph_1 = self.get_graph(self.U_Function_Mode2).set_color(BLUE_A)
            Mode2Bars_1 = self.get_riemann_rectangles(Mode2Graph_1, dx=deltaX, show_signed_area=False, start_color=BLUE_A, end_color=BLUE_A, fill_opacity=0.5)


            self.play(Transform(Mode1Bars_0, Mode1Bars_1), Transform(Mode2Bars_0, Mode2Bars_1), run_time=2/15)  

        
        self.wait(1)
        

































        # #Sum the two waves
        # TextYaxis3=TextMobject("$u_1(x) \sigma_1 v_1(t) + u_2(x) \sigma_2 v_2(t)$").set_color(WHITE).move_to([.25, 3.5, 0])  
        # self.play(Transform(TextYaxis, TextYaxis3), run_time=1.5)  


        # for i in range(0,len(Ugroup[0].submobjects)):
        #     R1=Ugroup[0].submobjects[i].get_vertices()
        #     R2=Ugroup[1].submobjects[i].get_vertices()
        #     R1y=np.max(abs(R1[:,1])) * np.sign(np.sum(np.sign(R1[:,1])))
        #     #R1 is fixed
        #     if i<10:
        #         rt=0.5
        #     else:
        #         rt=0.15
        #     self.play(ApplyMethod(Ugroup[1].submobjects[i].shift,[0,R1y,0]),run_time=rt)

        # self.wait(1)

        # #============================
        # #Create the summed wave here
        # IndividualGraph = self.get_graph(self.SumModes1And2).set_color(YELLOW)
        # IndividualWave = self.get_riemann_rectangles(IndividualGraph, dx=1, show_signed_area=False, start_color=YELLOW, end_color=BLUE_A, fill_opacity=1)

        # self.play(FadeOut(Ugroup[0]), FadeOut(Ugroup[1]), FadeIn(IndividualWave))

        # for i in range(15, time_length):
        #     iVal=i
        #     IndividualGraph2 = self.get_graph(self.SumModes1And2).set_color(YELLOW)
        #     IndividualWave2 = self.get_riemann_rectangles(IndividualGraph2, dx=1, show_signed_area=False, start_color=YELLOW, end_color=BLUE_A, fill_opacity=1)
        #     self.play(Transform(IndividualWave,IndividualWave2), run_time=2/15)

        # self.wait(1)























        





        
        
        

        






