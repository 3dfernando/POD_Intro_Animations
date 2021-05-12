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


vField = loadmat("VelocityFieldData.mat")
vx = vField["vx"]
vy = vField["vy"]

skip = 3
dvdx = np.subtract(vy[0:-1:skip, 1::skip, :], vy[0:-1:skip, 0:-1:skip, :])
dudy = np.subtract(vx[1::skip, 0:-1:skip, :], vx[0:-1:skip, 0:-1:skip, :])
omega = dudy - dvdx  # Computes vorticity here


# Position of the sensors that will make to the 3D point cloud representation
sensor1row = 19
sensor1col = 15
sensor2row = 27
sensor2col = 21
sensor3row = 19
sensor3col = 35

scaleFactor = 100

omega = omega * scaleFactor #Scales Omega
omegaMax=np.max(np.abs(omega)) 

np.random.seed(0)
noiseStd = 0.25
omegaNoisy1=np.add(omega[sensor1row, sensor1col, :], noiseStd*np.random.randn(1,omega.shape[2]))
omegaNoisy2=np.add(omega[sensor2row, sensor2col, :], noiseStd*np.random.randn(1,omega.shape[2]))
omegaNoisy3=np.add(omega[sensor3row, sensor3col, :], noiseStd*np.random.randn(1,omega.shape[2]))

v_index=0


class GraphRepresentation(MovingCameraScene):  
    def setup(self):
        MovingCameraScene.setup(self)
    def construct(self):
        self.camera.frame.save_state()

        dot = Dot(ORIGIN , radius=0.075)
        dot2 = Dot([omega[sensor1row, sensor1col, 0],omega[sensor2row, sensor2col, 0],omega[sensor3row, sensor3col, 0]], radius=0.075)
    
        #============1=============
        #So let's first consider we want to represent two measurements, acquired by two different instruments at the same time.

        #Grid animation   
        print("============1=============")
        grid = NumberPlane(axis_config={"stroke_width": 5, "stroke_color": YELLOW})
        self.add(grid)
        self.play(Create(grid, run_time=2, lag_ratio=0.2),)
        self.play(FadeIn(dot))

        line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
        
        hLine = DashedLine(dot2.get_center(), [omega[sensor1row, sensor1col, 0], 0, 0]).set_color(GRAY)
        vLine = DashedLine(dot2.get_center(), [0, omega[sensor2row, sensor2col, 0],0]).set_color(GRAY)
        zLine = DashedLine(dot2.get_center(), [0, 0, omega[sensor3row, sensor3col, 0]]).set_color(GRAY)

        b1 = Brace(line,direction=[0, -1, 0])
        b2 = Brace(line,direction=[1, 0, 0])
        b1Text = b1.get_text("Measurement 1")
        b2Text = b2.get_text("Measurement 2")

        self.play(Transform(dot,dot2), GrowFromPoint(line,ORIGIN), GrowFromPoint(hLine,ORIGIN), GrowFromPoint(vLine,ORIGIN))
        self.wait(1)        
        self.play(ApplyMethod(grid.set_opacity,0.5))
        self.play(Create(b1), Create(b1Text))
        self.wait(1)
        self.play(Create(b2), Create(b2Text))
        self.wait(1)

        #============2=============
        #Now let's assume we take a time series of these measurements:
        print("============2=============")
        self.add(dot2)
        self.play(FadeOut(line),FadeOut(hLine),FadeOut(vLine),FadeOut(b1),FadeOut(b2),FadeOut(b1Text),FadeOut(b2Text))

        newDots=[]
        newDotsObjects=[]
        for i in range(1,omega.shape[2]):  # range(1,omega.shape[2]):            
            newDot=Dot([omegaNoisy1[0,i],omegaNoisy2[0,i],omegaNoisy3[0,i]], radius=0.075)
            if i<10:
                rt=0.3
            else:
                rt=0.15
            newDots.append(newDot.get_center())
            newDotsObjects.append(newDot)
            self.play(Transform(dot, newDot), run_time=rt)
            self.add(newDotsObjects[-1])

        self.wait(1)
        
        #============3=============
        #We can still represent each measurement as the original vectors, but it might be more insightful to choose a different set of basis vectors to represent
        # the data set as a whole:
        print("============3=============")

        Vector1 = Arrow(ORIGIN, [1, 0, 0], buff=0, color=YELLOW)
        Vector2 = Arrow(ORIGIN, [0, 1, 0], buff=0, color=YELLOW)
        
        OriginalVector1 = Arrow(ORIGIN, [1, 0, 0], buff=0, color=YELLOW)
        OriginalVector2 = Arrow(ORIGIN, [0, 1, 0], buff=0, color=YELLOW)
        
        BasisVector1a = Arrow(ORIGIN, [np.cos(np.pi/4), np.sin(np.pi/4), 0], buff=0, color=YELLOW)
        BasisVector2a = Arrow(ORIGIN, [np.cos(-np.pi/3), np.sin(-np.pi/3), 0], buff=0, color=YELLOW)

        BasisVector1b = Arrow(ORIGIN, [np.cos(-0.8*np.pi/2), np.sin(-0.8*np.pi/2), 0], buff=0, color=YELLOW)
        BasisVector2b = Arrow(ORIGIN, [np.cos(-3*np.pi/4), np.sin(-3*np.pi/4), 0], buff=0, color=YELLOW)
        
        BasisVector1c = Arrow(ORIGIN, [np.cos(3*np.pi/4), np.sin(3*np.pi/4), 0], buff=0, color=YELLOW)
        BasisVector2c = Arrow(ORIGIN, [np.cos(np.pi/4), np.sin(np.pi/4), 0], buff=0, color=YELLOW)

        self.play(Create(Vector1),Create(Vector2))
        self.wait(1)
        self.play(Transform(Vector1, BasisVector1a),Transform(Vector2, BasisVector2a))
        self.wait(1.5)
        self.play(Transform(Vector1, BasisVector1b),Transform(Vector2, BasisVector2b))
        self.wait(1.5)
        self.play(Transform(Vector1, BasisVector1c),Transform(Vector2, BasisVector2c))
        self.wait(1.5)

        #============4=============
        #This is the principle of Proper Orthogonal Decomposition: How to choose a set of basis vectors such that we better represent the data set we're handling.

        #============5=============
        #Let's first note that in this case we have more data points than dimensions. 
        print("============5=============")
        self.play(Transform(Vector1, OriginalVector1),Transform(Vector2, OriginalVector2))

        textVec1 = Text("1", font='CMU Serif', color=YELLOW).next_to(Vector1.get_end(), UR)
        textVec2 = Text("2", font='CMU Serif', color=YELLOW).next_to(Vector2.get_end(), UR)

        self.play(Create(textVec1), run_time=0.2)
        self.wait(0.5)
        self.play(Create(textVec2), run_time=0.2)
        self.wait(0.5)

        allDotTexts=[]
        txDot = Text("1", font='CMU Serif', size=0.3).next_to(dot2.get_center(), UR)
        allDotTexts.append(txDot)
        self.play(Create(allDotTexts[-1]))
        for i in range(0, len(newDots)): 
            loc=newDots[i]
            txDot = Text(str(i+2), font='CMU Serif', size=0.3).next_to(loc, UR)
            allDotTexts.append(txDot)
            self.play(Create(allDotTexts[-1]), run_time=0.1)

        self.wait(2)

        for i in range(0, len(allDotTexts)):
            self.play(FadeOut(allDotTexts[i]), run_time=0.025)
       
        self.play(FadeOut(textVec1),FadeOut(textVec2), run_time=0.2)
        self.play(FadeOut(Vector1),FadeOut(Vector2), run_time=0.2)
        self.wait(2)
    
class VectorProjections(GraphScene, MovingCameraScene): 
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": np.pi/4},
                y_axis_config={"tick_frequency": 100},
                **kwargs
            )
    def setup(self):
        MovingCameraScene.setup(self)
        GraphScene.setup(self)

    def len_A_dot_Eig(self,theta):
        vx=np.cos(theta)
        vy=np.sin(theta)

        product=np.linalg.norm(np.add(omegaNoisy1*vx, omegaNoisy2*vy))
        return product**2

    def construct(self):
        self.camera.frame.save_state()
        


        dot = Dot(ORIGIN , radius=0.075)
        dot2 = Dot([omega[sensor1row, sensor1col, 0],omega[sensor2row, sensor2col, 0],omega[sensor3row, sensor3col, 0]], radius=0.075)
    
        #============0=============
        #Reloads the previous scene
        print("============0=============")

        grid = NumberPlane(axis_config={"stroke_width": 5, "stroke_color": YELLOW}, x_axis_config={"x_min": -20, "x_max": 20})
        grid.set_opacity(0.5)
        self.add(grid)     

        newDotsObjects=[]
        for i in range(0,omega.shape[2]): 
            newDot=Dot([omegaNoisy1[0,i],omegaNoisy2[0,i],omegaNoisy3[0,i]], radius=0.075)
            newDot.set_opacity(0.5)
            newDotsObjects.append(newDot)
            self.add(newDotsObjects[-1])

        self.wait(1)

        #============6=============
        print("============6=============")
        # So this problem has really no exact solution, and what we are looking for here is 
        #a way to optimize the representation based on some criterion. 
        #The POD algorithm is defined such that it maximizes the energy along each one of its dimensions, sequentially.
        #Let's define the first basis vector we seek to find. The energy of a point along the basis vector will be defined as
        # the length of the projection of the point onto this basis vector, squared:
        theta=np.pi/3
        dtheta=np.pi/24
        EigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)
        txEig = MathTex(r"\vec{e}_1").next_to(EigenVector.get_end(), UR)

        self.play(Create(EigenVector), Create(txEig))


        #Projects the dot to the vector
        self.wait(2)
        self.play(ApplyMethod(newDotsObjects[0].set_color,YELLOW), run_time=0.03)
        self.play(ApplyMethod(newDotsObjects[0].set_opacity,1), run_time=0.8)
        self.wait(0.5)
        projection=np.dot(newDotsObjects[0].get_center(),EigenVector.get_end())
        projLine1 = DashedLine(newDotsObjects[0].get_center(), projection*EigenVector.get_end()).set_color(GRAY)
        projLine2 = Line(ORIGIN, projection*EigenVector.get_end(), stroke_width=3).set_color(GRAY)
        self.play(Create(projLine1), Create(projLine2), run_time=1.2)

        self.wait(2)
        self.play(self.camera.frame.animate.move_to([3,0,0]))

        #=============================
        #Create graph
        border = Rectangle(height=6, width=7, color=WHITE).move_to([6, 0, 0])
        bg = Rectangle(height=6, width=7).move_to([6, 0, 0])
        bg.set_fill(BLACK, opacity=1)

        self.play(FadeIn(bg), Create(border))

        self.graph_origin = [3, -2, 0]
        self.x_axis_width = 6
        self.y_axis_height = 4
        self.x_min = 0
        self.x_max = np.pi*2
        self.y_min = 0
        self.y_max = 600
        self.x_axis_label = "$\\theta$"
        self.y_axis_label = "$\\mid A\\cdot \\vec{e}_1 \\mid ^2$"
        self.setup_axes(animate=True)

        EigenGraph = self.get_graph(self.len_A_dot_Eig).set_color(YELLOW)
        EigenRects = self.get_riemann_rectangles(EigenGraph, dx=dtheta)

        currRect = math.floor(theta/dtheta)

        rectVertices=EigenRects[currRect].get_vertices()
        rectBottom=rectVertices[[2, 3],:]
        rectTop=rectVertices[[1, 0],:]

        fracRect=1/len(newDotsObjects)
        rectLineCoords=np.add(rectBottom,fracRect*np.subtract(rectTop,rectBottom))
        rectLineBuildup=Line(start=rectLineCoords[0,:], end=rectLineCoords[1,:], stroke_color=YELLOW, stroke_width=3)

        self.play(ReplacementTransform(projLine2, rectLineBuildup), run_time=1)
        self.wait(1)

        #============7=============
        print("============7=============")
        #We will sum these squared lengths all together, for every data point we have. This will give us a number to work with, which is actually the Frobenius norm of the dot product between the data matrix A and this tentative eigenvector e_1.
        for i in range(1,omega.shape[2]): #omega.shape[2]
            self.play(ApplyMethod(newDotsObjects[i].set_color,YELLOW), run_time=0.03)
            self.play(FadeOut(projLine1), FadeOut(projLine2),ApplyMethod(newDotsObjects[i].set_opacity,1), run_time=0.4)
            self.wait(0.5)
            projection=np.dot(newDotsObjects[i].get_center(),EigenVector.get_end())
            projLine1 = DashedLine(newDotsObjects[i].get_center(), projection*EigenVector.get_end()).set_color(GRAY)
            projLine2 = Line(ORIGIN, projection*EigenVector.get_end(), stroke_width=3).set_color(GRAY)
            self.play(Create(projLine1), Create(projLine2), run_time=0.5)
            
            fracRect=(i+1)/len(newDotsObjects)
            rectLineCoords=np.add(rectBottom,fracRect*np.subtract(rectTop,rectBottom))
            self.play(FadeOut(rectLineBuildup),run_time=0.05)
            rectLineBuildup=Line(start=rectLineCoords[0,:], end=rectLineCoords[1,:], stroke_color=YELLOW, stroke_width=3)
            self.play(ReplacementTransform(projLine2, rectLineBuildup), run_time=0.5)

            if i<6:
                self.wait(1)
            
            # if i==4:               
            #     circle=Ellipse(width=2.5,height=1.1,color=WHITE).move_to(self.y_axis_label_mob)
            #     self.play(Create(circle))
            #     self.wait(1)
            #     self.play(FadeOut(circle))
            #     self.wait(2)
                
        self.play(Create(EigenRects[currRect]),FadeOut(rectLineBuildup), FadeOut(projLine1, projLine2))
        self.wait(2)



        
        #============8=============
        print("============8=============")
        #Now, we will do that for every possible eigenvector in this 2D space. 
        # In 2D, we can plot this overall norm as a function of the eigenvector angle theta, with respect to the origin. 
        # For higher dimensional data, plotting this function might not be as simple, though.

        for t in range(7,3,-1): #range(7,3,-1)
            currRect=t
            theta=dtheta*currRect
            NewEigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)


            allProjLine1=[]
            allProjLine2=[]
            for i in range(1,omega.shape[2]): 
                projection=np.dot(newDotsObjects[i].get_center(),NewEigenVector.get_end())
                projLine1 = DashedLine(newDotsObjects[i].get_center(), projection*NewEigenVector.get_end()).set_color(GRAY)
                projLine2 = Line(ORIGIN, projection*NewEigenVector.get_end(), stroke_width=3).set_color(GRAY)
                
                allProjLine1.append(projLine1)
                allProjLine2.append(projLine2)

            projLineGroup = VGroup(*allProjLine1, *allProjLine2)

            if t>5:
                rt=2
            else:
                rt=0.5
            
            endVec=NewEigenVector.get_end()*1.4

            txEig.generate_target()
            txEig.target.move_to(endVec)

            self.play(Transform(EigenVector, NewEigenVector), MoveToTarget(txEig), run_time=rt/2) #Try this (not sure)
            self.play(Create(projLineGroup),run_time=rt)
            self.play(Transform(projLineGroup, EigenRects[currRect]),run_time=rt)
        
        
        for t in range(3,-1,-1):
            currRect=t
            theta=dtheta*currRect
            NewEigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)

            endVec=NewEigenVector.get_end()*1.4

            txEig.generate_target()
            txEig.target.move_to(endVec)
            self.play(Create(EigenRects[currRect]), Transform(EigenVector, NewEigenVector), MoveToTarget(txEig), run_time=0.3)
        
        
        for t in range(47,8,-1):
            currRect=t
            theta=dtheta*currRect
            NewEigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)

            endVec=NewEigenVector.get_end()*1.4

            txEig.generate_target()
            txEig.target.move_to(endVec)
            self.play(Create(EigenRects[currRect]), Transform(EigenVector, NewEigenVector), MoveToTarget(txEig), run_time=0.3)
        
        self.wait(2)
        #============9=============
        print("============9=============")
        #Note that this dot product function has two maxima. We can pick either one, to represent the eigenvector along the direction of maximal energy.
        maxRect2 = 21
        maxRect1 = 21 + 24
        
        theta=dtheta*maxRect1
        NewEigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)
        endVec=NewEigenVector.get_end()*1.4

        txEig.generate_target()
        txEig.target.move_to(endVec)
        self.play(FadeToColor(EigenRects[maxRect1], YELLOW),Transform(EigenVector, NewEigenVector), MoveToTarget(txEig), run_time=1)
        self.wait(1.5)

        theta=dtheta*maxRect2
        NewEigenVector = Arrow(ORIGIN, [np.cos(theta), np.sin(theta), 0], buff=0, color=WHITE, stroke_width=10)
        endVec=NewEigenVector.get_end()*1.4

        txEig.generate_target()
        txEig.target.move_to(endVec)
        self.play(FadeToColor(EigenRects[maxRect2], YELLOW),Transform(EigenVector, NewEigenVector), MoveToTarget(txEig), run_time=1)
        self.wait(2)

        self.play(self.camera.frame.animate.move_to([0,0,0]))

        self.wait(2)

class ThreeDimensionalRotations(ThreeDScene): 
    def setup(self):
        ThreeDScene.setup(self)

        #grid = NumberPlane(axis_config={"stroke_width": 5, "stroke_color": YELLOW}, x_axis_config={"x_min": -10, "x_max": 10}, y_axis_config={"y_min": -10, "y_max": 10})
        grid = NumberPlane(axis_config={"stroke_width": 5, "stroke_color": YELLOW}, y_axis_config={"y_min": -10, "y_max": 10})
        grid.set_opacity(0.5)
        self.add(grid)     

        newDotsObjects=[]
        for i in range(0,omega.shape[2]): #omega.shape[2]
            newDot=Sphere(radius=0.075, fill_color=YELLOW, stroke_color=YELLOW)
            newDot.move_to([omegaNoisy1[0,i],omegaNoisy2[0,i],omegaNoisy3[0,i]])
            newDot.set_opacity(0.5)
            newDotsObjects.append(newDot)

        newDotGroup=VGroup(*newDotsObjects)
        self.add(newDotGroup)



        #Computes the actual SVD of this object
        omegaAll = np.vstack((omegaNoisy1,omegaNoisy2,omegaNoisy3))
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)

        self.wait(1)

        #============10=============
        print("============10=============")        
        #Now that we chose our first basis vector, we choose the second eigenvector to be perpendicular to this vector.
        # In 2D, this becomes an obvious solution – but in a higher dimensional case we have to solve a similar problem:
        EigenVector1 = Arrow(ORIGIN, omegaU[:,0], buff=0, color=WHITE, stroke_width=10)       
        EigenVector2 = Arrow(ORIGIN, omegaU[:,1], buff=0, color=WHITE, stroke_width=10)       
        EigenVector3 = Arrow(ORIGIN, omegaU[:,2], buff=0, color=WHITE, stroke_width=10)    
   
        
        self.add(EigenVector1);
        self.play(Create(EigenVector3));
        self.wait(1)

        self.move_camera(0.4*np.pi/2, -0.4*np.pi, run_time=1.5)
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(2)

        self.play(Rotate(EigenVector3, angle=PI/4, axis=omegaU[:,0],about_point=ORIGIN))
        self.wait(1)
        self.play(Rotate(EigenVector3, angle=-PI/4, axis=omegaU[:,0],about_point=ORIGIN))
        self.play(Rotate(EigenVector3, angle=-PI/3, axis=omegaU[:,0],about_point=ORIGIN))
        self.play(Rotate(EigenVector3, angle=-PI/3, axis=omegaU[:,0],about_point=ORIGIN))
        self.play(Rotate(EigenVector3, angle=-PI/2, axis=omegaU[:,0],about_point=ORIGIN))
        self.play(Rotate(EigenVector3, angle=-PI/3, axis=omegaU[:,0],about_point=ORIGIN))
        self.play(Rotate(EigenVector3, angle=-PI/4, axis=omegaU[:,0],about_point=ORIGIN))
        self.wait(1)
        self.play(Rotate(EigenVector3, angle=-PI/4, axis=omegaU[:,0],about_point=ORIGIN), run_time=3)
        self.wait(2)

        #============11=============
        print("============11=============")  
        #Once we find the second best basis vector, then we proceed to the third basis vector, 
        # making it perpendicular to the previous two. Until we run out of dimensions in the data.
        self.play(Create(EigenVector2))
        self.wait(6)
        #In fact, POD really shines when the data set has a very large number of dimensions, 
        # where the underlying data is noisy and complex, but the rules that generated the data are somewhat simple. 

        #============12=============
        print("============12=============")  
        #So let’s have a look at how we’re going to deal with more dimensions, since we can’t use this kind of point cloud plot anymore. 
        # We can represent each data point by a list of numbers, one for each dimension, in a [Nx1] matrix.        
        text3d = MathTex(r"\begin{bmatrix}" + format(omegaNoisy1[0,1], '>0.2f') + " \\\\ " + format(omegaNoisy2[0,1], '>0.2f') + " \\\\ " + format(omegaNoisy3[0,1], '>0.2f') + " \end{bmatrix}")
        self.add_fixed_in_frame_mobjects(text3d)        
        text3d.to_corner(UR)
        text3d.set_opacity(0)

        text3d2=text3d.copy()
        text3d2.set_opacity(1)        
        

        self.play(Transform(newDotGroup[0],text3d2)) 
        self.move_camera(0, 0, run_time=1)       

        self.wait(2)

        #============13=============
        print("============13=============")  
        #This is a good way to represent the data in a computer, but for us humans it might be better to assign a color value for each number in this vector:
        self.play(FadeOut(newDotGroup),FadeOut(EigenVector1), FadeOut(EigenVector2), FadeOut(EigenVector3), FadeOut(grid))
    
class HigherColorDimensions(MovingCameraScene): 
    def construct(self):
        self.camera.frame.save_state()    
        text3d = MathTex(r"\begin{bmatrix}" + format(omegaNoisy1[0,1], '>0.2f') + " \\\\ " + format(omegaNoisy2[0,1], '>0.2f') + " \\\\ " + format(omegaNoisy3[0,1], '>0.2f') + " \end{bmatrix}")
              
        text3d.to_corner(UR)
        
        emptyMatrix = MathTex(r"\begin{bmatrix} \qquad \quad \\ \qquad \quad \\ \qquad \quad  \end{bmatrix}").move_to([1.5, 1, 0])
        
        omega1=DecimalNumber(omegaNoisy1[0,1], num_decimal_places=2,include_sign=True,unit=None).move_to([1.5,1.5,0])
        tracker1=ValueTracker(omegaNoisy1[0,1])
        omega1.add_updater(lambda d: d.set_value(tracker1.get_value()))
        
        omega2=DecimalNumber(omegaNoisy2[0,1], num_decimal_places=2,include_sign=True,unit=None).move_to([1.5,1,0])
        tracker2=ValueTracker(omegaNoisy2[0,1])
        omega2.add_updater(lambda d: d.set_value(tracker2.get_value()))
        
        omega3=DecimalNumber(omegaNoisy3[0,1], num_decimal_places=2,include_sign=True,unit=None).move_to([1.5,0.5,0])
        tracker3=ValueTracker(omegaNoisy3[0,1])
        omega3.add_updater(lambda d: d.set_value(tracker3.get_value()))
        
        self.play(Transform(text3d,emptyMatrix), Create(omega1), Create(omega2), Create(omega3))

        #============13=============
        print("============13=============")  
        ##This is a good way to represent the data in a computer, but most of us will probably appreciate a more visual representation. 
        # For example, we can assign a color to each one of these values according to a color scale:  
        emptyMatrix2 = MathTex(r"\rightarrow \begin{bmatrix} \quad \\ \quad \\ \quad  \end{bmatrix}").next_to(emptyMatrix,RIGHT)
        newText = text3d.copy()
        
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-omegaMax/2,vmax=omegaMax/2)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])

        clr=cpick.to_rgba(omegaNoisy1[0,1])        
        colorRect1=Square(side_length=0.45, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([3.86,1.5,0])
        clr2=cpick.to_rgba(omegaNoisy2[0,1])        
        colorRect2=Square(side_length=0.45, fill_color=rgb_to_color(clr2[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([3.86,1,0])
        clr3=cpick.to_rgba(omegaNoisy3[0,1])        
        colorRect3=Square(side_length=0.45, fill_color=rgb_to_color(clr3[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([3.86,0.5,0])

        self.play(Transform(newText, emptyMatrix2), Create(colorRect1), Create(colorRect2), Create(colorRect3))

        #============14=============
        print("============14=============") 
        newVal=5
        newVal2=-4
        newVal3=0
        clr=cpick.to_rgba(newVal)
        clr2=cpick.to_rgba(newVal2)
        clr3=cpick.to_rgba(newVal3)
        self.play(ApplyMethod(tracker1.set_value, float(newVal)), FadeToColor(colorRect1, rgb_to_color(clr[0:3])),
            ApplyMethod(tracker2.set_value, float(newVal2)), FadeToColor(colorRect2, rgb_to_color(clr2[0:3])),
            ApplyMethod(tracker3.set_value, float(newVal3)), FadeToColor(colorRect3, rgb_to_color(clr3[0:3])), run_time=2)

        self.wait(1)
        
        newVal=-3
        newVal2=2
        newVal3=-5
        clr=cpick.to_rgba(newVal)
        clr2=cpick.to_rgba(newVal2)
        clr3=cpick.to_rgba(newVal3)
        self.play(ApplyMethod(tracker1.set_value, float(newVal)), FadeToColor(colorRect1, rgb_to_color(clr[0:3])),
            ApplyMethod(tracker2.set_value, float(newVal2)), FadeToColor(colorRect2, rgb_to_color(clr2[0:3])),
            ApplyMethod(tracker3.set_value, float(newVal3)), FadeToColor(colorRect3, rgb_to_color(clr3[0:3])), run_time=2)

        self.wait(1)

        newVal=omegaNoisy1[0,1]
        newVal2=omegaNoisy2[0,1]
        newVal3=omegaNoisy3[0,1]
        clr=cpick.to_rgba(newVal)
        clr2=cpick.to_rgba(newVal2)
        clr3=cpick.to_rgba(newVal3)
        self.play(ApplyMethod(tracker1.set_value, float(newVal)), FadeToColor(colorRect1, rgb_to_color(clr[0:3])),
            ApplyMethod(tracker2.set_value, float(newVal2)), FadeToColor(colorRect2, rgb_to_color(clr2[0:3])),
            ApplyMethod(tracker3.set_value, float(newVal3)), FadeToColor(colorRect3, rgb_to_color(clr3[0:3])), run_time=2)

        self.wait(1)

        #============15=============
        print("============15=============") 
        #Makes the specific rectangles in the matrix
        pxWid=0.1
        left_x=-omega.shape[0]*pxWid/2
        bottom_y=-5
        delta=0.11
        
        yVal=left_x+sensor1row*delta
        xVal=bottom_y+sensor1col*delta
        clr=cpick.to_rgba(omega[sensor1row, sensor1col, 0])       
        vectorPixel1=Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0, stroke_color=BLACK).move_to([xVal,yVal,0])

        yVal=left_x+sensor2row*delta
        xVal=bottom_y+sensor2col*delta
        clr=cpick.to_rgba(omega[sensor2row, sensor2col, 0])       
        vectorPixel2=Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0, stroke_color=BLACK).move_to([xVal,yVal,0])

        yVal=left_x+sensor3row*delta
        xVal=bottom_y+sensor3col*delta
        clr=cpick.to_rgba(omega[sensor3row, sensor3col, 0])       
        vectorPixel3=Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0, stroke_color=BLACK).move_to([xVal,yVal,0])


        #Makes rectangles to build the image
        pixel=[]
        for i in range(0,omega.shape[0]):            
            for j in range(0,omega.shape[1]):
                yVal=left_x+i*delta
                xVal=bottom_y+j*delta
                clr=cpick.to_rgba(omega[i, j, 0])       
                pixel.append(Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
        
        pixelGroup=VGroup(*pixel)


        self.play(Transform(colorRect1,vectorPixel1), Transform(colorRect2,vectorPixel2), Transform(colorRect3,vectorPixel3),
            FadeIn(pixelGroup), FadeOut(text3d), FadeOut(omega1), FadeOut(omega2), FadeOut(omega3), FadeOut(newText))
        self.play(FadeOut(colorRect1), FadeOut(colorRect2), FadeOut(colorRect3))
        self.wait(1)
        
        #Makes a vector
        pixelVec=[]
        pxHei=0.0035
        vecWid=0.035
        topLeft=3.5

        for i in range(0,omega.shape[0]):            
            for j in range(0,omega.shape[1]):
                yVal=topLeft-(j+i*omega.shape[0])*pxHei
                xVal=1
                clr=cpick.to_rgba(omega[i, j, 0])
                pixelVec.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
        
        pixelGroup2=VGroup(*pixelVec)
        self.play(Transform(pixelGroup, pixelGroup2), lag_ratio=0.4, run_time=5)
        self.wait(1)

        #============16=============
        print("============16=============") 
        #Builds the first few columns of the matrix here
        for n in range(1,5):            
            pixel=[]
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=left_x+i*delta
                    xVal=bottom_y+j*delta
                    clr=cpick.to_rgba(omega[i, j, n])       
                    pixel.append(Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            
            pixelGroup=VGroup(*pixel)
            self.play(FadeIn(pixelGroup))
            self.wait(0.5)
            pixelVec=[]
            
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=topLeft-(j+i*omega.shape[0])*pxHei
                    xVal=1+n*vecWid
                    clr=cpick.to_rgba(omega[i, j, n])       
                    pixelVec.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            
            pixelGroup2=VGroup(*pixelVec)
            if i==4:
                self.play(FadeIn(pixelGroup2), lag_ratio=0.4, run_time=3)
            else:
                self.play(Transform(pixelGroup, pixelGroup2), lag_ratio=0.4, run_time=3)
            
        
        #============17=============
        print("============17=============") 
        #Builds the next columns of the matrix here
        for n in range(5,omega.shape[2] ):  #omega.shape[2]          
            pixel=[]
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=left_x+i*delta
                    xVal=bottom_y+j*delta
                    clr=cpick.to_rgba(omega[i, j, n])       
                    pixel.append(Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            
            pixelGroup=VGroup(*pixel)
            self.play(FadeIn(pixelGroup))
            self.wait(0.5)
            pixelVec=[]
            
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=topLeft-(j+i*omega.shape[0])*pxHei
                    xVal=1+n*vecWid
                    clr=cpick.to_rgba(omega[i, j, n])       
                    pixelVec.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            
            pixelGroup2=VGroup(*pixelVec)
            self.play(FadeIn(pixelGroup2), lag_ratio=0.4, run_time=1)
        
        #============18=============
        print("============18=============") 
        #Moves the data matrix to the left of the screen

        self.play(self.camera.frame.animate.move_to([2,0,0]))

class SVD_Definition(MovingCameraScene): 
    def construct(self):
        Stext = MathTex(r"\Sigma", height=0.5).move_to([9.25, 4.1, 0])
    
        #============18=============
        print("============18=============") 
        #Computes SVD        
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        omegaU=np.reshape(omegaU, [oShape[0], oShape[1], oShape[2]])

        pxHei=0.0035
        vecWid=0.035
        topLeft=3.5
        #Builds the next columns of the matrix here   
        
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-omegaMax/2,vmax=omegaMax/2)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])

        pixelVec=[]
        for n in range(0,omega.shape[2] ):# [0, 1, 2, 3, 74]:  #
            
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=topLeft-(j+i*omega.shape[0])*pxHei
                    xVal=1+n*vecWid
                    clr=cpick.to_rgba(omega[i, j, n])       
                    pixelVec.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            print(n)
            
        A_Matrix=VGroup(*pixelVec)

        self.add(A_Matrix)            
        self.play(self.camera.frame.animate.move_to([7,1,0]))

        #============19=============
        print("============19=============") 
        Atext = Text("A", font='CMU Serif', color=WHITE)
        Atext.move_to([2.2, 4.1, 0])
        self.play(Create(Atext))
        self.wait(1)


        #============U Matrix=============
        pixelVecU=[]
        for n in range(0,omega.shape[2] ):# [0, 1, 2, 3, 74]:  #
            
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=topLeft-(j+i*omega.shape[0])*pxHei
                    xVal=5+n*vecWid
                    clr=cpick.to_rgba(250*omegaU[i, j, n])       
                    pixelVecU.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            print(n)
            
        U_Matrix=VGroup(*pixelVecU)

        Utext = Text("U", font='CMU Serif', color=WHITE).move_to([6.25, 4.1, 0])
        Eqtext = Text("=", font='CMU Serif', color=WHITE).move_to([4.25, 4.1, 0])
        Eqtext2 = Text("=", font='CMU Serif', color=WHITE).move_to([4.25, 0, 0])

        self.play(Create(Utext), Create(Eqtext), Create(Eqtext2), FadeIn(U_Matrix), lag_ratio=0.4, run_time=1)
        self.wait(2)        
        
        #============S Matrix=============
        pixelVecS=[]
        for i in range(0,omegaS.shape[0]):            
            for j in range(0,omegaS.shape[0]):
                yVal=topLeft-i*vecWid
                xVal=8+j*vecWid
                if i==j:
                    clr=cpick.to_rgba(omegaS[i])    
                else:
                    clr=cpick.to_rgba(0)      
                   
                pixelVecS.append(Rectangle(height=vecWid, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
        
            
        S_Matrix=VGroup(*pixelVecS)


        self.play(Create(Stext), FadeIn(S_Matrix), lag_ratio=0.4, run_time=2)
        self.wait(2)        

        
        #============V Matrix=============
        pixelVecV=[]
        for i in range(0,omegaV.shape[0]):            
            for j in range(0,omegaV.shape[1]):
                yVal=topLeft-i*vecWid
                xVal=11+j*vecWid
                clr=cpick.to_rgba(omegaV[i,j]*25)    
                   
                pixelVecV.append(Rectangle(height=vecWid, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
        
            
        V_Matrix=VGroup(*pixelVecV)

        Vtext = MathTex(r"V^T", height=0.65).move_to([12.5, 4.15, 0])
        self.play(Create(Vtext), FadeIn(V_Matrix), lag_ratio=0.4, run_time=2)
        self.wait(2)        

class U_Matrix(MovingCameraScene): 
    def construct(self):

        #============18=============
        print("============18=============") 
        #Computes SVD        
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        omegaU=np.reshape(omegaU, [oShape[0], oShape[1], oShape[2]])

        pxWid=0.1
        left_x=omega.shape[0]*pxWid/2
        bottom_y=-5
        delta=0.11

        pxHei=0.0035
        vecWid=0.035
        topLeft=3.5

        #Builds the next columns of the matrix here
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-omegaMax/2,vmax=omegaMax/2)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])

        #============U Matrix=============
        pixelVecU=[]
        Modes=[]
        for n in range(0,omega.shape[2] ): #:# [0, 1, 2, 3, 74]:  #
            
            for i in range(0,omega.shape[0]):            
                for j in range(0,omega.shape[1]):
                    yVal=topLeft-(j+i*omega.shape[0])*pxHei
                    xVal=5+n*vecWid
                    clr=cpick.to_rgba(250*omegaU[i, j, n])       
                    pixelVecU.append(Rectangle(height=pxHei, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            Mode=VGroup(*pixelVecU)
            Modes.append(Mode)
            pixelVecU=[]
            print(n)
            
        U_Matrix=VGroup(*Modes)

        Utext = Text("U", font='CMU Serif', color=WHITE).move_to([6.25, 4.1, 0])
        Eqtext = Text("=", font='CMU Serif', color=WHITE).move_to([4.25, 4.1, 0])
        Eqtext2 = Text("=", font='CMU Serif', color=WHITE).move_to([4.25, 0, 0])

        self.play(self.camera.frame.animate.move_to([7,1,0]), run_time=0.1)
        self.add(U_Matrix)
        #self.play(Create(Utext), Create(Eqtext), Create(Eqtext2), FadeIn(U_Matrix), run_time=0.1)
        self.wait(1)   

        #============19=============
        print("============19=============") 
        #Transforms into the mode      
        # 
        U_Mode=[]
        for m in range(0,10):  
            pixel=[]
            for i in range(0,omegaU.shape[0]):            
                for j in range(0,omegaU.shape[1]):
                    yVal=0.5+left_x-i*delta
                    xVal=2+bottom_y+j*delta
                    if m==0:
                        sf=0.3
                    else:
                        sf=1
                    clr=cpick.to_rgba(sf*omegaS[m]*omegaU[i, j, m])       
                    pixel.append(Square(side_length=pxWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
                
            U_Mode.append(VGroup(*pixel))  

            self.play(Transform(Modes[m],U_Mode[m]),self.camera.frame.animate.move_to([3.5,0,0]),run_time=2)
            self.wait(1)
            
class S_Matrix(GraphScene, MovingCameraScene): 
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": 10},
                y_axis_config={"tick_frequency": 100},
                **kwargs
            )
    def setup(self):
        MovingCameraScene.setup(self)
        GraphScene.setup(self)

    def S_Diag_Function(self, x):
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        xx=math.floor(x)
        return omegaS[xx]

    def construct(self):

        #============20=============
        print("============20=============") 
        #Computes SVD        
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        omegaU=np.reshape(omegaU, [oShape[0], oShape[1], oShape[2]])

        pxWid=0.1
        left_x=omega.shape[0]*pxWid/2
        bottom_y=-5
        delta=0.11

        pxHei=0.0035
        vecWid=0.035
        topLeft=3.5

        #Builds the next columns of the matrix here
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-omegaMax/2,vmax=omegaMax/2)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])
    
        #Makes a vector
        pixelVec=[]
        pxHei=0.0035
        vecWid=0.035
        #============S Matrix=============
        pixelVecS=[]
        pixelVecDiag=[]
        for i in range(0,omegaS.shape[0]):            
            for j in range(0,omegaS.shape[0]):
                yVal=2.5-i*vecWid
                xVal=3+j*vecWid
                if i==j:
                    clr=cpick.to_rgba(omegaS[i]) 
                    pixelVecDiag.append(Rectangle(height=vecWid, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0])) 
                else:
                    clr=cpick.to_rgba(0)     
                    pixelVecS.append(Rectangle(height=vecWid, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))        
            
        S_Matrix=VGroup(*pixelVecS)
        S_Diag=VGroup(*pixelVecDiag)

        self.play(FadeIn(S_Matrix),FadeIn(S_Diag))
        self.wait(1)

        #============21============
        print("============21=============") 
        #============S Diagonal values=============
        self.graph_origin = [-5, -2.5, 0]
        self.x_axis_width = 6
        self.y_axis_height = 5
        self.x_min = 0
        self.x_max = omegaS.shape[0]-1
        self.y_min = 0
        self.y_max = omegaS[0]
        self.x_axis_label = "Mode \#"
        self.y_axis_label = "Energy"
        self.setup_axes(animate=True)

        EigenGraph = self.get_graph(self.S_Diag_Function).set_color(YELLOW)
        EigenRects = self.get_riemann_rectangles(EigenGraph, dx=1)

        self.play(Transform(S_Diag, EigenRects), lag_ratio=0.4, run_time=4)
        self.play(FadeOut(S_Matrix))
        self.wait(1)

class V_Matrix(GraphScene, MovingCameraScene): 
    def __init__(self, **kwargs):
        GraphScene.__init__(
                self,
                x_axis_config={"tick_frequency": 10},
                y_axis_config={"tick_frequency": .1},
                **kwargs
            )
    def setup(self):
        MovingCameraScene.setup(self)
        GraphScene.setup(self)

    def V_Matrix_Function(self, x):
        global v_index
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        xx=math.floor(x)
        #pdb.set_trace()
        return omegaV[v_index, xx]

    def construct(self):
        global v_index

        #============22=============
        print("============22=============") 
        #Computes SVD        
        oShape=omega.shape
        omegaAll=np.reshape(omega, [oShape[0]*oShape[1], oShape[2]])
        omegaU, omegaS, omegaV = np.linalg.svd(omegaAll, full_matrices=False)
        omegaU=np.reshape(omegaU, [oShape[0], oShape[1], oShape[2]])

        pxWid=0.1
        left_x=omega.shape[0]*pxWid/2
        bottom_y=-5
        delta=0.11

        pxHei=0.0035
        vecWid=0.035
        topLeft=3.5

        #Builds the next columns of the matrix here
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","w","r"])
        cnorm = mcol.Normalize(vmin=-omegaMax/2,vmax=omegaMax/2)
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
        cpick.set_array([])
    
        #Makes a vector
        pixelVec=[]
        pxHei=0.0035
        vecWid=0.035
        
        V_Matrix=[]
        for i in range(0,omegaV.shape[0]): 
            pixelVecV=[]           
            for j in range(0,omegaV.shape[1]):
                yVal=2.5-i*vecWid
                xVal=3+j*vecWid
                clr=cpick.to_rgba(omegaV[i,j]*25)    
                   
                pixelVecV.append(Rectangle(height=vecWid, width=vecWid, fill_color=rgb_to_color(clr[0:3]), fill_opacity=1.0, stroke_opacity=0).move_to([xVal,yVal,0]))
            
            V_Matrix.append(VGroup(*pixelVecV))
            
        AllVmatrix=VGroup(*V_Matrix)

        self.play(FadeIn(AllVmatrix))
        self.wait(1)

        #============23============
        print("============23=============") 
        self.graph_origin = [-5, 0, 0]
        self.x_axis_width = 6
        self.y_axis_height = 6
        #============V values=============
        for i in range(0,7):
            v_index=i
            self.x_min = 0
            self.x_max = omegaS.shape[0]-1            
            self.y_min = -np.max(omegaV[1,:])*1.1     
            self.y_max = np.max(omegaV[1,:])*1.1
            self.x_axis_label = "time"
            self.y_axis_label = "V Coefficient"
            self.setup_axes(animate=False)

            EigenGraph = self.get_graph(self.V_Matrix_Function).set_color(YELLOW)
            EigenRects = self.get_riemann_rectangles(EigenGraph, dx=1)

            self.play(Transform(V_Matrix[v_index], EigenRects), lag_ratio=0.4, run_time=4)        
            self.wait(1)
            self.play(FadeOut(V_Matrix[v_index]))







































        

        















    




        












              









        






























        
    

        

















        



























