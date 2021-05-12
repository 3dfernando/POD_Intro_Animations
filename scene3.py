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


class CrossCorrelationEigenvalues(Scene):
    def setup(self):
        Scene.setup(self)
    
    def construct(self):
        StartEquation = MathTex(r"A=U\Sigma V^T").move_to([0, 3, 0])
        StartEquationT = MathTex(r"A^T=(U\Sigma V^T)^T").move_to([9, 3, 0])


        self.play(Create(StartEquation))
        self.wait(1)
        self.add(StartEquationT)
        self.play(ApplyMethod(StartEquation.move_to,[-2, 3, 0]), ApplyMethod(StartEquationT.move_to,[2, 3, 0]), run_time=0.5)

        self.wait(1)
        StartEquationT2 = MathTex(r"A^T= (V^T)^T \Sigma^T U^T").move_to([2, 3, 0])
        self.play(Transform(StartEquationT,StartEquationT2))
        self.wait(1)
        StartEquationT3 = MathTex(r"A^T= V \Sigma U^T").move_to([2, 3, 0])
        self.play(Transform(StartEquationT,StartEquationT3))
        self.wait(1)
        
        StartEquation2 = MathTex(r"A=U\Sigma V^T").move_to([-2, 3, 0])
        StartEquationT4 = MathTex(r"A^T= V \Sigma U^T").move_to([2, 3, 0])

        
        #==================================================================
        StartEqns_Grp=VGroup(StartEquation2, StartEquationT4)

        RectEq1=Rectangle(height=.8, width=1.2).move_to([-6.3, 2, 0])
        RecText1=TextMobject("Temporal").scale(0.5)
        RecText2=TextMobject("Correlation Matrix").scale(0.5)
        RecTextAll=VGroup(RecText1, RecText2)
        RecTextAll.arrange_submobjects(DOWN, buff=0.15, aligned_edge=LEFT)
        RecTextAll.move_to([-5.8, 2.75, 0])
        LeftEquation1 = MathTex(r"A A^T=(U\Sigma V^T)(V \Sigma U^T)").move_to([-4, 2, 0])
        LeftEquation2 = MathTex(r"A A^T=U\Sigma V^T V \Sigma U^T").move_to([-4, 2, 0])
        LeftEquation3 = MathTex(r"A A^T=U\Sigma \underbrace{V^T V}_{I} \Sigma U^T").move_to([-4, 1.65, 0])


        self.play(Transform(StartEqns_Grp,LeftEquation1))
        self.wait(1)
        self.play(Create(RectEq1), Write(RecTextAll))
        self.wait(1)
        self.play(Transform(StartEqns_Grp,LeftEquation2), ApplyMethod(RectEq1.move_to,[-5.8, 2, 0]))
        self.wait(1)
        self.play(FadeOut(StartEqns_Grp),FadeIn(LeftEquation3), ApplyMethod(RectEq1.move_to,[-5.85, 2, 0]))
        self.wait(1)

        LeftEquation_Line2_1 = MathTex(r"A A^T=U\Sigma \Sigma U^T").move_to([-4, .5, 0])
        LeftEquation_Line2_2 = MathTex(r"A A^T=U\Sigma^2 U^T").move_to([-4, .5, 0])

        self.play(Transform(StartEqns_Grp,LeftEquation_Line2_1))
        self.wait(1)
        self.play(Transform(StartEqns_Grp,LeftEquation_Line2_2))
        self.wait(1)

        LeftEquation_Line3_1 = MathTex(r"A A^T U=U\Sigma^2 U^T U").move_to([-4, -.5, 0])
        LeftEquation_Line3_2 = MathTex(r"A A^T U=U\Sigma^2 \underbrace{U^T U}_{I}").move_to([-4, -0.85, 0])

        self.play(Transform(LeftEquation_Line2_2,LeftEquation_Line3_1))
        self.wait(1)
        self.play(FadeOut(LeftEquation_Line2_2),FadeIn(LeftEquation_Line3_2))
        self.wait(1)

        LeftEquation_Line4_1 = MathTex(r"A A^T U=U\Sigma^2").move_to([-4, -2, 0])
        self.play(Transform(LeftEquation_Line3_2.copy(),LeftEquation_Line4_1))
        self.wait(1)

        UnderBrace=BraceText(LeftEquation_Line4_1, "Eigenvalue Problem",label_scale=0.5)
        self.play(Create(UnderBrace))
        self.wait(1)

        #==================================================================
        #Second equation
        LineSeparator=Line(start=[0, 2.25, 0], end=[0, -3.75, 0], stroke_color=BLUE_A)
        self.play(Create(LineSeparator), ApplyMethod(StartEquationT.move_to,[9, 3, 0]), ApplyMethod(StartEquation.move_to,[-.5, 3, 0]))

        #Right set
        StartEqns_Grp2=VGroup(StartEquation2, StartEquationT4)

        RectEq1=Rectangle(height=.8, width=1.2).move_to([-6.15+8, 2, 0])
        RecText1=TextMobject("Spatial").scale(0.5)
        RecText2=TextMobject("Correlation Matrix").scale(0.5)
        RecTextAll=VGroup(RecText1, RecText2)
        RecTextAll.arrange_submobjects(DOWN, buff=0.15, aligned_edge=LEFT)
        RecTextAll.move_to([-5.8+8, 2.75, 0])
        LeftEquation1 = MathTex(r"A^T A=(V \Sigma U^T)(U\Sigma V^T)").move_to([4, 2, 0])
        LeftEquation2 = MathTex(r"A^T A=V \Sigma U^TU\Sigma V^T").move_to([4, 2, 0])
        LeftEquation3 = MathTex(r"A^T A=V\Sigma \underbrace{U^T U}_{I} \Sigma V^T").move_to([4, 1.65, 0])


        self.play(Create(LeftEquation1))
        self.wait(1)
        self.play(Create(RectEq1), Write(RecTextAll))
        self.wait(1)
        self.play(Transform(LeftEquation1,LeftEquation2), ApplyMethod(RectEq1.move_to,[-5.8+8, 2, 0]))
        self.wait(1)
        self.play(FadeOut(LeftEquation1),FadeIn(LeftEquation3), ApplyMethod(RectEq1.move_to,[-5.85+8, 2, 0]))
        self.wait(1)

        ###
        LeftEquation_Line2_1 = MathTex(r"A^T A=V\Sigma \Sigma V^T").move_to([4, .5, 0])
        LeftEquation_Line2_2 = MathTex(r"A^T A=V\Sigma^2 V^T").move_to([4, .5, 0])

        self.play(Transform(LeftEquation1,LeftEquation_Line2_1))
        self.wait(1)
        self.play(Transform(LeftEquation1,LeftEquation_Line2_2))
        self.wait(1)

        LeftEquation_Line3_1 = MathTex(r"A^T A V = V\Sigma^2 V^T V").move_to([4, -.5, 0])
        LeftEquation_Line3_2 = MathTex(r"A^T A V = V\Sigma^2 \underbrace{V^T V}_{I}").move_to([4, -0.85, 0])

        self.play(Transform(LeftEquation_Line2_2,LeftEquation_Line3_1))
        self.wait(1)
        self.play(FadeOut(LeftEquation_Line2_2),FadeIn(LeftEquation_Line3_2))
        self.wait(1)

        LeftEquation_Line4_1 = MathTex(r"A^T A V = V\Sigma^2").move_to([4, -2, 0])
        self.play(Transform(LeftEquation_Line3_2.copy(),LeftEquation_Line4_1))
        self.wait(1)

        UnderBrace=BraceText(LeftEquation_Line4_1, "Eigenvalue Problem",label_scale=0.5)
        self.play(Create(UnderBrace))
        self.wait(1)






























