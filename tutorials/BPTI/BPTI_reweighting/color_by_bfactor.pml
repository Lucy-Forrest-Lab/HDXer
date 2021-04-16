# Script to color BPTI structure by bfactor
load Reweighted-Predicted_byatom_120.0min.pdb, 5PTI 
hide everything
show cartoon, 5PTI
bg_color white
spectrum b, blue_white_red, minimum=-0.25, maximum=0.25
select no_dfracs, resi 1+2+3+4+8+9+11+13+15+26+39+40+46+49+50+57+58
color gray, no_dfracs
rotate x, -90
rotate y, 90
set ray_opaque_background, 0
set ray_shadows, 0
set specular, 0
set depth_cue, 0.1
set ray_trace_fog, 0
set orthoscopic, on
png Reweighted-Predicted_byatom_120.0min.png, width=900, height=1200, dpi=600, ray=1
