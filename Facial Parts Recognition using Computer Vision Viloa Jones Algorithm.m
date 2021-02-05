
clc;clear;
close all;
camObj = webcam();

% Preview a stream of image frames.
preview(camObj);

% Acquire and display a single image frame.

% 'FrontalFaceCART'
% 'FrontalFaceLBP'
% 'ProfileFace'
% 'Mouth'
% 'Nose'
% 'EyePairBig'
% 'EyePairSmall'
% 'RightEye'
% 'LeftEye'
% 'RightEyeCART'
% 'LeftEyeCART'
% 'UpperBody'
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
eyeDetector = vision.CascadeObjectDetector('EyePairBig');% Default: finds faces
while(1)
img = snapshot(camObj);
bboxes = step(faceDetector, img);
% Detect faces
bboxeseye = step(eyeDetector, img);
% Annotate detected faces
IFaces = insertObjectAnnotation(img, 'rectangle', bboxes, 'Face');
hold on;
IFaces = insertObjectAnnotation(IFaces, 'rectangle', bboxeseye, 'Eye');
imshow(IFaces), title('Detected faces');
end
