%Clear the workspace 
clear all;
%Close all opened figures
close all;
%Clear the command  window
clc;
%Load the extaced feature file for object classification
load fat2.mat

% Object for reading video file.
filename = 'last.mp4';
hVidReader = vision.VideoFileReader(filename, 'ImageColorSpace', 'RGB',...
                              'VideoOutputDataType', 'single');
%Optical flow object for estimating direction and speed of object motion
hOpticalFlow = vision.OpticalFlow( ...
    'OutputValue', 'Horizontal and vertical components in complex form', ...
    'ReferenceFrameDelay', 3);
%Create two objects for analyzing optical flow vectors.

hMean1 = vision.Mean;
% calculate mean over a sequance of inputs 
hMean2 = vision.Mean('RunningMean', true);
%Filter object for removing speckle noise(multiplicative noise to the image I)
%introduced during segmentation
hMedianFilt = vision.MedianFilter;
%Morphological closing object for filling holes in blobs.
%creates a linear structuring element that is symmetric with respect to the 
%strel() neighborhood center. deg specifies the angle (in degrees) of the line as
% measured in a counterclockwise direction from the horizontal axis. 
%len is approximately the distance between the centers of the structuring 
%element members at opposite ends of the line.
hclose = vision.MorphologicalClose('Neighborhood', strel('line',5,45));
%Create a blob analysis System object to segment cars in the video.
hblob = vision.BlobAnalysis(...
    'CentroidOutputPort', false, 'AreaOutputPort', true, ...
    'BoundingBoxOutputPort', true, 'OutputDataType', 'double', ...
    'MinimumBlobArea', 250, 'MaximumBlobArea', 3600, 'MaximumCount', 80);
%Morphological erosion object for removing portions of the road and other unwanted objects.
herode = vision.MorphologicalErode('Neighborhood', strel('square',2));
%Create objects for drawing the bounding boxes and motion vectors
hshapeins1 = vision.ShapeInserter('BorderColor', 'Custom', ...
                                  'CustomBorderColor', [0 1 0]);
hshapeins2 = vision.ShapeInserter('BorderColor', 'Custom', ...
                                  'CustomBorderColor', [1 0 0]);
hshapeins3 = vision.ShapeInserter('BorderColor', 'Custom', ...
                                  'CustomBorderColor', [0 0 1]);

% This object will write the number of tracked cars in the output image
htextins = vision.TextInserter('Text', '%4d', 'Location',  [1 1], ...
                               'Color', [0 1 0], 'FontSize', 12);
htextins1 = vision.TextInserter('Text', '%4d', 'Location',  [10 10], ...
                               'Color', [1 0 0], 'FontSize', 22);
%Create System objects to display the original video, motion vector video, the thresholded video and the final result.
sz = get(0,'ScreenSize');
%Q1.What sz(4) will do here?????????
pos = [20 sz(4)-300 200 200];

hVideo1 = vision.VideoPlayer('Name','Original Video','Position',pos);
pos(1) = pos(1)+220; % move the next viewer to the right
hVideo4 = vision.VideoPlayer('Name','Results','Position',pos);

% Initialize variables used in Creating ROI and detection of vehicals.

line2 = 100;       %y-position of ROI
lineRowy = 100;    %x-Position of ROI 
line2y = 240 ;     %x-Position of ROI
count = 0;         % for counting number of frame processed
count1 =0;         % For counting the result displayed for one object 
alto_no = 0;       %for counting of Alto cars
s_no =0;           %for counting of Swift cars
c_no = 0;          %for counting of Honda City cars
t_no = 0;          %for counting of Trucks
text='';           % String for storing label of predicted vehicals  
k=1;               %for counting of number of cell formed 
curLocation = 'C:\Users\lenovo\Desktop\matlab\Ajay'; % get current working directory location
%Creat figure for displaying Result and original video.
figure;
%Maximize it to full screen. 
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); 
%Create the processing loop to track the cars in video
while ~isDone(hVidReader)  % Stop when end of file is reached
    frame  = step(hVidReader);  % Read input video frame
    count = count+1;  % Increase the count of  no. of frame 
    %frame = imrotate(frame,-90,'bilinear');
    %frame = imresize(frame,0.8);
    grayFrame = rgb2gray(frame);
    ofVectors = step(hOpticalFlow, grayFrame);   % Estimate optical flow
    % Show the Original video
    subplot(1,3,1);
    imshow(frame);
    caption = sprintf('Showing Frame No : %4d ', count);
    title(caption);
    drawnow;
    % The optical flow vectors are stored as complex numbers. Compute their
    % magnitude squared which will later be used for thresholding.
    y1 = ofVectors .* conj(ofVectors);
    % Compute the velocity threshold from the matrix of complex velocities.
    vel_th = 0.5 * step(hMean2, step(hMean1, y1));

    % Threshold the image and then filter it to remove speckle noise.
    segmentedObjects = step(hMedianFilt, y1 >= vel_th);

    % Thin-out the parts of the road and fill holes in the blobs.
    %First Apply Eroding nd then Morphological Closing
    segmentedObjects = step(hclose, step(herode, segmentedObjects));

    % Estimate the area and bounding box of the blobs.
    [area, bbox] = step(hblob, segmentedObjects);
    % Select boxes inside ROI (below white line).
     X1 = ( bbox(:,2) > line2);
     Y1=  (180 > bbox(:,1)) & (bbox(:,1) > 10);
     Idx = X1&Y1;

    % Based on blob sizes, filter out objects which can not be cars.
    % When the ratio between the area of the blob and the area of the
    % bounding box is above 0.4 (40%), classify it as a car.
    ratio = zeros(length(Idx), 1);
    ratio(Idx) = single(area(Idx,1))./single(bbox(Idx,3).*bbox(Idx,4)); %Q2.How Ratio is Calculated?????
    ratiob = (0.2 < ratio) & (0.75 > ratio);     % for all objects
    ratiob1 = ratio >= 0.75;  
   
    a = length(ratiob);  
    b = length(ratiob1);
    c = length(Idx); 
    %Rejecting unwanted bboxes    
    bbox(~ratiob, :) = int32(-1);
    % Draw bounding boxes around the tracked cars.
    y2 = frame;
    nbbox =length(bbox(:,1));
    %cell([1000 1]);
    for i=1:nbbox          %Loop through All bboxes
          if(Idx(i))          %if they lies in ROI
              size = bbox(i,:); %Get size of bbox
              %cell{k,1}= bbox(i,:);
              %k = k+1;
              % Crop the bbox
              img = imcrop(frame,[size(1) size(2) size(3) size(4)]); 
              %img = imresize(img,6);
              if(~isempty(img))   % If croped image is not empty then 
                         
                    cell{1,1}=[0 0 0 0] % initialize a cell 
                    cell{k+1,1}= bbox(i,:);%Enter values in cell (dimention of bbox )
                
                    
                    b = [cell{k,1}];  %Get position of previousaly formed bbox 
                    d = [cell{k+1,1}]; %get position of newly formed bbox 
                             
                    %Condition for Observation of detected vehical
                    if (d(2)> 80)& (d(2) < 110)   
           
                        if (d(2)-b(2))>=0   % Identify that the detected vehical is same or not if yes.. 
                 
                            if ratio(i) < 0.75  % Condition for car 
                                %Predict the category of vehical
                                [labelIdx, scores] = predict(categoryClassifier, img); 
                                text =  categoryClassifier.Labels(labelIdx);
                                displayEndOfDemoMessage(mfilename);
                            else  % the vehical will be truck
                                text = 'truck';
                            end 
                            % store the string label
                            cell{1,2} = 'default';%Give defoult value to cell's first entry
                            cell{k+1,2}= text;%Enter newly obtained value of text
                            %k = k+1;
                            %check detected vehical lies in which category
                            if strcmp('alto',cell{k+1,2})
                                alto_no = alto_no + 1;     %if alto increase count of alto 
                            end
                            if strcmp('swift',cell{k+1,2})
                                s_no = s_no + 1;%increase count of swift
                            end
                            if strcmp('City',cell{k+1,2})
                                c_no = c_no + 1;%increase count of city
                            end
                            if strcmp('truck',cell{k+1,2})
                                t_no = t_no+1;
                            end    
                            %Find maximum times detected class for that vehical 
                            max_no =[alto_no;s_no;c_no;t_no];
                            [M,I]= max(max_no(:));
                            % Assigne the name of class that is detected most of
                            % time 
                            if(I==1)
                                text = 'Alto';
                            end
                            if(I==2)
                                text ='Swift';
                            end
                            if(I==3)
                                text = 'city';
                            end
                            if(I == 4)
                                text = 'truck';
                            end   
                  
                            %if the new object entered then reset the count variable 
                            %for new vehical
                        else                  
                            alto_no = 0;
                            s_no = 0;
                            c_no = 0;
                            t_no = 0;
                            
                        end
                    end
                   
                   if (b(2)>110) & (b(2)<180)
                       count1 = count1 + 1;   
                       % insert Annotation
                          y2 = insertObjectAnnotation(frame,'rectangle',bbox(i,:),text,'FontSize',17);
                          if count1 == 2
                          %show the croped vehical image
                          subplot(1,3,3);
                          imshow(img);  
                          title('Detected vehical');
                          % Store Croped image to database
                          if strcmp('alto',text)
                              %location = fullfile(curLocation,'cAlto');
                              name = sprintf('C:\\Users\\lenovo\\Desktop\\matlab\\Ajay\\cAlto\\%d',count)
                              imwrite(img,name,'jpg');
                          end
                          if strcmp('City',text)
                               %location = fullfile('curLocation','cCity');
                               name = sprintf('C:\\Users\\lenovo\\Desktop\\matlab\\Ajay\\cCity\\%d',count)
                               imwrite(img,name,'jpg');
                               imwrite();
                          end
                          if strcmp ('Swift',text)
                              %location = fullfile('curLocation','cSuzuki');
                              name = sprintf('C:\\Anup\\%d',count)
                              imwrite(img,name,'jpg');
                          end    
                          end
                   end
                   if b(2) > 170   % if object goes out of our ROI 
                       count1 = 0; % clear the counter
                   end
                       k=k+1;
                      
            end   %End of condition if croped bbox not contain image
            
        end   %End of ROI region condition
     end      %End of loop for nbbox
     
    %Draw the ROI 
    y2(150:152,100:240,:) = 1;   % The white line.
    y2(50:52,100:240,:)   = 1;   % The white line.
    y2(50:150,100:102,:)  = 1;   % The white line.
    y2(50:150,240:242,:)  = 1;   % The white line.
    %% show the result .
    subplot(1,3,2);
    imshow(y2);
    title('Detected vehicals');
    
end
release(hVidReader);